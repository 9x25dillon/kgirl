"""Reflection: turn what was said into what is remembered.

After every few exchanges (and when a conversation ends) the reflector reads the turns since its last
pass, together with the rolling summary, and asks the model for a small JSON record:

    summary      the conversation so far, rewritten (this is what later prompts carry)
    title        a short name for the thread
    threads      open questions and unfinished tasks
    facts        durable facts about the user, their projects and their machine
    preferences  how the user wants kgirl to work
    lessons      what helped in kgirl's own answers
    mistakes     what the user had to correct, stated as what to avoid
    feedback     for each answer the user reacted to: approved, corrected, or neither

How it becomes habit (and why it cannot run away): facts, preferences, lessons and mistakes are *proposed*
to the Curator with source "reflection". A proposal is staged; an independent later observation of the same
thing reinforces it; the curator promotes it only once support >= 2 and its utility lower bound clears
the policy, and decays and retires it when it stops helping. One conversation can suggest a habit; only
repetition makes it one. Anything the user states explicitly ("remember that ...") is stored active at once.

Feedback closes the loop: the memory fragments an answer was built from are credited with a win when the
user approves the answer and a loss when the user corrects it, so useful memories rise in recall and
misleading ones sink. There are three sources, and an answer is judged once:

    command     /good and /bad: the user's explicit word (weight 1, and it overrides the other two)
    reaction    the opening words of the user's next message, read by `quick_feedback` (weight 1).
                Only a reaction addressed to the answer counts: "thanks, that works", "no, that's wrong".
                A message that merely mentions failure ("the test fails when ...") is a new request, not
                a verdict, and credits nothing.
    reflection  the model's reading of the excerpt, for answers the first two left unjudged (weight 0.5,
                because a small model's reading is weaker evidence than the user's own words)
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable

from ..soup import Curator, Soup
from .conversations import ConversationStore, Turn
from .text import first_json, one_line

REFLECT_SYSTEM = """You maintain the long-term memory of kgirl, a local coding and writing assistant.
Read the conversation excerpt and the previous summary, then return ONLY one JSON object:
{"summary": "<the whole conversation so far in <= 120 words: goals, decisions, results, current state>",
 "title": "<3-7 word title>",
 "threads": ["<open question or unfinished task>", ...],
 "facts": ["<durable fact about the user, their projects, tools or machine>", ...],
 "preferences": ["<how the user wants the assistant to behave or write>", ...],
 "lessons": ["<a reusable lesson about what worked in the assistant's answers>", ...],
 "mistakes": ["<something the assistant got wrong that the user corrected, stated as what to avoid>", ...],
 "feedback": {"A1": "positive" | "negative" | "none", ...}}
Rules: only include what the excerpt actually shows; no guesses. Each item is one self-contained sentence
that still makes sense months later (name the project, no "this" or "it"). Empty lists are fine.
"feedback" has one entry for each ASSISTANT [An] answer that a USER message follows: "positive" if that
next message accepts or thanks, "negative" if it corrects the answer or says it is wrong or did not work,
"none" for anything else, including a new or follow-up request."""

FEEDBACK_SCOPE = 240      # a verdict is in the opening of a message, not somewhere in a pasted log

# Interjections that may come before the verdict ("ok thanks", "sorry, that didn't work").
_LEAD = (r"^\W*(?:(?:ok(?:ay)?|oh|ah|hm+|well|yes|yeah|yep|yup|cool|alright|so|and|actually|sorry|stop|wait|"
         r"hold\s+on|hang\s+on|whoa|ugh)\b[\s,.!:;-]*)*")
# One-word praise counts only when it stands alone or leads into the answer: "great, now ...", "perfect thanks",
# "awesome that works". "Nice to meet you" and "good morning" are not verdicts.
_AFTER_PRAISE = r"(?=\s*(?:[,.!;:)]|$|\s+(?:thanks|thank|work|job|that|this|it|and|now|so|what\s+i)\b))"
_POS = re.compile(
    _LEAD + r"(?:thanks\b|thank\s+you\b|thx\b|tysm\b|ty\b|cheers\b"
    r"|(?:perfect|great|awesome|excellent|brilliant|fantastic|good|nice|exactly|lovely|beautiful|sweet|"
    r"spot\s+on|well\s+done|good\s+(?:job|work|one|call|answer)|nice\s+(?:job|work|one)|love\s+it|nailed\s+it)"
    + _AFTER_PRAISE +
    r"|correct(?=\s*(?:[,.!;:)]|$))"                 # "Correct." is a verdict; "correct this function" is a request
    r"|(?:that|this|it)(?:'s|\s+is|\s+was|\s+looks|\s+seems)?\s+"
    r"(?:right|correct|perfect|great|good|exactly|it|what\s+i\s+(?:wanted|needed|meant|asked))\b"
    r"|(?:that|this|it)\s+(?:works|worked|helps|helped|fixed\s+it|did\s+it|solved\s+it|makes\s+sense)\b"
    r"|works(?:\s+(?:now|great|perfectly|fine|well))?(?=\s*(?:[,.!;:)]|$))"
    r"|worked(?=\s*(?:[,.!;:)]|$)))", re.I)
_NEG = re.compile(
    _LEAD + r"(?:no(?=\s*(?:[,.!;:]|$)|\s+(?:that|this|it|you|not|i\s+(?:meant|said|asked|want(?:ed)?))\b)"
    r"|nope\b|nah\b|wrong\b|incorrect\b"
    r"|not\s+(?:quite|really|right|correct|what\s+i|working|true)\b"
    r"|(?:that|this|it)(?:'s|\s+is|\s+was|\s+looks|\s+seems)\s+(?:wrong|incorrect|broken|still)\b"
    r"|(?:that|this|it)(?:'s|\s+is|\s+was)\s+not\s+(?:right|correct|true|what|working|quite|it|the|how)\b"
    r"|(?:that|this|it)\s+(?:isn'?t|wasn'?t)\s+(?:right|correct|true|what|working|quite|it|the|how)\b"
    r"|(?:that|this|it)\s+(?:doesn'?t|didn'?t|does\s+not|did\s+not|won'?t|will\s+not|can'?t|cannot)\s+"
    r"(?:work|help|compile|run|build|pass|fix|solve|answer|match|handle|do)\b"
    r"|(?:that|this|it)\s+still\b"
    r"|still\s+(?:broken|wrong|failing|fails|not|doesn'?t|the\s+same|getting|seeing|errors?)\b"
    r"|you\s+(?:misunderstood|missed|got\s+it\s+wrong|are\s+wrong|were\s+wrong|didn'?t|did\s+not|forgot|ignored|broke)\b"
    r"|you'?re\s+wrong\b|try\s+again\b|didn'?t\s+work\b|doesn'?t\s+work\b|does\s+not\s+work\b|did\s+not\s+work\b)",
    re.I)
_CONTRAST = re.compile(r"\b(?:but|however|though|although|except|unfortunately)\b(.*)", re.I | re.S)
_COMPLAINT = re.compile(r"\b(?:still|wrong|incorrect|broken|fails?|failing|failed|errors?|crash(?:es|ed)?|missing|"
                        r"missed|doesn'?t|didn'?t|isn'?t|won'?t|not\s+(?:work|right|correct|quite|what))\b", re.I)
_SENTENCE_END = re.compile(r"[.!?\n]")

_REMEMBER = re.compile(
    r"^\s*(?:(?:please|pls)\s+|kgirl[,:]?\s+)*"
    r"(?:(?:remember|keep\s+in\s+mind|note|don'?t\s+forget)(?:\s+that\b|\s*[:,])\s*"
    r"|(?P<todo>remember\s+(?=to\b))"
    r"|(?P<standing>from\s+now\s+on|in\s+future|for\s+future\s+reference|going\s+forward)\s*[:,]?\s*)"
    r"(?P<text>.+)$", re.I | re.S)
_PREFERENCE = re.compile(r"\b(?:i\s+(?:prefer|like|want|hate|need)|always|never|don'?t|do\s+not|please)\b", re.I)

LLM = Callable[[str, str, int, float], str]   # (system, prompt, max_tokens, temperature) -> text


def quick_feedback(text: str) -> str:
    """The user's verdict on the previous answer, read from the opening of their next message.

    Returns "positive", "negative" or "none". Deliberately narrow: it must be a reaction addressed to the
    answer, at the start of the message, and not a question. Praise followed in the same sentence by a
    complaint ("great, but it still fails on empty input") is negative: the answer did not do the job.
    Praise, then a complaint in a later sentence, is left unjudged.
    """
    head = (text or "")[:FEEDBACK_SCOPE].replace("’", "'")
    first = re.split(r"(?<=[.!?])\s|\n", head.strip(), maxsplit=1)[0]
    if first.rstrip().endswith("?"):
        return "none"
    if _NEG.match(head):
        return "negative"
    praise = _POS.match(head)
    if not praise:
        return "none"
    tail = _CONTRAST.search(head, praise.end())
    if tail and _COMPLAINT.search(tail.group(1)):
        return "none" if _SENTENCE_END.search(head, praise.end(), tail.start()) else "negative"
    return "positive"


def explicit_memory(text: str) -> tuple[str, str] | None:
    """What the user asked kgirl to remember, as (text, kind), or None.

    "remember that the staging box is 10.0.0.7" -> ("the staging box is 10.0.0.7", "fact")
    "from now on, answer in British English"    -> ("answer in British English", "preference")
    A question ("remember when we fixed the parser?") is not an instruction to remember.
    """
    m = _REMEMBER.match(text or "")
    if not m:
        return None
    said = " ".join(m.group("text").split())
    if not said or said.endswith("?"):
        return None
    if m.group("todo"):
        said = said[3:].strip()                       # "remember to run the tests first" -> "run the tests first"
        return (said, "preference") if said else None
    kind = "preference" if m.group("standing") or _PREFERENCE.search(said) else "fact"
    return said, kind


@dataclass
class Reflection:
    summary: str = ""
    title: str = ""
    threads: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)       # answer label ("A2", or "LAST") -> verdict
    proposed: list[tuple[int, str]] = field(default_factory=list)
    credited: list[int] = field(default_factory=list)            # ids of the answers this reflection judged

    @property
    def feedback(self) -> str:
        """The newest explicit verdict in this reflection, or "none"."""
        return list(self.verdicts.values())[-1] if self.verdicts else "none"

    @classmethod
    def parse(cls, obj) -> "Reflection":
        if not isinstance(obj, dict):
            return cls()

        def items(key: str) -> list[str]:
            v = obj.get(key) or []
            v = v if isinstance(v, list) else [v]
            return [one_line(str(x), 280) for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:6]

        fb = obj.get("feedback")
        verdicts: dict[str, str] = {}
        if isinstance(fb, dict):
            for label, verdict in fb.items():
                verdict = str(verdict).strip().lower()
                if verdict in ("positive", "negative"):
                    verdicts[str(label).strip().upper()] = verdict
        elif isinstance(fb, str) and fb.strip().lower() in ("positive", "negative"):
            verdicts["LAST"] = fb.strip().lower()     # the older single-verdict form: the last answer reacted to
        return cls(one_line(str(obj.get("summary", "")), 1200), one_line(str(obj.get("title", "")), 80),
                   items("threads"), items("facts"), items("preferences"), items("lessons"), items("mistakes"),
                   verdicts)


class Reflector:
    def __init__(self, llm: LLM, store: ConversationStore, soup: Soup, curator: Curator, every: int = 3):
        self.llm = llm
        self.store = store
        self.soup = soup
        self.curator = curator
        self.every = every
        self._lock = threading.Lock()            # one reflection at a time (it holds the lock across a model call)
        self._credit_lock = threading.Lock()     # check-and-set of an answer's verdict; never held across a call

    def due(self, conv_id: int) -> bool:
        conv = self.store.get(conv_id)
        if not conv:
            return False
        fresh = self.store.turns(conv_id, since_id=conv.reflected_turn, roles=("user",))
        return len(fresh) >= self.every

    # ------------------------------------------------------------------ feedback

    def credit_answer(self, conv_id: int, verdict: str, source: str = "reaction", weight: float = 1.0,
                      turn: Turn | None = None, override: bool = False) -> list[int] | None:
        """Record a verdict on an answer (default: the newest one) and credit the memory it was built from.

        Returns the credited fragment ids (possibly empty), or None when nothing was recorded: no answer
        yet, or the answer already has a verdict and `override` is false.
        """
        if verdict not in ("positive", "negative"):
            return None
        with self._credit_lock:
            answer = self.store.turn(turn.id) if turn is not None else self.store.last_answer(conv_id)
            if answer is None or not answer.is_answer or (answer.verdict and not override):
                return None
            ids = [i for i in answer.recalled if isinstance(i, int)]
            self.soup.credit(ids, success=verdict == "positive", weight=weight)
            self.store.set_meta(answer.id, feedback={"verdict": verdict, "source": source, "weight": weight})
            return ids

    # ------------------------------------------------------------------ reflection

    def reflect(self, conv_id: int, force: bool = False) -> Reflection | None:
        with self._lock:
            conv = self.store.get(conv_id)
            if not conv:
                return None
            turns = [t for t in self.store.turns(conv_id, since_id=conv.reflected_turn, roles=("user", "assistant"))
                     if not t.partial]
            if not turns or (not force and sum(t.role == "user" for t in turns) < self.every):
                return None
            shown = list(turns)
            if shown[0].role == "user":      # show the answer that message reacts to, so its verdict can be read
                prior = self.store.last_answer(conv_id, before_id=shown[0].id)
                if prior is not None and not prior.verdict:
                    shown.insert(0, prior)
            excerpt, labels, reacted = render_excerpt(shown, 6000)
            prompt = (f"PREVIOUS SUMMARY:\n{conv.summary or '(none: new conversation)'}\n\n"
                      f"CONVERSATION EXCERPT:\n{excerpt}")
            try:
                raw = self.llm(REFLECT_SYSTEM, prompt, 900, 0.1)
            except Exception:  # noqa: BLE001 - reflection must never break a chat
                return None
            ref = Reflection.parse(first_json(raw))
            if not ref.summary:
                return None
            self.store.update(conv_id, summary=ref.summary, title=conv.title or ref.title or None,
                              threads=ref.threads, reflected_turn=turns[-1].id)
            self._episode(conv_id, ref, conv.scope)
            src = f"reflection:conv{conv_id}"
            # Who the user is and what they want holds everywhere, so facts and preferences are proposed
            # globally (swarms recall them too). What worked or failed in answers stays with the project
            # the conversation was about.
            for kind, texts, scope in (("fact", ref.facts, "*"), ("preference", ref.preferences, "*"),
                                       ("abstraction", ref.lessons, conv.scope),
                                       ("antipattern", ref.mistakes, conv.scope)):
                for text in texts:
                    fid, how = self.curator.propose(text, kind=kind, tags="conversation", source=src, scope=scope)
                    ref.proposed.append((fid, how))
            for label, verdict in ref.verdicts.items():
                if label == "LAST":
                    label = reacted[-1] if reacted else ""
                answer = self.store.turn(labels[label]) if label in labels and label in reacted else None
                if answer is None:
                    continue                 # an unknown label, or an answer nobody has reacted to yet
                if self.credit_answer(conv_id, verdict, source="reflection", weight=0.5, turn=answer) is not None:
                    ref.credited.append(answer.id)
            return ref

    def _episode(self, conv_id: int, ref: Reflection, scope: str) -> int:
        """One active `episode` fragment per conversation, replaced as the summary grows."""
        marker = episode_marker(conv_id)
        title = f"{ref.title}: " if ref.title else ""
        text = " ".join(f"conversation #{conv_id} {title}{ref.summary}".split())
        rows = self.soup.db.execute("SELECT id, text FROM fragments WHERE kind='episode' AND data=? AND status='active'",
                                    (marker,)).fetchall()
        for row in rows:
            if row["text"] == text:
                return row["id"]
            self.soup.set_status(row["id"], "retired", "superseded by a newer summary")
        fid = self.soup.add(text, kind="episode", tags="conversation", source="conversation", scope=scope, data=marker)
        self.soup.set_status(fid, "active", "episode summary")   # an identical retired summary comes back
        return fid


def episode_marker(conv_id: int) -> str:
    """The `data` value that ties an episode fragment to its conversation."""
    return f'{{"conv": {conv_id}}}'


def render_excerpt(turns: list[Turn], budget_chars: int) -> tuple[str, dict[str, int], list[str]]:
    """The newest turns that fit the budget, with answers labelled A1, A2, ...

    Returns (text, {label: answer turn id}, labels of answers that a user message follows).
    """
    picked: list[tuple[Turn, str]] = []
    used = 0
    for t in reversed(turns):
        body = t.content if len(t.content) < 1600 else t.content[:1500] + " …[cut]"
        if picked and used + len(body) > budget_chars:
            break
        picked.append((t, body))
        used += len(body)
    picked.reverse()
    lines: list[str] = []
    labels: dict[str, int] = {}
    reacted: list[str] = []
    open_label = ""
    for t, body in picked:
        if t.role == "user":
            if open_label:
                reacted.append(open_label)
                open_label = ""
            lines.append(f"USER: {body}")
        else:
            open_label = f"A{len(labels) + 1}"
            labels[open_label] = t.id
            lines.append(f"ASSISTANT [{open_label}]: {body}")
    return "\n\n".join(lines), labels, reacted
