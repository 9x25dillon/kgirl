"""Long-form writing: outline, draft by section, critique, revise, and pick up where it stopped.

A small model writing 1,500 words in one pass drifts, repeats itself and forgets the brief. The writer
keeps each call small and focused:

    outline  brief + what kgirl remembers (the user's writing style, relevant facts, optional repo
             context) -> a JSON outline: title, audience, thesis, sections with their points and length
    draft    one call per section, each given the outline, the brief, what the earlier sections already
             covered and the tail of the previous section
    critique one call that lists concrete problems against the brief (missing points, repetition,
             unsupported claims, tone that breaks the user's style). With a length target, a section far
             off its share of it is also an issue (found by counting, not asked of the model)
    revise   per section, only where an issue names that section

STYLE is the user's voice: the profile `evolve` keeps and the writing preferences recalled for the brief
(`style_guide` builds it). It goes into every call, apart from CONTEXT (facts to use, not to imitate).

A draft is a plain record (`Draft.to_dict`). The writer hands it to `checkpoint` after every step, so the
caller can keep it as a conversation artifact, and `write(resume=draft)` continues from the step it stopped
at: no section is drafted twice and no revision is repeated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .text import first_json, one_line, strip_reasoning

LLM = Callable[[str, str, int, float], str]

STAGES = ("outline", "drafting", "critique", "revising", "done")
MAX_SECTIONS = 7
LONG, SHORT = 1.5, 0.5          # a section is flagged beyond these multiples of its target (counted in words)

PLAN_SYSTEM = """You plan a piece of writing. Return ONLY JSON:
{"title": "...", "audience": "...", "thesis": "<one sentence>",
 "sections": [{"heading": "...", "points": ["...", "..."], "words": 150}, ...]}
3 to 7 sections. Points are concrete (names, numbers, examples), not topics. Respect the user's
style and preferences and use the facts given; never invent facts that are not in the brief or the context."""

DRAFT_SYSTEM = """You write one section of a longer piece. Write only this section's body in Markdown
(no heading line). Cover every point, in order, with concrete detail. Match the requested length within
20%. Keep the voice consistent with the previous section and with the user's STYLE when one is given. Do
not repeat what earlier sections already covered. Do not summarize the whole piece or add a conclusion
unless this is the last section."""

CRITIC_SYSTEM = """You are a demanding editor. Compare the draft with the brief and the outline. Return ONLY
JSON: {"issues": [{"section": "<heading>", "problem": "...", "fix": "..."}, ...]}
List at most 8 issues, most important first: missing or wrong content, repetition, vague claims, broken
logic between sections, tone that conflicts with the user's style. An empty list is fine."""

REVISE_SYSTEM = """Revise this section of a longer piece to fix the listed issues. Keep everything that is
not affected. Return only the revised section body in Markdown, no heading line."""


@dataclass
class Draft:
    title: str
    outline: dict
    sections: list[tuple[str, str]] = field(default_factory=list)       # (heading, body) drafted so far
    issues: list[dict] = field(default_factory=list)
    recalled: list[int] = field(default_factory=list)                   # memory the style and context came from
    brief: str = ""
    context: str = ""
    style: str = ""
    words: int | None = None                                            # the total length asked for
    revise: bool = True
    plan: list[dict] = field(default_factory=list)                      # the outline's sections, with length targets
    stage: str = "outline"
    critiqued: bool = False
    revised: list[str] = field(default_factory=list)                    # headings already revised

    def markdown(self) -> str:
        body = "\n\n".join(f"## {h}\n\n{t.strip()}" for h, t in self.sections)
        return f"# {self.title}\n\n{body}\n"

    @property
    def done(self) -> bool:
        return self.stage == "done"

    def progress(self) -> str:
        """One line for a list of drafts: where it stands."""
        drafted = f"{len(self.sections)}/{len(self.plan)} sections" if self.plan else "no outline yet"
        return f"{self.stage}: {drafted}" if not self.done else f"done: {len(self.sections)} sections"

    def to_dict(self) -> dict:
        return {"title": self.title, "outline": self.outline, "sections": [[h, t] for h, t in self.sections],
                "issues": self.issues, "recalled": self.recalled, "brief": self.brief, "context": self.context,
                "style": self.style, "words": self.words, "revise": self.revise, "plan": self.plan,
                "stage": self.stage, "critiqued": self.critiqued, "revised": self.revised}

    @classmethod
    def from_dict(cls, d: dict) -> "Draft":
        def listed(key: str, kind) -> list:
            v = d.get(key)
            return [x for x in v if isinstance(x, kind)] if isinstance(v, list) else []

        secs = [(str(x[0]), str(x[1])) for x in listed("sections", (list, tuple)) if len(x) == 2]
        words = d.get("words")
        return cls(str(d.get("title") or "Untitled"), d.get("outline") if isinstance(d.get("outline"), dict) else {},
                   secs, listed("issues", dict), [i for i in listed("recalled", int)], str(d.get("brief") or ""),
                   str(d.get("context") or ""), str(d.get("style") or ""),
                   words if isinstance(words, int) and words > 0 else None, bool(d.get("revise", True)),
                   listed("plan", dict), d.get("stage") if d.get("stage") in STAGES else "outline",
                   bool(d.get("critiqued")), [str(h) for h in listed("revised", str)])


class Writer:
    def __init__(self, llm: LLM, emit: Callable[[str, dict], None] | None = None):
        self.llm = llm
        self.emit = emit or (lambda topic, payload: None)

    def write(self, brief: str, context: str = "", words: int | None = None, revise: bool = True,
              progress: Callable[[str], None] | None = None, style: str = "", resume: Draft | None = None,
              checkpoint: Callable[[Draft], None] | None = None) -> Draft:
        """Write the piece. With `resume`, continue that draft: its brief, context, style, length target and
        revise setting are the ones it was started with, and `brief` and the other arguments are ignored."""
        say = progress or (lambda msg: None)
        d = resume or Draft("", {}, brief=brief, context=context, style=style, words=words, revise=revise)
        save = checkpoint or (lambda draft: None)
        brief, words, revise = d.brief, d.words, d.revise
        ctx = f"\n\nCONTEXT (what kgirl knows; use, don't contradict):\n{d.context}" if d.context else ""
        sty = f"\n\nSTYLE (the user's; write in it):\n{d.style}" if d.style else ""
        if d.done:
            return d

        if not d.plan:
            target = f"\nTarget length: about {words} words in total." if words else ""
            say("planning")
            outline = first_json(self.llm(PLAN_SYSTEM, f"BRIEF:\n{brief}{target}{ctx}{sty}", 1200, 0.3))
            if not isinstance(outline, dict) or not outline.get("sections"):
                outline = {"title": brief[:60], "thesis": brief, "sections": [{"heading": "Draft", "points": [brief],
                                                                                "words": words or 500}]}
            secs = [s for s in outline["sections"] if isinstance(s, dict) and s.get("heading")][:MAX_SECTIONS]
            _set_targets(secs, words)
            d.title, d.outline, d.plan, d.stage = str(outline.get("title") or "Untitled"), outline, secs, "drafting"
            save(d)
        plan = _render_outline(d.outline, d.plan)

        for i in range(len(d.sections), len(d.plan)):
            s = d.plan[i]
            say(f"drafting {i + 1}/{len(d.plan)}: {s['heading']}")
            last = "This is the LAST section: close the piece." if i == len(d.plan) - 1 else ""
            prev = d.sections[-1][1] if d.sections else ""
            covered = ("\n\nALREADY COVERED (do not repeat):\n" + "\n".join(
                f"- {h}: {one_line(t.split('. ')[0], 120)}" for h, t in d.sections)) if d.sections else ""
            prompt = (f"BRIEF:\n{brief}{ctx}{sty}\n\nOUTLINE:\n{plan}{covered}\n\nPREVIOUS SECTION (end):\n"
                      f"{prev[-900:] or '(none)'}"
                      f"\n\nWRITE SECTION: {s['heading']}\nPoints: {'; '.join(map(str, s.get('points', [])))}\n"
                      f"Length: about {s.get('words', 200)} words. {last}")
            d.sections.append((str(s["heading"]), strip_reasoning(self.llm(DRAFT_SYSTEM, prompt, 1600, 0.6))))
            save(d)

        if revise and d.sections:
            if not d.critiqued:
                d.stage = "critique"
                say("critiquing")
                crit = first_json(self.llm(CRITIC_SYSTEM, f"BRIEF:\n{brief}{ctx}{sty}\n\nOUTLINE:\n{plan}\n\nDRAFT:\n"
                                           f"{d.markdown()[:12000]}", 1000, 0.2))
                d.issues = [x for x in (crit or {}).get("issues", []) if isinstance(x, dict)][:8] \
                    if isinstance(crit, dict) else []
                if words:
                    d.issues += _length_issues(d.sections, d.plan)
                d.critiqued = True
                save(d)
            d.stage = "revising"
            for k, (head, text) in enumerate(d.sections):
                mine = [x for x in d.issues if str(x.get("section", "")).strip().lower() == head.strip().lower()]
                if not mine or head in d.revised:
                    continue
                say(f"revising: {head}")
                fixes = "\n".join(f"- {x.get('problem', '')} -> {x.get('fix', '')}" for x in mine)
                d.sections[k] = (head, strip_reasoning(self.llm(
                    REVISE_SYSTEM, f"BRIEF:\n{brief}{sty}\n\nSECTION: {head}\n{text}\n\nISSUES:\n{fixes}", 1600, 0.4)))
                d.revised.append(head)
                save(d)
        d.stage = "done"
        save(d)
        self.emit("write.done", {"title": d.title, "sections": len(d.sections), "issues": len(d.issues)})
        return d


def style_guide(profile: str = "", preferences: list[str] | None = None, limit: int = 1400) -> str:
    """The user's writing voice as one block: their consolidated profile, then the preferences recalled for the
    brief. Empty when kgirl knows nothing yet."""
    parts = []
    if profile.strip():
        parts.append("Who the user is and how they like to work:\n" + profile.strip())
    prefs = [p.strip() for p in preferences or [] if p.strip()]
    if prefs:
        parts.append("Their preferences:\n" + "\n".join(f"- {p}" for p in prefs))
    text = "\n\n".join(parts)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _num(v) -> int:
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError):
        return 0


def _set_targets(secs: list[dict], words: int | None) -> None:
    """Give each section a length target. With a total length asked for: sections the planner left without one
    share it equally (at least 80 words each); a plan that gave all of them but misses the total by over a
    quarter is scaled to it."""
    if not words or not secs:
        return
    planned = [_num(s.get("words")) for s in secs]
    if all(planned) and abs(sum(planned) - words) > 0.25 * words:
        scale = words / sum(planned)
        planned = [max(40, round(w * scale)) for w in planned]
        for s, w in zip(secs, planned):
            s["words"] = w
        return
    per = max(80, words // len(secs))
    for s, w in zip(secs, planned):
        s["words"] = w or per


def _length_issues(sections: list[tuple[str, str]], plan: list[dict]) -> list[dict]:
    """Sections far off their length target, as issues the revision fixes. Counted, not asked of the model."""
    out = []
    for (head, text), spec in zip(sections, plan):
        target, got = _num(spec.get("words")), len(text.split())
        if target < 80:
            continue
        if got > LONG * target and got - target > 40:
            out.append({"section": head, "problem": f"too long: {got} words against a target of about {target}",
                        "fix": f"cut it to about {target} words; keep the points, drop the repetition", "auto": True})
        elif got < SHORT * target:
            out.append({"section": head, "problem": f"too short: {got} words against a target of about {target}",
                        "fix": f"expand it to about {target} words with concrete detail for each point", "auto": True})
    return out


def _render_outline(outline: dict, secs: list[dict]) -> str:
    lines = [f"Title: {outline.get('title', '')}", f"Audience: {outline.get('audience', '')}",
             f"Thesis: {outline.get('thesis', '')}"]
    for s in secs:
        lines.append(f"- {s['heading']} (~{s.get('words', 200)} words): " + "; ".join(map(str, s.get("points", []))))
    return "\n".join(lines)
