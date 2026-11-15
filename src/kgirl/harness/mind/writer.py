"""Long-form writing: plan, draft by section, critique, revise.

A small model writing 1,500 words in one pass drifts, repeats itself and forgets the brief. The writer
keeps each call small and focused:

    plan     brief + what kgirl remembers (the user's writing preferences, relevant facts, optional repo
             context) -> a JSON outline: title, audience, thesis, sections with their points
    draft    one call per section, each given the outline, the brief and the tail of the previous section
    critique one call that lists concrete problems against the brief (missing points, repetition,
             unsupported claims, tone that breaks the user's preferences)
    revise   per section, only where the critique names that section

Preferences recalled into the plan are credited by the chat loop's feedback like any other memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .text import first_json, strip_reasoning

LLM = Callable[[str, str, int, float], str]

PLAN_SYSTEM = """You plan a piece of writing. Return ONLY JSON:
{"title": "...", "audience": "...", "thesis": "<one sentence>",
 "sections": [{"heading": "...", "points": ["...", "..."], "words": 150}, ...]}
3 to 7 sections. Points are concrete (names, numbers, examples), not topics. Respect the user's
preferences and use the facts given; never invent facts that are not in the brief or the context."""

DRAFT_SYSTEM = """You write one section of a longer piece. Write only this section's body in Markdown
(no heading line). Cover every point, in order, with concrete detail. Match the requested length within
20%. Keep the voice consistent with the previous section. Do not summarize the whole piece or add a
conclusion unless this is the last section."""

CRITIC_SYSTEM = """You are a demanding editor. Compare the draft with the brief and the outline. Return ONLY
JSON: {"issues": [{"section": "<heading>", "problem": "...", "fix": "..."}, ...]}
List at most 8 issues, most important first: missing or wrong content, repetition, vague claims, broken
logic between sections, tone that conflicts with the user's preferences. An empty list is fine."""

REVISE_SYSTEM = """Revise this section of a longer piece to fix the listed issues. Keep everything that is
not affected. Return only the revised section body in Markdown, no heading line."""


@dataclass
class Draft:
    title: str
    outline: dict
    sections: list[tuple[str, str]] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    recalled: list[int] = field(default_factory=list)

    def markdown(self) -> str:
        body = "\n\n".join(f"## {h}\n\n{t.strip()}" for h, t in self.sections)
        return f"# {self.title}\n\n{body}\n"


class Writer:
    def __init__(self, llm: LLM, emit: Callable[[str, dict], None] | None = None):
        self.llm = llm
        self.emit = emit or (lambda topic, payload: None)

    def write(self, brief: str, context: str = "", words: int | None = None, revise: bool = True,
              progress: Callable[[str], None] | None = None) -> Draft:
        say = progress or (lambda msg: None)
        ctx = f"\n\nCONTEXT (what kgirl knows; use, don't contradict):\n{context}" if context else ""
        target = f"\nTarget length: about {words} words in total." if words else ""
        say("planning")
        outline = first_json(self.llm(PLAN_SYSTEM, f"BRIEF:\n{brief}{target}{ctx}", 1200, 0.3))
        if not isinstance(outline, dict) or not outline.get("sections"):
            outline = {"title": brief[:60], "thesis": brief, "sections": [{"heading": "Draft", "points": [brief],
                                                                            "words": words or 500}]}
        secs = [s for s in outline["sections"] if isinstance(s, dict) and s.get("heading")][:7]
        if words and secs:
            per = max(80, words // len(secs))
            for s in secs:
                s.setdefault("words", per)
        plan = _render_outline(outline, secs)
        d = Draft(str(outline.get("title") or "Untitled"), outline)
        prev = ""
        for i, s in enumerate(secs):
            say(f"drafting {i + 1}/{len(secs)}: {s['heading']}")
            last = "This is the LAST section: close the piece." if i == len(secs) - 1 else ""
            prompt = (f"BRIEF:\n{brief}{ctx}\n\nOUTLINE:\n{plan}\n\nPREVIOUS SECTION (end):\n{prev[-900:] or '(none)'}"
                      f"\n\nWRITE SECTION: {s['heading']}\nPoints: {'; '.join(map(str, s.get('points', [])))}\n"
                      f"Length: about {s.get('words', 200)} words. {last}")
            text = strip_reasoning(self.llm(DRAFT_SYSTEM, prompt, 1600, 0.6))
            d.sections.append((str(s["heading"]), text))
            prev = text
        if revise and d.sections:
            say("critiquing")
            crit = first_json(self.llm(CRITIC_SYSTEM, f"BRIEF:\n{brief}{ctx}\n\nOUTLINE:\n{plan}\n\nDRAFT:\n"
                                       f"{d.markdown()[:12000]}", 1000, 0.2))
            d.issues = [x for x in (crit or {}).get("issues", []) if isinstance(x, dict)][:8] \
                if isinstance(crit, dict) else []
            for k, (head, text) in enumerate(d.sections):
                mine = [x for x in d.issues if str(x.get("section", "")).strip().lower() == head.strip().lower()]
                if not mine:
                    continue
                say(f"revising: {head}")
                fixes = "\n".join(f"- {x.get('problem', '')} -> {x.get('fix', '')}" for x in mine)
                d.sections[k] = (head, strip_reasoning(self.llm(
                    REVISE_SYSTEM, f"BRIEF:\n{brief}\n\nSECTION: {head}\n{text}\n\nISSUES:\n{fixes}", 1600, 0.4)))
        self.emit("write.done", {"title": d.title, "sections": len(d.sections), "issues": len(d.issues)})
        return d


def _render_outline(outline: dict, secs: list[dict]) -> str:
    lines = [f"Title: {outline.get('title', '')}", f"Audience: {outline.get('audience', '')}",
             f"Thesis: {outline.get('thesis', '')}"]
    for s in secs:
        lines.append(f"- {s['heading']} (~{s.get('words', 200)} words): " + "; ".join(map(str, s.get("points", []))))
    return "\n".join(lines)
