"""The assistant: Atlas + Soup + Hermes + Jev behind one object.

    ask(q)       intent -> (blast | card | retrieval) -> scout model answers from a
                 token-bounded context pack with repo:path:line citations
    task(goal)   Jev swarm in sandboxes -> verified diff -> distilled into Soup
    index(srcs)  Atlas ingest + architecture cards + repo facts into Soup

Every cross-layer call goes through the Hermes bus (`atlas.*`, `soup.*`,
`llm.complete`, `jev.*`, `swarm.*`), so one trace explains any answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .atlas import Atlas, architecture_card, blast_radius, render_card, resolve_source
from .atlas.store import IngestStats
from .hermes import Bus, Intent, NoBackend, Router, classify
from .hermes.router import SCOUT
from .jev import Swarm, SwarmConfig, SwarmResult
from .llm import Message
from .soup import Curator, Soup
from .util import clip, estimate_tokens, harness_home

ASK_SYSTEM = """You are kgirl, a personal engineering assistant that knows the user's repositories by
their structure. Answer ONLY from the CONTEXT. Cite every claim as repo:path:line. If the context
does not contain the answer, say exactly what is missing and which symbol or file to index or open.
Be dense and technical; prefer lists of concrete symbols over prose."""

SUMMARY_SYSTEM = """Summarize a software repository's architecture from its structural card in at most
5 sentences: purpose, main subsystems, entrypoints, key dependencies, and anything fragile
(unparseable files, duplicated modules). No marketing language."""

_TARGET = re.compile(r"`([^`]+)`|((?:[\w.-]+/)*[\w.-]+\.(?:py|ts|js|jl|rs|go))|\b([A-Z][A-Za-z0-9]+(?:\.[A-Za-z_]\w*)?)\b")


@dataclass
class Answer:
    text: str
    intent: str
    citations: list[str] = field(default_factory=list)
    backend: str = "extractive"
    context_tokens: int = 0


class Assistant:
    def __init__(self, home: str | Path | None = None, router: Router | None = None, atlas: Atlas | None = None,
                 soup: Soup | None = None, bus: Bus | None = None, swarm_config: SwarmConfig | None = None,
                 trace: bool = True):
        self.home = Path(home) if home else harness_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.atlas = atlas or Atlas(self.home / "atlas.db")
        self.soup = soup or Soup(self.home / "soup.db")
        self.curator = Curator(self.soup)
        self.bus = bus or Bus(trace_path=self.home / "trace.jsonl" if trace else None)
        self.router = router or Router.from_env()
        self.swarm_config = swarm_config or SwarmConfig()
        self._register()

    def _register(self) -> None:
        b = self.bus
        b.serve("llm.complete", self.router.as_service(), "router")
        b.serve("atlas.search", lambda p: self.atlas.search(p["q"], p.get("limit", 12), p.get("repo")), "atlas")
        b.serve("atlas.blast", lambda p: blast_radius(self.atlas, p["target"], p.get("depth", 3)), "atlas")
        b.serve("atlas.card", lambda p: architecture_card(self.atlas, p["repo"]), "atlas")
        b.serve("soup.recall", lambda p: self.soup.recall(p["q"], p.get("budget", 400), scope=p.get("scope")),
                "soup")

    # ------------------------------------------------------------------ model access

    def llm(self, role: str, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        out = self.bus.request("llm.complete", {"role": role, "system": system, "max_tokens": max_tokens,
                                                "temperature": temperature,
                                                "messages": [Message("user", prompt)]}, sender="assistant")
        return out.text

    # ------------------------------------------------------------------ knowledge

    def index(self, specs: list[str], refresh: bool = False, cache_dir: Path | None = None) -> list[IngestStats]:
        stats = []
        for spec in specs:
            src = resolve_source(spec, cache_dir=cache_dir or self.home / "repos", refresh=refresh)
            st = self.atlas.ingest(src.root, src.name, src.origin)
            self.bus.publish("atlas.indexed", st.__dict__, "assistant")
            stats.append(st)
        for st in stats:
            card = architecture_card(self.atlas, st.repo)
            self.atlas.set_card(st.repo, card)
            hubs = "; ".join(card["hub_modules"][:4]) or "none"
            langs = ", ".join(f"{k}:{v['files']}" for k, v in list(card["languages"].items())[:4])
            self.soup.add(f"repo {st.repo}: languages {langs}; hub modules {hubs}; entrypoints "
                          f"{', '.join(card['entrypoints'][:4]) or 'none'}", kind="fact", tags=st.repo,
                          source="atlas", scope=st.repo)
        return stats

    def summarize_repo(self, repo: str) -> str:
        card = architecture_card(self.atlas, repo)
        summary = self.llm(SCOUT, SUMMARY_SYSTEM, render_card(card), 400)
        card["summary"] = " ".join(summary.split())
        self.atlas.set_card(repo, card)
        return card["summary"]

    def context_pack(self, question: str, repo: str | None = None, budget: int = 2800,
                     snippets: int = 3) -> tuple[str, list[str]]:
        hits = self.bus.request("atlas.search", {"q": question, "limit": 14, "repo": repo}, "assistant")
        parts, cites, used = [], [], 0
        code_hits = [h for h in hits if h.kind != "section"]
        for h in code_hits[:snippets]:
            try:
                src = self.atlas.read_source(h.repo, h.path, max(1, h.line - 2), h.line + 28)
            except (FileNotFoundError, OSError):
                continue
            block = f"### {h.cite()}  {h.signature}\n{src}"
            if used + estimate_tokens(block) > budget * 0.6:
                break
            parts.append(block)
            cites.append(h.cite())
            used += estimate_tokens(block)
        index_lines = []
        for h in hits:
            line = f"- {h.cite()} [{h.kind}] {clip(h.signature, 110)}" + (f" — {clip(h.doc, 120)}" if h.doc else "")
            if used + estimate_tokens(line) > budget * 0.85:
                break
            index_lines.append(line)
            used += estimate_tokens(line)
            if h.cite() not in cites:
                cites.append(h.cite())
        if index_lines:
            parts.append("### matching symbols\n" + "\n".join(index_lines))
        rec = self.bus.request("soup.recall", {"q": question, "budget": max(120, budget - used), "scope": repo},
                               "assistant")
        if rec.fragments:
            parts.append("### memory\n" + rec.render())
        return "\n\n".join(parts), cites

    # ------------------------------------------------------------------ user-facing verbs

    def ask(self, question: str, repo: str | None = None) -> Answer:
        intent = classify(question)
        env = self.bus.publish("assistant.ask", {"q": question, "intent": intent.value, "repo": repo}, "user")
        if intent is Intent.BLAST:
            target = self._extract_target(question)
            if target:
                rep = blast_radius(self.atlas, target)
                if rep.seeds:
                    return Answer(rep.render(), intent.value, [f"{r}:{p}" for r, p in rep.seeds])
        if intent is Intent.ARCH:
            name = repo or self._extract_repo(question)
            if name:
                return Answer(render_card(architecture_card(self.atlas, name)), intent.value, [name])
        context, cites = self.context_pack(question, repo)
        if not context:
            return Answer("Nothing in the atlas or memory matches; index the relevant repo first "
                          "(`python -m kgirl.harness index <path|owner/repo>`).", intent.value)
        try:
            text = self.llm(SCOUT, ASK_SYSTEM, f"QUESTION: {question}\n\nCONTEXT:\n{context}", 1200)
            backend = self.router.pick(SCOUT)
            ans = Answer(text, intent.value, cites, f"{backend.name}:{backend.model}" if backend else "?",
                         estimate_tokens(context))
        except NoBackend as exc:
            ans = Answer(f"(no model reachable — {exc})\n\n{context}", intent.value, cites, "extractive",
                         estimate_tokens(context))
        self.bus.publish("assistant.answer", {"backend": ans.backend, "citations": ans.citations[:10]}, "assistant",
                         parent=env)
        return ans

    def task(self, goal: str, repo_path: str | Path, verify_cmd: list[str] | None = None, size: int | None = None,
             apply: bool = False, repo_name: str | None = None) -> SwarmResult:
        cfg = self.swarm_config
        if size:
            cfg = SwarmConfig(size=size, temperatures=cfg.temperatures, memory_budget=cfg.memory_budget,
                              jev=cfg.jev, stop_on_first_success=cfg.stop_on_first_success,
                              keep_workspaces=cfg.keep_workspaces, allow_unverified=cfg.allow_unverified)
        root = self.bus.publish("assistant.task", {"goal": goal, "repo": str(repo_path)}, "user")
        swarm = Swarm(
            llm=lambda role, system, prompt, mt, t: self.llm(role, system, prompt, mt, t),
            soup=self.soup, curator=self.curator, config=cfg,
            blast=lambda repo, path: blast_radius(self.atlas, f"{repo}:{path}").render(20),
            emit=lambda topic, payload: self.bus.publish(topic, payload, "swarm", parent=root))
        return swarm.run(goal, repo_path, verify_cmd, repo_name=repo_name, apply=apply)

    def remember(self, text: str, kind: str = "note", tags: str = "", scope: str = "*") -> int:
        return self.soup.add(text, kind=kind, tags=tags, source="user", scope=scope)

    # ------------------------------------------------------------------ helpers

    def _extract_target(self, text: str) -> str | None:
        for m in _TARGET.finditer(text):
            cand = next(g for g in m.groups() if g)
            if self.atlas.find_symbol(cand.split("::")[-1]) or self.atlas.db.execute(
                    "SELECT 1 FROM files WHERE path=? OR path LIKE ?", (cand, "%/" + cand)).fetchone():
                return cand
        return None

    def _extract_repo(self, text: str) -> str | None:
        names = {r["name"].lower(): r["name"] for r in self.atlas.repos()}
        for w in re.findall(r"[\w.-]+", text):
            if w.lower() in names:
                return names[w.lower()]
        return None

    def close(self) -> None:
        self.atlas.close()
        self.soup.close()
