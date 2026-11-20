"""The textual app behind `kgirl ui`. Imported only after `textual` is known to be installed.

Left: chat (resumes the latest conversation of this workspace). Right: status, the job tray and code
search. Every model call runs as a console job; the app is one more client of the console Hub, so chat
deltas, job progress, workspace notices and approval requests all arrive on the same queue the web
console's SSE stream reads. Approvals open a modal: kgirl asks before it edits or runs anything.
"""

from __future__ import annotations

import colorsys
import queue
import re
import time
from pathlib import Path

from rich.markdown import Markdown
from rich.markup import escape
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, RichLog, Static, TextArea

from ..console.core import audit_line, budget_line, related_line

LOGO = [
    "██╗  ██╗ ██████╗ ██╗██████╗ ██╗     ",
    "██║ ██╔╝██╔════╝ ██║██╔══██╗██║     ",
    "█████╔╝ ██║  ███╗██║██████╔╝██║     ",
    "██╔═██╗ ██║   ██║██║██╔══██╗██║     ",
    "██║  ██╗╚██████╔╝██║██║  ██║███████╗",
    "╚═╝  ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═╝╚══════╝",
]
SMALL_LOGO = "▌k g i r l▐"
NARROW = 100            # columns below which the panes stack
LIVE_LINES = 10         # the live strip shows the newest lines of a streaming answer
STUCK_S = 120.0         # a running job silent this long is flagged, as in the console tray
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

HELP = """[b]conversation[/b]
  [cyan]/new[/] \\[title]     start a new conversation      [cyan]/list[/]          recent conversations
  [cyan]/open[/] ID         switch to conversation ID     [cyan]/find[/] words    search past messages
  [cyan]/recap[/]           what this conversation is about
  [cyan]/ask[/] QUESTION    one-shot answer from your indexed repos (not saved to the conversation)
  [cyan]/good[/] [cyan]/bad[/]       your verdict on the last answer
  [cyan]/rules[/]           your standing rules (the rules check enforces them on answers)
  [cyan]/context[/]         what the last prompt held, and what the budget cut
[b]project[/b] (the directory kgirl ui was started in, or --dir)
  [cyan]/diff[/]            what kgirl changed this session   [cyan]/undo[/]   take back its last change
  [cyan]/auto[/] on|off     apply edits without asking         [cyan]/ws[/]     workspace details
  [cyan]/apply[/]           write the last task's verified diff into the project
  [cyan]/verify[/] CMD      the command that proves a task's change    [cyan]/allow[/] CMD  run CMD without asking
[b]jobs[/b]
  [cyan]/task[/] GOAL       sandboxed specialists work a goal (then /apply)
  [cyan]/build[/] SPEC      spec -> tested code (SPEC inline, or @file, or ctrl+o then ctrl+b)
  [cyan]/explain[/]         the latest build's votes: suites, candidates, repairs, why the winner won
  [cyan]/write[/] BRIEF     long-form writing    [cyan]/drafts[/]  saved drafts    [cyan]/resume[/] N  continue draft N
  [cyan]/helix[/] TASK      a HELIX-9 run      [cyan]/jobs[/]  all jobs      [cyan]/job[/] ID  one job's result
[b]view[/b]
  [cyan]/s[/] words         search code        [cyan]/events[/]  show or hide tool and memory events
  [cyan]/code[/]            is the running code still the code on disk?    [cyan]/doctor[/]  health checks
  [cyan]/md[/] on|off       render answers as Markdown (on)
  [cyan]/paste[/]           a multi-line message or spec (also ctrl+o)
  [cyan]/clear[/]           clear the chat pane   [cyan]/help[/]  this text   [cyan]/quit[/]
[b]keys[/b]  ctrl+f search · ctrl+l chat · ctrl+o compose · ctrl+n new chat · ctrl+r refresh · f1 or ? help · ctrl+q quit"""


def rainbow(line: str, phase: float, spread: float = 0.022, sat: float = 0.75, val: float = 1.0) -> Text:
    out = Text()
    for i, ch in enumerate(line):
        if ch == " ":
            out.append(ch)
            continue
        r, g, b = colorsys.hsv_to_rgb((phase + i * spread) % 1.0, sat, val)
        out.append(ch, style=f"bold #{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def rules_chips(meta: dict) -> Text | None:
    """The console's rules chips, as one line: revised (yellow) and still broken (red)."""
    hits, left = meta.get("rules_hits") or [], meta.get("rules_left") or []
    if not hits and not left:
        return None
    t = Text("  ")
    if hits:
        t.append(f" rules check: revised ({', '.join(hits)}) ", style="black on #e0af68")
        t.append(" ")
    if left:
        t.append(f" rules check: still breaks {', '.join(left)} ", style="white on #c0392b")
    return t


def meta_lines(meta: dict) -> list[Text]:
    """The dim lines under an answer: the prompt budget, earlier conversations pulled in (and why), and the
    workspace audit of the tool calls behind it. Each part is optional; older turns have none of them."""
    out = []
    budget = meta.get("budget")
    if isinstance(budget, dict) and budget:
        out.append(Text("  " + budget_line(budget) + " · /context", style="dim"))
    related = [r for r in meta.get("related") or [] if isinstance(r, dict)]
    if related:
        out.append(Text("  related: " + "; ".join(related_line(r) for r in related), style="dim"))
    for e in meta.get("audit") or []:
        if isinstance(e, dict):
            bad = re.search(r"refus|denied|fail|error|timeout", str(e.get("outcome", "")), re.I)
            out.append(Text("  ⚙ " + audit_line(e), style="red" if bad else "#9aa5ce"))
    return out


def ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds}s" if seconds < 90 else f"{seconds // 60}m" if seconds < 5400 else f"{seconds // 3600}h"


def job_line(job: dict, now: float) -> Text:
    status = job.get("status", "?")
    stuck = status == "running" and now - (job.get("last_event") or job.get("created") or now) > STUCK_S
    dot, colour = {"running": ("◐", "yellow"), "done": ("●", "green"), "error": ("●", "red")}.get(status, ("○", "dim"))
    t = Text()
    t.append(dot + " ", style="red" if stuck else colour)
    t.append(f"{job.get('kind', '?'):<6} ", style="bold")
    t.append(f"{job.get('id', '')[:6]} ", style="dim")
    t.append((job.get("label") or "")[:28])
    end = job.get("ended") or now
    t.append(f"  {ago(end - (job.get('created') or end))}", style="dim")
    if stuck:
        t.append("  silent — stuck?", style="red")
    elif status == "error":
        t.append(f"  {(job.get('error') or '')[:40]}", style="red")
    return t


def diff_text(diff: str) -> Text:
    t = Text()
    for line in diff.splitlines():
        style = ("green" if line.startswith("+") and not line.startswith("+++")
                 else "red" if line.startswith("-") and not line.startswith("---")
                 else "cyan" if line.startswith("@@") else "")
        t.append(line + "\n", style=style)
    return t


class Title(Static):
    """The rainbow logo; the hue drifts slowly to the right."""

    def on_mount(self) -> None:
        self.phase = 0.0
        self.tick()
        self.set_interval(1 / 12, self.tick)

    def tick(self) -> None:
        self.phase = (self.phase + 0.006) % 1.0
        t = Text(justify="center")
        if self.size.width >= len(LOGO[0]) + 2 and self.app.size.height >= 30:
            for n, line in enumerate(LOGO):
                t.append_text(rainbow(line, self.phase + n * 0.03))
                t.append("\n")
        else:
            t.append_text(rainbow(SMALL_LOGO, self.phase, spread=0.08))
            t.append("\n")
        t.append_text(rainbow("repo-aware · memory-backed · local-first", self.phase + 0.5, spread=0.012, sat=0.45))
        self.update(t)


class ApprovalScreen(ModalScreen[bool]):
    """kgirl wants to edit a file or run a command: show what, and wait for y/n."""

    BINDINGS = [Binding("y", "answer(True)", "yes"), Binding("n", "answer(False)", "no"),
                Binding("escape", "answer(False)", "no")]
    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; }
    #approval { width: 90%; max-width: 120; height: auto; max-height: 85%; border: thick $warning;
                background: $surface; padding: 1 2; }
    #approval-detail { height: auto; max-height: 30; border: round $primary 50%; margin: 1 0; }
    #approval-buttons { height: auto; align-horizontal: right; }
    #approval-buttons Button { margin-left: 2; }
    """

    def __init__(self, approval_id: str, question: str, detail: str):
        super().__init__()
        self.approval_id, self.question, self.detail = approval_id, question, detail

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Static(Text(self.question, style="bold"), id="approval-question")
            with VerticalScroll(id="approval-detail"):
                yield Static(diff_text(self.detail) if self.detail else Text("(no detail)", style="dim"))
            with Horizontal(id="approval-buttons"):
                yield Button("No  [n]", id="no", variant="error")
                yield Button("Yes [y]", id="yes", variant="success")

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        self.dismiss(ev.button.id == "yes")

    def action_answer(self, yes: bool) -> None:
        self.dismiss(yes)


class ComposeScreen(ModalScreen):
    """A multi-line editor for a long message or a /build spec: ctrl+s sends, ctrl+b builds, esc cancels.
    Dismisses with ("chat" | "build", text), or None."""

    BINDINGS = [Binding("ctrl+s", "done('chat')", "send"), Binding("ctrl+b", "done('build')", "build spec"),
                Binding("escape", "cancel", "cancel")]
    DEFAULT_CSS = """
    ComposeScreen { align: center middle; }
    #compose { width: 90%; max-width: 140; height: 80%; border: thick $accent; background: $surface; padding: 0 1; }
    #compose-text { height: 1fr; }
    #compose-hint { height: 1; color: $text-muted; }
    """

    def __init__(self, text: str = ""):
        super().__init__()
        self.initial = text

    def compose(self) -> ComposeResult:
        with Vertical(id="compose"):
            yield TextArea(self.initial, id="compose-text", soft_wrap=True, show_line_numbers=False)
            yield Static("ctrl+s send as a message · ctrl+b build it as a spec · esc cancel", id="compose-hint")

    def on_mount(self) -> None:
        self.query_one("#compose-text", TextArea).focus()

    def action_done(self, mode: str) -> None:
        text = self.query_one("#compose-text", TextArea).text
        self.dismiss((mode, text) if text.strip() else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,q,question_mark,f1", "close", "close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    #help { width: auto; max-width: 110; height: auto; max-height: 90%; border: round $accent; background: $surface;
            padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help"):
            yield Static(Text.from_markup(HELP + "\n\n[dim]esc closes[/]"))

    def action_close(self) -> None:
        self.dismiss(None)


class KgirlUI(App):
    TITLE = "kgirl"
    CSS = """
    Screen { background: $background; }
    Title { height: auto; padding: 1 0 0 0; content-align: center middle; }
    #body { height: 1fr; }
    #left { width: 2fr; min-width: 40; border: round $primary 60%; border-title-color: $accent; padding: 0 1; }
    #right { width: 1fr; min-width: 30; }
    #chatlog { height: 1fr; scrollbar-size-vertical: 1; }
    #live { height: auto; max-height: 14; padding: 0 1; border-left: thick $accent 50%; display: none; }
    #prompt { dock: bottom; margin-top: 1; border: tall $accent 40%; }
    #prompt:focus { border: tall $accent; }
    #mode { height: 1; color: $text-muted; padding: 0 1; }
    #status { height: auto; border: round $secondary 60%; border-title-color: $secondary; padding: 0 1; }
    #jobs { height: auto; max-height: 9; border: round $warning 60%; border-title-color: $warning; padding: 0 1; }
    #searchbox { border: round $success 60%; border-title-color: $success; height: 1fr; padding: 0 1; }
    #q { margin-bottom: 1; }
    #hits { height: 1fr; scrollbar-size-vertical: 1; }
    KgirlUI.-narrow #body { layout: vertical; }
    KgirlUI.-narrow #left { width: 1fr; height: 2fr; min-width: 0; }
    KgirlUI.-narrow #right { width: 1fr; height: 1fr; min-width: 0; }
    KgirlUI.-narrow #status { display: none; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "quit"),
        Binding("ctrl+f", "focus_search", "search"),
        Binding("ctrl+l", "focus_chat", "chat"),
        Binding("ctrl+n", "new_chat", "new chat"),
        Binding("ctrl+r", "refresh_status", "refresh"),
        Binding("ctrl+o", "compose", "compose"),
        Binding("f1", "help", "help"),
        Binding("question_mark", "help", "help", show=False),
    ]
    STATUS_EVERY = 30.0
    CODE_EVERY = 60.0

    def __init__(self, backend, new: bool = False, conv: int | None = None, pump_interval: float = 0.05):
        super().__init__()
        self.backend = backend
        self.inbox: queue.Queue = backend.events()
        self.pump_interval = pump_interval
        self.new_next = new
        self.start_conv = conv
        self.conv: int | None = None
        self.chat_job: str | None = None
        self.chat_started = 0.0
        self._buf = ""
        self.markdown = True
        self.show_events = False
        self.code_changed: dict | None = None
        self.approvals: list[dict] = []         # waiting to be shown, oldest first
        self.approval_open: str | None = None

    # ---------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Title()
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield RichLog(id="chatlog", wrap=True, markup=False, highlight=False)
                yield Static(id="live")
                yield Input(placeholder="talk to kgirl…  (/help for commands)", id="prompt")
                yield Static(id="mode")
            with Vertical(id="right"):
                yield Static(id="status")
                yield Static(id="jobs")
                with Vertical(id="searchbox"):
                    yield Input(placeholder="search code…", id="q")
                    yield RichLog(id="hits", wrap=True, markup=False, auto_scroll=False)
        yield Footer()

    WIDGETS = ("left", "chatlog", "live", "prompt", "mode", "status", "jobs", "searchbox", "q", "hits")

    def w(self, name: str):
        """A widget of the main screen. Cached at mount: `query_one` searches the ACTIVE screen, so a timer
        firing while a modal is open (or during shutdown) would not find the panes."""
        return self._widgets[name]

    def on_mount(self) -> None:
        self._widgets = {name: self.query_one(f"#{name}") for name in self.WIDGETS}
        self.theme = "tokyo-night"
        self.w("left").border_title = "chat"
        self.w("status").border_title = "status"
        self.w("jobs").border_title = "jobs"
        self.w("searchbox").border_title = "code search"
        self.set_class(self.size.width < NARROW, "-narrow")
        self.open_conversation(self.first_conversation())
        self.render_jobs()
        self.set_interval(self.pump_interval, self.pump)
        self.set_interval(1.0, self.render_jobs)
        self.action_refresh_status()
        self.set_interval(self.STATUS_EVERY, self.action_refresh_status)
        self.check_code()
        self.set_interval(self.CODE_EVERY, self.check_code)
        self.w("prompt").focus()

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < NARROW, "-narrow")

    @property
    def log_(self) -> RichLog:
        return self.w("chatlog")

    def say(self, text: str | Text, style: str = "") -> None:
        self.log_.write(text if isinstance(text, Text) else Text(text, style=style))

    def first_conversation(self) -> int:
        if self.start_conv is not None and self.backend.exists(self.start_conv):
            return self.start_conv
        if self.start_conv is not None:
            self.say(f"no conversation #{self.start_conv}; resuming the latest", "red")
        conv = self.backend.conversation(new=self.new_next)
        self.new_next = False
        return conv

    def open_conversation(self, conv: int) -> None:
        self.conv = conv
        title = self.backend.title(conv)
        self.w("left").border_title = f"chat · #{conv}" + (f" · {title}" if title else "")
        history = self.backend.history(conv)
        if history:
            self.say(f"── conversation #{conv}, last {len(history)} messages ──", "dim")
            for turn in history:
                self.show_turn(turn)
        else:
            self.say(f"── conversation #{conv} (new) · type a message, or /help ──", "dim")
        self.show_mode()

    def show_turn(self, turn: dict) -> None:
        if turn["role"] == "user":
            self.say(Text.assemble(("\nyou ", "bold cyan"), ("› ", "dim"), turn["text"]))
            return
        self.say(Text.assemble(("kgirl ", "bold magenta"), ("› ", "dim")))
        self.log_.write(Markdown(turn["text"], code_theme="monokai") if self.markdown else Text(turn["text"]))
        chips = rules_chips(turn)
        if chips is not None:
            self.say(chips)
        for line in meta_lines(turn):
            self.say(line)

    def show_mode(self) -> None:
        ws = self.backend.workspace()
        parts = [f"#{self.conv}"]
        if self.chat_job:
            parts.insert(0, "[magenta]thinking…[/]")
        parts.append(f"[yellow]{escape(ws['root'])}[/] · " + ("[red]edits auto[/]" if ws["auto_edit"] else "asks before edits")
                     if ws else "no project dir")
        parts.append("events shown" if self.show_events else "events hidden")
        if self.code_changed:
            parts.append("[b yellow]code changed · restart kgirl ui[/]")
        self.w("mode").update(Text.from_markup(" · ".join(parts)))

    # ---------------------------------------------------------------- actions
    def action_focus_search(self) -> None:
        self.w("q").focus()

    def action_focus_chat(self) -> None:
        self.w("prompt").focus()

    def action_new_chat(self, title: str = "") -> None:
        if self.chat_job:
            self.say("kgirl is still answering; start a new conversation when it is done", "yellow")
            return
        self.log_.clear()
        self.open_conversation(self.backend.conversation(new=True, title=title))

    def action_compose(self, text: str = "") -> None:
        if isinstance(self.screen, ModalScreen):
            return

        def done(result) -> None:
            if not result:
                return
            mode, body = result
            if mode == "build":
                self.start_job("build", body)
            else:
                self.converse(body.strip())

        self.push_screen(ComposeScreen(text), done)

    def action_help(self) -> None:
        if not isinstance(self.screen, ModalScreen):
            self.push_screen(HelpScreen())

    @work(exclusive=True, group="status", thread=True)
    def action_refresh_status(self) -> None:
        try:
            st = self.backend.status()
        except Exception as exc:  # noqa: BLE001 - a status probe failing is shown, never fatal
            st = {"error": f"{type(exc).__name__}: {exc}"}
        self.call_from_thread(self.render_status, st)

    def render_status(self, st: dict) -> None:
        t = Text()

        def row(label: str, value: str, ok: bool | None) -> None:
            t.append("● ", style={True: "green", False: "red", None: "yellow"}[ok])
            t.append(f"{label:<7}", style="bold")
            t.append(value + "\n")

        if st.get("error"):
            row("status", st["error"][:60], False)
        route = st.get("route") or {}
        for role in ("chat", "build"):
            if role in route:
                row(role, route[role] or "no backend", bool(route[role]))
        gpu = st.get("gpu")
        if gpu:
            temp = gpu.get("temperature", 0)
            row("gpu", f"{temp:.0f}°C · {gpu.get('memory_used', 0) / 1024:.1f}/{gpu.get('memory_total', 0) / 1024:.1f} GB",
                None if 80 <= temp < 95 else temp < 80)
        ollama = st.get("ollama") or {}
        if ollama.get("error"):
            row("models", "Ollama unreachable", False)
        elif "models" in ollama:
            names = [f"{m['name']}" + ("" if m.get("gpu_pct", 100) >= 100 else f" ({m['gpu_pct']:.0f}% GPU)")
                     for m in ollama["models"]]
            row("models", ", ".join(names) or "none loaded", True if names else None)
        if "retriever" in st:
            row("chunks", "indexed" if st["retriever"] else "no chunk index (symbols only)", st["retriever"] or None)
        for w in st.get("warnings") or []:
            t.append(f"  ! {w}\n", style="red")
        t.append(time.strftime("updated %H:%M:%S"), style="dim")
        self.w("status").update(t)

    def render_jobs(self) -> None:
        jobs = self.backend.jobs()
        now = time.time()
        t = Text()
        for job in jobs[:6]:
            t.append_text(job_line(job, now))
            t.append("\n")
        if not jobs:
            t.append("no jobs yet · /task /build /helix", style="dim")
        elif len(jobs) > 6:
            t.append(f"+{len(jobs) - 6} more · /jobs", style="dim")
        t.rstrip()
        self.w("jobs").update(t)

    @work(exclusive=True, group="code", thread=True)
    def check_code(self) -> None:
        try:
            st = self.backend.code()
        except Exception:  # noqa: BLE001 - the hint is advisory
            return
        if st.get("changed"):
            self.call_from_thread(self.code_drifted, st)

    def code_drifted(self, st: dict) -> None:
        first = self.code_changed is None
        self.code_changed = st
        if first:
            self.say(f"the code under {st.get('root')} changed since kgirl ui started "
                     f"({st.get('startup')} → {st.get('current')}); restart kgirl ui to run it", "bold yellow")
        self.show_mode()

    # ---------------------------------------------------------------- the event pump
    def pump(self) -> None:
        for _ in range(500):
            try:
                ev = self.inbox.get_nowait()
            except queue.Empty:
                return
            try:
                self.handle_event(ev)
            except Exception as exc:  # noqa: BLE001 - one bad event must not stop the pump
                self.say(f"(ui event error: {type(exc).__name__}: {exc})", "red")

    def handle_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "job":
            e = ev.get("event") or {}
            if ev.get("job") == self.chat_job:
                if e.get("type") == "delta":
                    self.stream(e.get("text", ""))
                elif e.get("type") == "queued" or (self.show_events and e.get("type") == "status"):
                    self.say(f"  · {e.get('text', '')}", "dim")
        elif kind == "job_start":
            self.render_jobs()
        elif kind == "job_end":
            self.render_jobs()
            if ev.get("job") == self.chat_job:
                self.finish_chat(ev)
            else:
                self.job_ended(ev)
        elif kind == "approval":
            self.approvals.append(ev)
            self.next_approval()
        elif kind == "approval_done":
            self.approvals = [a for a in self.approvals if a.get("id") != ev.get("id")]
            if self.approval_open == ev.get("id") and isinstance(self.screen, ApprovalScreen):
                self.approval_open = None
                self.screen.dismiss(None)      # answered elsewhere or timed out: nothing to send
        elif kind == "workspace":
            self.say(f"  · {ev.get('text', '')}", "yellow")

    # ---------------------------------------------------------------- approvals
    def next_approval(self) -> None:
        if self.approval_open is not None or not self.approvals:
            return
        ap = self.approvals.pop(0)
        self.approval_open = ap["id"]

        def answered(yes: bool | None) -> None:
            if yes is not None and self.approval_open == ap["id"]:
                ok = self.backend.approve(ap["id"], yes)
                self.say(f"  · {'approved' if yes else 'declined'}: {ap.get('question', '')}"
                         + ("" if ok else " (too late: it had already timed out)"), "green" if yes else "yellow")
            self.approval_open = None
            self.next_approval()

        self.push_screen(ApprovalScreen(ap["id"], ap.get("question", ""), ap.get("detail", "")), answered)

    # ---------------------------------------------------------------- chat
    def converse(self, msg: str) -> None:
        if self.chat_job:
            self.say("kgirl is still answering; wait for it, or watch the job tray", "yellow")
            return
        self.say(Text.assemble(("\nyou ", "bold cyan"), ("› ", "dim"), msg))
        try:
            self.chat_job = self.backend.chat(self.conv, msg)
        except Exception as exc:  # noqa: BLE001
            self.say(f"could not start the chat: {exc}", "red")
            return
        self.chat_started = time.monotonic()
        self._buf = ""
        self.show_mode()

    def stream(self, text: str) -> None:
        """The answer as it arrives, in the live pane under the log. The log gets the final answer once it ends
        (rendered as Markdown, and after any rules-check revision), so the stream itself stays plain."""
        self._buf += text
        live = self.w("live")
        live.display = True
        tail = "\n".join(self._buf.split("\n")[-LIVE_LINES:])[-LIVE_LINES * 160:]   # the newest text stays in view
        live.update(Text.assemble(("kgirl › ", "bold magenta"), tail))

    def end_stream(self) -> str:
        live = self.w("live")
        live.update("")
        live.display = False
        text, self._buf = self._buf, ""
        return text

    def finish_chat(self, ev: dict) -> None:
        streamed = self.end_stream()
        job = self.backend.job(ev["job"]) or {}
        result = job.get("result") or {}
        if ev.get("status") == "error":
            if streamed.strip():
                self.say(Text(streamed, style="dim"))
            self.say(f"chat failed: {ev.get('error') or job.get('error')}", "red")
        elif result.get("text") or streamed.strip():
            self.show_turn({**result, "role": "assistant", "text": result.get("text") or streamed})
        took = time.monotonic() - self.chat_started
        extra = [f"recalled {len(result['recalled'])}"] if result.get("recalled") else []
        extra += [f"remembered {', '.join(result['stored'])}"] if result.get("stored") else []
        self.say("  " + " · ".join([f"chat · {took:.1f}s", *extra]), "dim")
        self.chat_job = None
        self.show_mode()

    def job_ended(self, ev: dict) -> None:
        jid, kind, status = ev.get("job", ""), ev.get("kind", "job"), ev.get("status")
        if status == "error":
            self.say(f"  ✗ {kind} {jid[:6]} failed: {ev.get('error', '')}", "red")
            return
        job = self.backend.job(jid) or {}
        result = job.get("result") or {}
        if kind == "ask":
            self.show_ask(result)
            return
        hint = f"/job {jid[:6]} to see it"
        if kind == "task" and result.get("files"):
            hint = f"{len(result['files'])} file(s) verified · /job {jid[:6]} for the diff · /apply writes it"
        elif kind == "write" and result.get("draft") is not None:
            hint = f"saved as draft #{result['draft']} · /job {jid[:6]} to read it · /resume {result['draft']} continues it"
        elif kind in ("build", "builder") and "confidence" in result:
            hint = f"confidence {result['confidence']:.2f} · /job {jid[:6]} for the code"
        self.say(f"  ✓ {kind} {jid[:6]} done · {hint}", "green")

    def ask(self, question: str) -> None:
        if not question:
            self.say("/ask QUESTION", "red")
            return
        self.say(Text.assemble(("\nyou ", "bold cyan"), ("ask › ", "dim"), question))
        self.backend.ask(question)
        self.render_jobs()

    def show_ask(self, result: dict) -> None:
        """A finished /ask: the answer like a chat answer, then where it came from. It stays out of the conversation."""
        self.say(Text.assemble(("kgirl ", "bold magenta"), ("ask › ", "dim")))
        text = result.get("text") or "(no answer)"
        self.log_.write(Markdown(text, code_theme="monokai") if self.markdown else Text(text))
        foot = [f"ask · {result.get('intent', '?')} via {result.get('backend', '?')}"]
        if result.get("context_tokens"):
            foot.append(f"~{result['context_tokens']} ctx tokens")
        self.say("  " + " · ".join(foot), "dim")
        cites = result.get("citations") or []
        if cites:
            self.say("  " + ", ".join(cites[:10]), "dim")

    # ---------------------------------------------------------------- input
    @on(Input.Submitted, "#prompt")
    def chat_submitted(self, ev: Input.Submitted) -> None:
        msg = ev.value.strip()
        ev.input.value = ""
        if not msg:
            return
        if msg.startswith("/"):
            cmd, _, arg = msg[1:].partition(" ")
            try:
                self.slash(cmd.lower(), arg.strip())
            except Exception as exc:  # noqa: BLE001 - keep the session alive
                self.say(f"(/{cmd}: {type(exc).__name__}: {exc})", "red")
            return
        self.converse(msg)

    @on(Input.Submitted, "#q")
    def search_submitted(self, ev: Input.Submitted) -> None:
        self.code_search(ev.value)

    def slash(self, cmd: str, arg: str) -> None:
        b = self.backend
        if cmd in ("quit", "exit", "q"):
            self.exit()
        elif cmd == "help":
            self.say(Text.from_markup(HELP))
        elif cmd == "clear":
            self.log_.clear()
        elif cmd == "paste":
            self.action_compose(arg)
        elif cmd == "new":
            self.action_new_chat(arg)
        elif cmd == "list":
            for c in b.conversations():
                self.say(f"#{c['id']:<5} {(c['title'] or '(untitled)')[:36]:<36} {(c['summary'] or '')[:70]}")
        elif cmd == "open":
            if not arg.isdigit() or not b.exists(int(arg)):
                self.say(f"no conversation {arg or '(give an id)'}; /list shows them", "red")
            elif self.chat_job:
                self.say("kgirl is still answering; switch when it is done", "yellow")
            else:
                self.log_.clear()
                self.open_conversation(int(arg))
        elif cmd == "find":
            hits = b.find(arg) if arg else []
            for h in hits:
                self.say(f"#{h['conv_id']:<5} {h['role']:<9} {' '.join(h['text'].split())[:100]}")
            if not hits:
                self.say("no matching messages" if arg else "/find words", "dim")
        elif cmd == "recap":
            self.say(b.recap(self.conv))
        elif cmd in ("good", "bad"):
            self.say(b.feedback(self.conv, "positive" if cmd == "good" else "negative"), "dim")
        elif cmd == "rules":
            r = b.rules()
            if not r.get("available"):
                self.say("the rules check is not part of this build", "dim")
            for rule in r.get("rules", []):
                self.say(f"#{rule['id']:<5} {rule['text']}" + (f"  [{rule['scope']}]" if rule.get("scope") not in (None, "*") else ""))
            if r.get("available") and not r.get("rules"):
                self.say('no standing rules yet · tell kgirl "never …" or "always …" to add one', "dim")
        elif cmd == "context":
            ctx = b.context(self.conv)
            if ctx.get("report"):
                self.say(ctx["report"])
            elif ctx.get("budget"):
                self.say(budget_line(ctx["budget"]))
            else:
                self.say("no prompt budget recorded for this conversation" if ctx.get("available")
                         else "this build does not record the prompt budget", "dim")
        elif cmd == "write":
            if not arg:
                self.say("/write BRIEF  (or ctrl+o for a long brief)", "red")
            else:
                self.say(f"  ◐ write {b.write(self.conv, arg)[:6]} started · it shows in the job tray", "dim")
                self.render_jobs()
        elif cmd == "drafts":
            d = b.drafts()
            if not d.get("available"):
                self.say("this build keeps no drafts", "dim")
            for x in d.get("drafts", []):
                self.say(f"#{x['id']:<5} {(x['title'] or '(untitled)')[:40]:<40} {x['progress']}  (conv #{x['conv_id']})")
            if d.get("available") and not d.get("drafts"):
                self.say("no drafts yet · /write BRIEF", "dim")
        elif cmd == "resume":
            if not arg.isdigit():
                self.say("/resume DRAFT_ID  (/drafts lists them)", "red")
            else:
                self.say(f"  ◐ write {b.resume_draft(int(arg))[:6]} resumes draft #{arg}", "dim")
                self.render_jobs()
        elif cmd == "doctor":
            self.say("  · checking health (read-only)…", "dim")
            self.run_doctor()
        elif cmd == "verify":
            if arg:
                verify = b.set_verify(arg)
                self.say(f"verifier: {verify}")
            else:
                self.say("/verify CMD sets the command that proves a task's change, e.g. /verify python3 -m pytest -q", "dim")
        elif cmd == "allow":
            self.say(b.allow(arg) if arg else "/allow CMD lets kgirl run CMD here without asking")
        elif cmd == "md":
            if arg not in ("on", "off"):
                self.say(f"/md on|off  (answers render as Markdown: {'on' if self.markdown else 'off'})", "dim")
            else:
                self.markdown = arg == "on"
                self.say(f"Markdown rendering {arg}", "dim")
        elif cmd == "diff":
            d = b.diff()
            self.say(diff_text(d) if d.startswith(("---", "diff", "@@")) else Text(d))
        elif cmd == "undo":
            self.say(b.undo())
        elif cmd == "auto":
            if arg not in ("on", "off"):
                self.say("/auto on|off", "red")
            else:
                self.say(b.set_auto(arg == "on"))
                self.show_mode()
        elif cmd == "apply":
            self.say(b.apply(self.conv))
        elif cmd == "ws":
            ws = b.workspace()
            if ws is None:
                self.say("no project directory attached; start kgirl ui inside one, or pass --dir", "dim")
            else:
                self.say(f"project: {ws['root']}\nedits: {'apply without asking' if ws['auto_edit'] else 'ask first'}\n"
                         f"allowed commands: {', '.join(ws['allowed']) or 'none (kgirl asks)'}")
        elif cmd in ("task", "build", "helix"):
            self.start_job(cmd, arg)
        elif cmd == "ask":
            self.ask(arg)
        elif cmd == "jobs":
            jobs, now = b.jobs(), time.time()
            for job in jobs:
                self.say(job_line(job, now))
            if not jobs:
                self.say("no jobs yet", "dim")
        elif cmd == "job":
            self.show_job(arg)
        elif cmd == "explain":
            self.explain_build()
        elif cmd == "events":
            self.show_events = not self.show_events
            self.show_mode()
        elif cmd == "code":
            self.show_code()
        elif cmd in ("s", "search"):
            self.w("q").value = arg
            self.code_search(arg)
        else:
            self.say(f"unknown command /{cmd} — /help", "red")

    def start_job(self, kind: str, arg: str) -> None:
        if not arg:
            self.say({"task": "/task GOAL", "build": "/build SPEC  or  /build @spec.md",
                      "helix": "/helix TASK"}[kind], "red")
            return
        if kind == "build" and arg.startswith("@"):
            path = Path(arg[1:]).expanduser()
            try:
                arg = path.read_text(encoding="utf-8")
            except OSError as exc:
                self.say(f"cannot read {path}: {exc.strerror or exc}", "red")
                return
        if kind == "task":
            jid = self.backend.task(self.conv, arg)
        elif kind == "build":
            jid = self.backend.build(arg)
        else:
            jid = self.backend.helix(arg)
        self.say(f"  ◐ {kind} {jid[:6]} started · it shows in the job tray", "dim")
        self.render_jobs()

    def show_job(self, prefix: str) -> None:
        matches = [j for j in self.backend.jobs() if prefix and j["id"].startswith(prefix)]
        if len(matches) != 1:
            self.say("no such job; /jobs lists them" if not matches else f"{len(matches)} jobs start with {prefix}", "red")
            return
        job = self.backend.job(matches[0]["id"]) or {}
        self.say(job_line(job, time.time()))
        if job.get("error"):
            self.say(job["error"], "red")
        result = job.get("result") or {}
        if not isinstance(result, dict):
            self.say(str(result))
            return
        for key in ("render", "text", "markdown", "explain"):
            if isinstance(result.get(key), str) and result[key].strip():
                self.say(result[key])
        for f in result.get("files") or []:
            self.say(diff_text(f.get("diff", "")))
        if isinstance(result.get("code"), str) and result["code"].strip():
            self.say(Text(result["code"], style="#9aa5ce"))
        if job.get("status") == "running":
            last = [e.get("text", "") for e in job.get("events") or [] if e.get("type") in ("status", "queued")]
            self.say(f"  running · {last[-1] if last else 'no progress yet'}", "dim")

    def explain_build(self) -> None:
        """The latest finished build's explain(): its suites, the candidate x suite votes, the repairs and why
        the winner won. Builders without explain() leave it empty."""
        for summary in self.backend.jobs():
            if summary["kind"] in ("build", "builder") and summary["status"] == "done":
                result = (self.backend.job(summary["id"]) or {}).get("result") or {}
                text = result.get("explain") if isinstance(result, dict) else ""
                self.say(text if text and text.strip() else f"build {summary['id'][:6]} has no explanation "
                         "(this builder does not provide explain())")
                return
        self.say("no finished build yet · /build SPEC", "dim")

    @work(exclusive=True, group="doctor", thread=True)
    def run_doctor(self) -> None:
        try:
            d = self.backend.doctor()
        except Exception as exc:  # noqa: BLE001 - a failed probe is a result to show
            self.call_from_thread(self.say, f"doctor failed: {type(exc).__name__}: {exc}", "red")
            return
        self.call_from_thread(self.show_doctor, d)

    def show_doctor(self, d: dict) -> None:
        if not d.get("available"):
            self.say("kgirl doctor is not part of this build", "dim")
            return
        colours = {"OK": "green", "WARN": "yellow", "FAIL": "red"}
        for c in d.get("checks", []):
            t = Text()
            t.append(f"{c.get('status', '?'):<5}", style="bold " + colours.get(c.get("status"), ""))
            t.append(f"{c.get('name', '')}: ", style="bold")
            t.append(str(c.get("detail", "")))
            if c.get("fix"):
                t.append(f"\n      fix: {c['fix']}", style="dim")
            self.say(t)
        fails = sum(c.get("status") == "FAIL" for c in d.get("checks", []))
        self.say(f"  {len(d.get('checks', []))} checks · {fails} failing", "red" if fails else "green")

    @work(exclusive=True, group="code-now", thread=True)
    def show_code(self) -> None:
        st = self.backend.code()
        if not st.get("available"):
            msg = Text(f"no git checkout at {st.get('root')} to watch (set KGIRL_CODE)", style="dim")
        elif st.get("changed"):
            msg = Text(f"changed since start: {st.get('startup')} → {st.get('current')}; restart kgirl ui", style="yellow")
        else:
            msg = Text(f"running code matches {st.get('root')} ({st.get('startup')})", style="green")
        self.call_from_thread(self.say, msg)
        if st.get("changed"):
            self.call_from_thread(self.code_drifted, st)

    # ---------------------------------------------------------------- search
    @work(exclusive=True, group="search", thread=True)
    def code_search(self, q: str) -> None:
        hits = self.w("hits")
        self.call_from_thread(hits.clear)
        q = q.strip()
        if not q:
            return
        t0 = time.monotonic()
        try:
            res = self.backend.search(q)
        except Exception as exc:  # noqa: BLE001 - an unbuilt index is a state to show
            self.call_from_thread(hits.write, Text(f"search failed: {exc}", style="red"))
            return
        ms = (time.monotonic() - t0) * 1000
        words = [w for w in WORD.findall(q) if len(w) > 1]
        pat = re.compile("|".join(re.escape(w) for w in words), re.I) if words else None
        symbols, chunks = res.get("symbols") or [], res.get("chunks") or []
        out = [Text(f"{len(symbols)} symbols · {len(chunks)} chunks · {ms:.0f} ms", style="dim")]
        for s in symbols:
            t = Text()
            t.append(s.get("repo", ""), style="bold #e0855a")
            t.append(f":{s.get('path', '')}:{s.get('line', '')}", style="#9aa5ce")
            t.append(f"\n  {s.get('kind', '')} ", style="dim")
            t.append(s.get("qualname", ""), style="bold")
            if s.get("signature"):
                t.append(f"\n  {' '.join(str(s['signature']).split())[:120]}", style="#7f8699")
            out.append(t)
        for c in chunks:
            t = Text()
            t.append(c.get("repo", ""), style="bold #e0855a")
            t.append(f":{c.get('path', '')}:{c.get('start_line', '')}", style="#9aa5ce")
            if c.get("symbol"):
                t.append(f"\n  {c.get('kind', '')} ", style="dim")
                t.append(c["symbol"], style="bold")
            snippet = " ".join(str(c.get("text", "")).split())[:140]
            s = Text("\n  " + snippet, style="#7f8699")
            if pat is not None:
                for m in pat.finditer(snippet):
                    s.stylize("bold #ffd479", m.start() + 3, m.end() + 3)
            t.append_text(s)
            out.append(t)
        for t in out:
            self.call_from_thread(hits.write, t)
        self.call_from_thread(hits.scroll_home, animate=False)
