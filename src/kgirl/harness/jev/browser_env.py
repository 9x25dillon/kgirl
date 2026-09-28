"""Browser world for Jev (optional; needs `pip install playwright`).

Follows the Jev Ultrafast design: no screenshots, structured state only. Each
observation injects SNAPSHOT_JS, which numbers visible interactive elements
and returns a compact table. The model picks OP + index in one round trip;
text is generated (by the small model) only for TYPE_TEXT / SELECT.

Target validation: every element carries a signature (tag|role|label). Before
acting, the page is re-snapshotted and the element at that index must still
carry the same signature, otherwise the action is refused as stale — model
output never becomes a selector, coordinate or script.
"""

from __future__ import annotations

import time
from typing import Callable

from .env import ActionResult, Element, Observation
from .ops import Decision, OpSpec

BROWSER_OPS: dict[str, OpSpec] = {s.name: s for s in (
    OpSpec("CLICK", needs_target=True, help="click element i"),
    OpSpec("TYPE_TEXT", needs_target=True, help="type generated text into input i"),
    OpSpec("SELECT", needs_target=True, help="choose an option in select i"),
    OpSpec("SCROLL_UP", help="scroll up one screen"),
    OpSpec("SCROLL_DOWN", help="scroll down one screen"),
    OpSpec("WAIT", help="wait for the page to settle"),
    OpSpec("DONE", needs_arg=True, terminal=True, help="goal achieved; answer/summary"),
    OpSpec("BLOCKED", needs_arg=True, terminal=True, help="cannot proceed; reason"),
)}

SNAPSHOT_JS = r"""
(maxItems) => {
  const sel = 'a[href],button,input:not([type=hidden]),select,textarea,[role=button],[role=link],' +
              '[role=option],[role=tab],[role=combobox],[role=menuitem],[contenteditable=true],[onclick]';
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none' &&
           r.bottom > 0 && r.top < innerHeight * 2; };
  const label = el => (el.getAttribute('aria-label') || el.labels?.[0]?.innerText || el.placeholder ||
                       el.innerText || el.value || el.title || el.name || '').trim().replace(/\s+/g, ' ').slice(0, 80);
  document.querySelectorAll('[data-jev]').forEach(e => e.removeAttribute('data-jev'));
  const out = [];
  for (const el of document.querySelectorAll(sel)) {
    if (!vis(el) || el.disabled) continue;
    const idx = out.length; if (idx >= maxItems) break;
    el.setAttribute('data-jev', String(idx));
    const tag = el.tagName.toLowerCase(), role = el.getAttribute('role') || '';
    const opts = tag === 'select' ? Array.from(el.options).slice(0, 12).map(o => o.text.trim()) : [];
    out.push({idx, tag, role, type: el.type || '', label: label(el), value: (el.value || '').slice(0, 40), opts});
  }
  const text = (document.body?.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 300);
  return {title: document.title, url: location.href, items: out, text};
}
"""

TextFn = Callable[[str, str], str]   # (goal, element description) -> text to type / option to pick


def _sig(it: dict) -> str:
    return f"{it['tag']}|{it['role']}|{it['label']}"


class BrowserEnvironment:
    kind = "browser"
    ops = BROWSER_OPS

    def __init__(self, goal: str, start_url: str, text_fn: TextFn, headless: bool = True, max_items: int = 40,
                 settle_ms: int = 600, executable_path: str | None = None):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("BrowserEnvironment needs `pip install playwright`") from exc
        self.goal, self.text_fn, self.max_items, self.settle_ms = goal, text_fn, max_items, settle_ms
        self._pw = sync_playwright().start()
        kw = {"headless": headless}
        if executable_path:
            kw["executable_path"] = executable_path
        self._browser = self._pw.chromium.launch(**kw)
        self.page = self._browser.new_page()
        self.page.goto(start_url, wait_until="domcontentloaded")
        self._snap: dict = {}
        self.note = f"opened {start_url}"

    def _snapshot(self) -> dict:
        self._snap = self.page.evaluate(SNAPSHOT_JS, self.max_items)
        return self._snap

    def observe(self) -> Observation:
        snap = self._snapshot()
        els = []
        for it in snap["items"]:
            desc = it["label"] or "(no label)"
            if it["tag"] == "input":
                desc += f"  type={it['type']}" + (f" value={it['value']!r}" if it["value"] else "")
            if it["opts"]:
                desc += "  options: " + " | ".join(it["opts"])
            els.append(Element(it["role"] or it["tag"], desc, {"sig": _sig(it), "idx": it["idx"]}))
        header = f"page: {snap['title'][:60]} <{snap['url'][:80]}>\ntext: {snap.get('text', '')}"
        return Observation(header, tuple(els), self.note)

    def _locate(self, target: int):
        expected = None
        if target < len(self._snap.get("items", [])):
            expected = _sig(self._snap["items"][target])
        fresh = self._snapshot()["items"]
        if expected is None or target >= len(fresh) or _sig(fresh[target]) != expected:
            return None, fresh
        return self.page.locator(f'[data-jev="{target}"]').first, fresh

    def act(self, d: Decision) -> ActionResult:
        res = self._act(d)
        self.note = res.note
        return res

    def _act(self, d: Decision) -> ActionResult:
        if d.op == "DONE":
            return ActionResult(True, f"DONE: {d.arg}", terminal=True, success_claimed=True)
        if d.op == "BLOCKED":
            return ActionResult(False, f"BLOCKED: {d.arg}", terminal=True)
        if d.op in ("SCROLL_UP", "SCROLL_DOWN"):
            self.page.mouse.wheel(0, -700 if d.op == "SCROLL_UP" else 700)
            self._settle()
            return ActionResult(True, d.op)
        if d.op == "WAIT":
            self._settle(2000)
            return ActionResult(True, "waited")
        loc, fresh = self._locate(d.target)
        if loc is None:
            return ActionResult(False, f"{d.op} {d.target}: element changed since observation (stale); re-observe")
        item = fresh[d.target]
        if d.op == "CLICK":
            loc.click(timeout=5000)
        elif d.op == "TYPE_TEXT":
            text = self.text_fn(self.goal, f"{item['tag']} '{item['label']}'")[:200]
            loc.fill(text, timeout=5000)
            self._settle(900)  # comboboxes populate suggestions after typing
            return ActionResult(True, f"typed {text!r} into [{d.target}]")
        elif d.op == "SELECT":
            choice = self.text_fn(self.goal, f"select '{item['label']}' options: {item['opts']}").strip()
            match = next((o for o in item["opts"] if o.lower() == choice.lower()), None) or \
                next((o for o in item["opts"] if choice.lower() in o.lower()), None)
            if match is None:
                return ActionResult(False, f"SELECT: {choice!r} is not one of {item['opts']}")
            loc.select_option(label=match, timeout=5000)
            return ActionResult(True, f"selected {match!r}")
        self._settle()
        return ActionResult(True, f"{d.op} [{d.target}] {item['label'][:40]}")

    def _settle(self, ms: int | None = None) -> None:
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:  # noqa: BLE001 - settling is best-effort
            pass
        time.sleep((ms or self.settle_ms) / 1000)

    def close(self) -> None:
        for fn in (self._browser.close, self._pw.stop):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
