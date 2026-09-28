"""Browser Jev against a local page (skipped when Playwright/Chromium are absent)."""

import os
import re
import tempfile
import unittest
from pathlib import Path

import tests.harness._fixtures  # noqa: F401
from kgirl.harness.jev import Jev, JevConfig

try:
    import playwright  # noqa: F401
    from kgirl.harness.jev.browser_env import BrowserEnvironment
    HAVE_PW = True
except ImportError:  # pragma: no cover
    HAVE_PW = False

CHROME = next((p for p in ("/opt/pw-browsers/chromium", os.environ.get("KGIRL_CHROMIUM", "")) if p and
               Path(p).exists() and Path(p).is_file()), None)

PAGE = """<!doctype html><title>Flights</title>
<label for=from>From</label><input id=from placeholder="From city">
<label for=to>To</label><input id=to placeholder="To city">
<select id=cls aria-label="Cabin"><option>Economy</option><option>Business</option></select>
<button onclick="document.getElementById('out').textContent =
  'Results: ' + from.value + ' -> ' + to.value + ' (' + cls.value + ')'">Search</button>
<p id=out></p>"""


def decide(system, prompt):
    def idx(label):
        m = re.search(r"^\[(\d+)\] \w+\s+" + re.escape(label), prompt, re.M)
        return int(m.group(1))
    if "Results:" in prompt:
        return "DONE :: search submitted"
    if "selected 'Business'" in prompt:
        return f"CLICK {idx('Search')}"
    if "value='London'" in prompt:
        return f"SELECT {idx('Cabin')}"
    if "value='Zurich'" in prompt:
        return f"TYPE_TEXT {idx('To')}"
    return f"TYPE_TEXT {idx('From')}"


def text_fn(goal, element):
    if "From" in element:
        return "Zurich"
    if "To" in element:
        return "London"
    return "Business"


@unittest.skipUnless(HAVE_PW, "playwright not installed")
class BrowserJevTests(unittest.TestCase):
    def test_flight_search_form(self):
        tmp = tempfile.TemporaryDirectory()
        page = Path(tmp.name) / "f.html"
        page.write_text(PAGE)
        kw = {"executable_path": CHROME} if CHROME else {}
        try:
            env = BrowserEnvironment("search Zurich to London business", page.as_uri(), text_fn, **kw)
        except Exception as exc:  # no browser binary in this environment
            self.skipTest(f"chromium unavailable: {exc}")
        try:
            traj = Jev(env, decide, config=JevConfig(max_steps=8)).run(env.goal)
            self.assertEqual(traj.outcome, "done", [(s.raw, s.note) for s in traj.steps])
            self.assertEqual([s.op for s in traj.steps], ["TYPE_TEXT", "TYPE_TEXT", "SELECT", "CLICK", "DONE"])
            self.assertEqual(env.page.inner_text("#out"), "Results: Zurich -> London (Business)")
        finally:
            env.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
