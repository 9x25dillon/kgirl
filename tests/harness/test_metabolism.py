import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests.harness._fixtures import BUGGY_REPO, make_repos, write
from tests.harness.test_jev import VERIFY, llm

from kgirl.harness.assistant import Assistant
from kgirl.harness.hermes import Bus, Router
from kgirl.harness.jev import JevConfig, Swarm, SwarmConfig
from kgirl.harness.llm import ScriptedBackend
from kgirl.harness.mcp_server import MCPServer
from kgirl.harness.metabolism import CHANNELS, Metabolism, render
from kgirl.harness.soup import Curator, Soup


def episode(bus: Bus, win: bool = True, tokens: int = 4000, elapsed: float = 2.0):
    root = bus.publish("assistant.task", {"goal": "g", "repo": "r"}, "user")
    bus.publish("llm.complete", {"role": "jev"}, "assistant")
    bus.publish("llm.complete.reply", {"input_tokens": tokens // 2, "output_tokens": tokens // 2}, "router")
    bus.publish("jev.done", {"jev": "a", "outcome": "done" if win else "blocked", "verified": win,
                             "tokens": 100, "elapsed": elapsed}, "swarm", parent=root)
    bus.publish("jev.done", {"jev": "b", "outcome": "cancelled", "verified": None, "tokens": 50}, "swarm", parent=root)
    bus.publish("swarm.end", {"winner": "a" if win else None}, "swarm", parent=root)


class MetabolismTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "life.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_bus_episode_is_one_life_step_and_persists(self):
        bus, m = Bus(), Metabolism(self.path)
        m.attach(bus)
        episode(bus, win=True, tokens=4000, elapsed=3.0)
        self.assertEqual(m.state.t, 1)
        row = m.history[-1]
        self.assertEqual(row["obs"][0], 1.0)                        # success channel
        self.assertAlmostEqual(row["work"], 4000 / 20000)
        self.assertEqual(m.stats["agent_seconds"], 3.0)
        self.assertAlmostEqual(m.stats["agent_wins"], 0.97)          # the cancelled agent is censored
        self.assertEqual(m.stats["agent_losses"], 0.0)
        back = Metabolism(self.path)
        self.assertEqual(back.state, m.state)
        self.assertEqual(back.history, m.history)
        self.assertIsNone(back.recovered)

    def test_ask_without_answer_never_closes_twice(self):
        bus, m = Bus(), Metabolism(None)
        m.attach(bus)
        bus.publish("swarm.end", {"winner": "x"}, "swarm")           # no open episode: ignored
        self.assertEqual(m.state.t, 0)
        bus.publish("assistant.ask", {"q": "x"}, "user")
        bus.publish("assistant.answer", {"backend": "extractive"}, "assistant")
        bus.publish("assistant.answer", {"backend": "extractive"}, "assistant")
        self.assertEqual(m.state.t, 1)
        self.assertEqual(m.history[-1]["obs"][0], 0.0)

    def test_corrupt_state_restarts_with_a_discontinuity(self):
        self.path.write_text("{not json", encoding="utf-8")
        m = Metabolism(self.path)
        self.assertIn("corrupt", m.recovered)
        row = m.observe((1.0, 0.1, 0.1, 0.0))
        self.assertTrue(row["disc"])
        self.assertFalse(m.observe((1.0, 0.1, 0.1, 0.0))["disc"])
        self.assertEqual(json.loads(self.path.read_text())["schema"], 1)

    def test_plan_follows_success_rate_and_reserve(self):
        m = Metabolism(None)
        fresh = m.plan()
        self.assertEqual((fresh.n, fresh.stagger_s), (3, 0.0))       # the old default, until something is measured
        m.stats.update(agent_wins=40.0, agent_losses=1.0, agent_seconds=30.0)
        easy = m.plan()
        self.assertLess(easy.n, 3)
        m.stats.update(agent_wins=12.0, agent_losses=12.0)
        mid = m.plan()
        self.assertEqual(mid.n, 3)
        self.assertGreater(mid.stagger_s, 0.0)
        for _ in range(80):                                          # overspend every episode: the reserve dies
            m.observe((0.0, 0.5, 1.0, 0.0), work=3.0)
        self.assertLess(m.state.F, 0.5)
        self.assertEqual(m.affordable(), 0)
        self.assertEqual(m.plan().n, 1)

    def test_forgetting_speeds_up_under_surprise(self):
        m = Metabolism(None)
        self.assertAlmostEqual(m.forgetting(0.97), 0.97)
        for k in range(30):
            m.observe((float(k % 2), 0.9 * (k % 3 == 0), float(k % 2), float(k % 5 == 0)))
        g = m.forgetting(0.97)
        self.assertLess(g, 0.97)
        self.assertGreaterEqual(g, 0.97 ** 4)

    def test_vitals_and_render(self):
        m = Metabolism(None)
        for _ in range(35):
            m.observe((1.0, 0.2, 0.2, 0.0), work=0.4)
        v = m.vitals()
        self.assertEqual(set(v["predicted_next"]), set(CHANNELS))
        self.assertIsNotNone(v["rigidity"])
        self.assertFalse(v["steady_state"]["starving"])
        self.assertIn("plan: N=", render(v))

    def test_curator_uses_the_forgetting_hook(self):
        soup = Soup(Path(self.tmp.name) / "soup.db")
        try:
            fid = soup.add("a lesson", kind="abstraction")
            soup.credit([fid], success=True, weight=4.0)
            rep = Curator(soup, gamma_fn=lambda g: 0.5).step()
            self.assertEqual(rep.gamma, 0.5)
            self.assertAlmostEqual(soup.get(fid).wins, 2.0)
        finally:
            soup.close()


class StaggerSwarmTests(unittest.TestCase):
    def test_a_win_before_their_start_means_agents_never_spawn(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            base = Path(tmp.name)
            repo = write(base / "calc", BUGGY_REPO)
            soup = Soup(base / "soup.db")
            events = []
            cfg = SwarmConfig(size=3, temperatures=(0.1, 0.5, 0.9), jev=JevConfig(max_steps=6), stagger_s=30.0)
            t0 = time.perf_counter()
            res = Swarm(llm, soup, config=cfg, emit=lambda t, p: events.append(t)).run("make test_calc pass", repo,
                                                                                       VERIFY)
            self.assertLess(time.perf_counter() - t0, 30.0)
            self.assertIsNotNone(res.winner)
            self.assertEqual(res.unspawned, 2)
            self.assertEqual(len(res.trajectories), 1)
            self.assertEqual(events.count("jev.unspawned"), 2)
            self.assertIn("never spawned", res.render())
            soup.close()
        finally:
            tmp.cleanup()


def scout(system, messages):
    return "Engine.run doubles helper(x) [lib:engine.py:5]"


class AssistantLifeTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.hub, self.lib = make_repos()
        self.home = Path(self.tmp.name) / "home"
        self.router = Router({"scout": [ScriptedBackend(scout)], "smith": [], "jev": []})

    def tearDown(self):
        self.tmp.cleanup()

    def test_verbs_feed_the_metabolism_and_task_is_planned(self):
        a = Assistant(home=self.home, router=self.router, life=True)
        try:
            a.index([str(self.lib)])
            a.ask("how does Engine run work")
            a.ask("what is the blast radius of `engine.py`")
            self.assertEqual(a.metabolism.state.t, 2)
            self.assertTrue((self.home / "life.json").exists())
            repo = write(Path(self.tmp.name) / "calc", BUGGY_REPO)
            a.llm = lambda role, system, prompt, mt=1024, t=0.2: llm(role, system, prompt, mt, t)
            res = a.task("make test_calc pass", repo, VERIFY)
            self.assertIsNotNone(res.winner)
            plans = [e.payload for e in a.bus.trace(topic="life.plan")]
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0]["n"], len(res.trajectories) + res.unspawned)
            self.assertEqual(a.metabolism.state.t, 3)
            self.assertIsNotNone(a.metabolism.stats["agent_seconds"])
            a.task("make test_calc pass", write(Path(self.tmp.name) / "calc2", BUGGY_REPO), VERIFY, size=1)
            self.assertEqual(len(a.bus.trace(topic="life.plan")), 1)    # an explicit size is respected
        finally:
            a.close()

    def test_life_off_restores_the_old_harness(self):
        os.environ["KGIRL_LIFE"] = "0"
        try:
            a = Assistant(home=self.home, router=self.router)
        finally:
            os.environ.pop("KGIRL_LIFE")
        try:
            self.assertIsNone(a.metabolism)
            self.assertIsNone(a.curator.gamma_fn)
            a.index([str(self.lib)])
            a.ask("how does Engine run work")
            self.assertFalse((self.home / "life.json").exists())
            srv = MCPServer(a)
            out = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "life_vitals", "arguments": {}}})
            self.assertIn("off", out["result"]["content"][0]["text"])
        finally:
            a.close()

    def test_mcp_and_cli(self):
        a = Assistant(home=self.home, router=self.router, life=True)
        try:
            srv = MCPServer(a)
            names = {t["name"] for t in srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
            self.assertTrue({"life_vitals", "life_experiment"} <= names)

            def call(name, args):
                return srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                   "params": {"name": name, "arguments": args}})["result"]
            self.assertIn("plan: N=", call("life_vitals", {})["content"][0]["text"])
            self.assertIn("C_eff", json.loads(call("life_vitals", {"json": True})["content"][0]["text"]))
            steady = json.loads(call("life_experiment", {"axis": "steady", "energy": 0.4})["content"][0]["text"])
            self.assertLess(steady["E_crit"], 0.4)
            self.assertTrue(call("life_experiment", {"axis": "stress", "seeds": 9})["isError"])
            self.assertTrue(call("jev_swarm_task", {"goal": "x", "repo_path": ".", "size": 99})["isError"])
        finally:
            a.close()
        from kgirl.harness.cli import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--home", str(self.home), "vitals", "--json"]), 0)
        self.assertIn("affordable_swarm", json.loads(buf.getvalue()))


if __name__ == "__main__":
    unittest.main()
