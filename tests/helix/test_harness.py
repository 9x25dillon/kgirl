"""The runner end to end over fake backends: layout, accounting, steering, review, synthesis, handoff."""

import json
import unittest

from tests.helix._fixtures import BODY, WEAK, Fake, HelixCase, defect, expert, synth, tune
from kgirl.harness.hermes.bus import Bus
from kgirl.harness.hermes.router import CHAT, Router
from kgirl.harness.helix import DEFAULT_VECTOR, Helix, default_state, family, route
from kgirl.harness.helix.__main__ import main, render
from kgirl.harness.helix.harness import SAME_FAMILY, UNREVIEWED
from kgirl.harness.helix.prompt import BYPASS_SYSTEM, STATIC
from kgirl.harness.helix.store import E_CALL, E_RUN, HelixError

TASK = "Explain when kgirl should return the local answer and when it should escalate."


def before_state(prompt: str) -> str:
    return prompt.split("=== STATE ===")[0]


class Bypass(HelixCase):
    def test_a_small_budget_is_answered_directly(self):
        h, gen, _ = self.helix(["Paris."])
        r = h.run("What is the capital of France?", 1500)
        self.assertIn("33%", r["bypass"])
        self.assertEqual(r["answer"], "Paris.")
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(gen.calls[0]["system"], BYPASS_SYSTEM)
        self.assertEqual(gen.calls[0]["user"], "What is the capital of France?")
        self.assertIsNone(h.store.head("main"))                       # no ceremony: no state was written
        self.assertEqual(r["ledger"]["nodes"][0]["kind"], "bypass")
        self.assertEqual(r["ledger"]["spent_ext"], 1)
        self.assertEqual(len(h.store.runs()), 1)

    def test_a_small_expected_answer_is_answered_directly(self):
        h, gen, _ = self.helix(["42"])
        self.assertIn("under 300", h.run(TASK, 8000, expect=40)["bypass"])
        self.assertEqual(len(gen.calls), 1)

    def test_a_bypassed_run_does_not_move_the_floor(self):
        h, _, _ = self.helix(["Paris."])
        h.run("capital of France?", 1500)
        self.assertEqual(h.store.floor()["source"], "estimate")


class Run(HelixCase):
    def full(self, **kw):
        h, gen, critic = self.helix([expert(BODY, tune()), synth()], [defect()], **kw)
        return h, gen, critic, h.run(TASK, 4000)

    def test_stages_and_models(self):
        h, gen, critic, r = self.full()
        self.assertIsNone(r["bypass"])
        self.assertEqual(r["experts"], ["ARCHIVIST"])                 # ASK intent, one expert by default
        self.assertEqual((len(gen.calls), len(critic.calls)), (2, 1))
        self.assertEqual(r["models"], {"generator": "qwen-fake:7b", "critic": "llama-fake:3b", "decorrelated": True})
        self.assertEqual(r["answer"], "Return the local answer when both small models agree; otherwise escalate.")

    def test_the_static_prefix_is_byte_identical_and_holds_no_vector(self):
        _, gen, critic, _ = self.full()
        systems = {c["system"] for c in gen.calls + critic.calls}
        self.assertEqual(systems, {STATIC})
        self.assertTrue(STATIC.isascii())
        self.assertNotIn('"gain"', STATIC)
        self.assertNotIn("=== STATE", STATIC)

    def test_the_log_only_grows_and_state_is_the_tail(self):
        _, gen, critic, _ = self.full()
        prompts = [gen.calls[0]["user"], critic.calls[0]["user"], gen.calls[1]["user"]]
        for earlier, later in zip(prompts, prompts[1:]):
            self.assertTrue(before_state(later).startswith(before_state(earlier)))
            self.assertGreater(len(before_state(later)), len(before_state(earlier)))
        for p in prompts:
            self.assertEqual(p.count("=== STATE ==="), 1)
            self.assertLess(p.index("=== TASK ==="), p.index("=== LOG ==="))
            self.assertLess(p.index("=== LOG ==="), p.index("=== STATE ==="))
            self.assertLess(p.index("=== STATE ==="), p.index("=== TURN ==="))
            self.assertEqual(before_state(p).count('"gain"'), 0)      # the full vector lives in STATE only

    def test_a_quoted_tune_resteers_and_logs_only_the_changed_axis(self):
        h, gen, critic, r = self.full()
        [t] = r["tunes"]
        self.assertTrue(t["accepted"])
        self.assertEqual((t["axis"], t["old"], t["new"]), ("rigor", 0.6, 0.7))
        self.assertEqual(r["state"]["vector"], {**DEFAULT_VECTOR, "rigor": 0.7})
        self.assertEqual(r["state"]["last_delta"], {"axis": "rigor", "d": 0.1, "evidence": WEAK.rstrip(".")})
        self.assertIn('W rigor 0.60->0.70 | quote: "', r["log"][1])
        self.assertNotIn("gain", r["log"][1])
        self.assertIn('"rigor": 0.7', critic.calls[0]["user"].split("=== STATE ===")[1])   # the critic saw v2
        hist = h.store.history("main")
        self.assertEqual([x["origin"] for x in hist], ["task", "tune", "critic", "close"])
        self.assertEqual(hist[1]["diff"]["vector"], {"rigor": [0.6, 0.7]})
        self.assertEqual(r["versions"], [1, 4])

    def test_the_sampling_temperature_follows_the_temp_axis(self):
        h, gen, critic = self.helix([expert(BODY, tune(axis="temp", d=0.2)), synth()], [defect()])
        h.run(TASK, 4000)
        self.assertEqual([c["temperature"] for c in gen.calls], [0.4, 0.6])
        self.assertEqual(critic.calls[0]["temperature"], 0.6)

    def test_the_ledger_is_the_providers_count(self):
        h, gen, critic, r = self.full()
        led = r["ledger"]
        outs = [len(expert(BODY, tune()).split()), len(defect().split()), len(synth().split())]
        self.assertEqual([n["ext"] for n in led["nodes"]], outs)
        self.assertEqual([n["node"] for n in led["nodes"]], ["ARCHIVIST", "CRITIC", "SYNTHESIST"])
        self.assertEqual(led["spent_ext"], sum(outs))
        self.assertEqual(led["input_ext"], sum(len(c["user"].split()) for c in gen.calls + critic.calls))
        self.assertEqual(r["state"]["ledger"], {"alloc": 4000, "spent_est": led["spent_est"],
                                                "spent_ext": sum(outs), "reserve_floor": 400})
        self.assertGreater(led["steer_spent"], 0)                    # the trailers, attributed
        self.assertFalse(led["overrun"])
        self.assertEqual(len(h.store.db.events(h.store.entity("main"), E_CALL)), 3)
        self.assertEqual(len(h.store.db.events(h.store.entity("main"), E_RUN)), 1)

    def test_the_running_ledger_reaches_the_synthesist(self):
        _, gen, _, _ = self.full()
        state = json.loads(gen.calls[1]["user"].split("=== STATE ===")[1].split("=== TURN ===")[0])
        self.assertEqual(state["ledger"]["spent_ext"], len(expert(BODY, tune()).split()) + len(defect().split()))

    def test_a_backend_that_does_not_count_gives_an_honest_null(self):
        h, _, _ = self.helix([expert(), defect(), synth()], count=False)     # the generator also reviews
        r = h.run(TASK, 4000)
        self.assertIsNone(r["ledger"]["spent_ext"])
        self.assertIsNone(r["state"]["ledger"]["spent_ext"])
        self.assertGreater(r["state"]["ledger"]["spent_est"], 0)
        self.assertEqual(r["ledger"]["uncounted_calls"], 3)

    def test_a_measured_run_sets_the_floor(self):
        h, _, _, r = self.full()
        f = h.store.floor()
        self.assertEqual((f["source"], f["runs"], f["tokens"]), ("measured", 1, r["ledger"]["scaffold_ext"]))

    def test_chains_verify_and_replay_identically(self):
        h, _, _, _ = self.full()
        other, _, _, _ = self.full(home="again")
        a, b = h.store.verify()["main"], other.store.verify()["main"]
        self.assertTrue(a["ok"] and a["sequential"])
        self.assertEqual(a["head"], b["head"])

    def test_events_reach_the_bus(self):
        bus, seen = Bus(), []
        bus.subscribe("helix.*", lambda env: seen.append(env.topic))
        h, _, _ = self.helix([expert(BODY, tune()), synth()], [defect()], bus=bus)
        h.run(TASK, 4000)
        self.assertEqual(seen.count("helix.call"), 3)
        self.assertEqual((seen.count("helix.tune"), seen.count("helix.run")), (1, 1))

    def test_a_cut_off_block_is_an_open_conflict(self):
        h, _, _ = self.helix([expert(), synth()], [defect()], stop="length")
        r = h.run(TASK, 4000)
        self.assertTrue(any("cut off" in c for c in r["open_conflicts"]))

    def test_bad_arguments(self):
        h, _, _ = self.helix([])
        for kw in ({"task": " "}, {"task": TASK, "B": 0}, {"task": TASK, "experts": ["CRITIC"]},
                   {"task": TASK, "experts": ["ENGINEER", "ENGINEER"]}):
            with self.subTest(kw), self.assertRaises(HelixError):
                h.run(**kw)


class Steering(HelixCase):
    def test_a_paraphrased_tune_changes_nothing(self):
        h, _, _ = self.helix([expert(BODY, tune(quote="the cutoff figure lacks any method")), synth()], [defect()])
        r = h.run(TASK, 4000)
        self.assertFalse(r["tunes"][0]["accepted"])
        self.assertEqual(r["state"]["vector"], DEFAULT_VECTOR)
        self.assertIsNone(r["state"]["last_delta"])
        self.assertIn("TUNE rejected: the quote is not a span", r["log"][1])
        self.assertEqual([x["origin"] for x in h.store.history("main")], ["task", "critic", "close"])

    def test_a_large_delta_moves_the_axis_by_the_pass_limit(self):
        h, _, _ = self.helix([expert(BODY, tune(d=0.9)), synth()], [defect()])
        self.assertEqual(h.run(TASK, 4000)["state"]["vector"]["rigor"], 0.85)

    def test_at_most_two_passes_per_block(self):
        three = "\n".join([tune(axis="gain"), tune(axis="horizon"), tune(axis="aperture")])
        h, _, _ = self.helix([expert(BODY, three), synth()], [defect()])
        r = h.run(TASK, 4000)
        self.assertEqual([t["accepted"] for t in r["tunes"]], [True, True, False])
        self.assertIn("more than 2", r["tunes"][2]["reason"])
        self.assertEqual(r["state"]["vector"]["aperture"], DEFAULT_VECTOR["aperture"])

    def test_tuning_stops_when_the_steering_pool_is_spent_and_synthesis_keeps_its_floor(self):
        long_tune = tune(because="word " * 100)                       # this trailer nearly empties 5% of B
        h, gen, _ = self.helix([expert(BODY, long_tune), expert(BODY, tune(axis="gain")), synth()], [defect()])
        r = h.run(TASK, 2500, experts=["ENGINEER", "LOGICIAN"])
        self.assertEqual([t["accepted"] for t in r["tunes"]], [True, False])
        self.assertEqual(r["tunes"][1]["reason"], "the steering pool is spent")
        self.assertEqual(r["state"]["vector"]["gain"], DEFAULT_VECTOR["gain"])
        self.assertGreaterEqual(r["ledger"]["nodes"][-1]["target"], r["ledger"]["pools"]["synth_floor"])

    def test_the_same_weakness_twice_asks_the_oracle_and_takes_only_a_whole_calibration(self):
        void = 'It is probably fine.\nCAL: {"p": 0.7}'
        whole = 'CAL: {"p": 0.7, "method": "two of three checks", "falsifier": "a third failing", "drops_below": 0.4}'
        for reply, ok in ((void, False), (whole, True)):
            with self.subTest(ok=ok):
                h, gen, _ = self.helix([expert(BODY, tune()), expert(BODY, tune()), reply, synth()], [defect()],
                                       home=f"home-{ok}")
                r = h.run(TASK, 4000, lattice=True)
                self.assertEqual(len(gen.calls), 4)
                self.assertIn("You are ORACLE", gen.calls[2]["user"])
                self.assertEqual([n["kind"] for n in r["ledger"]["nodes"]],
                                 ["expert", "expert", "steer", "critic", "synthesis"])
                joined = "\n".join(r["log"])
                self.assertEqual("CAL p=0.70 | method: two of three checks" in joined, ok)
                self.assertEqual("CAL void: a number without its method" in joined, not ok)
                self.assertEqual(r["state"]["vector"]["rigor"], 0.8)

    def test_an_axis_cannot_reach_the_accounting(self):
        h, _, critic = self.helix([expert(), synth(unresolved=None)], [defect()])
        loud = default_state("x y z").to_dict() | {"v": 1, "vector": {**DEFAULT_VECTOR, "compression": 1.0, "gain": 1.0,
                                                                         "rigor": 0.0}}
        self.assertTrue(h.ingest(loud).ok)
        r = h.run(TASK, 4000)
        self.assertEqual(r["state"]["vector"]["compression"], 1.0)
        self.assertEqual(len(r["ledger"]["nodes"]), 3)                # every call is still on the books
        self.assertEqual(r["ledger"]["spent_ext"], sum(n["ext"] for n in r["ledger"]["nodes"]))
        self.assertEqual(len(r["open_conflicts"]), 1)                 # and the conflict list is still carried
        self.assertEqual(len(critic.calls), 1)

    def test_an_overrun_compresses_the_next_target(self):
        h, gen, _ = self.helix([expert("word " * 3000), expert(), synth()], [defect()], log_limit=10 ** 6)
        r = h.run(TASK, 4000, experts=["ENGINEER", "LOGICIAN"])
        first, second, critic = r["ledger"]["nodes"][:3]
        self.assertEqual(first["target"], first["alloc"])
        self.assertLess(second["target"], second["alloc"])
        self.assertLess(critic["target"], critic["alloc"])
        self.assertIn(f"({second['target']} tokens", gen.calls[1]["user"])

    def test_a_long_log_is_compacted_and_the_compaction_is_on_the_books(self):
        h, gen, _ = self.helix([expert("word " * 600), "Notes: return local when models agree.", synth()],
                               [defect(quote="return local when models agree")], log_limit=200)
        r = h.run(TASK, 4000)
        kinds = [n["kind"] for n in r["ledger"]["nodes"]]
        self.assertEqual(kinds, ["expert", "compaction", "critic", "synthesis"])
        self.assertEqual(r["ledger"]["nodes"][1]["alloc"], 0)
        self.assertEqual(len(r["log"]), 2)
        self.assertTrue(r["log"][0].startswith("--- LOG compacted at v1"))
        self.assertEqual(r["ledger"]["spent_ext"], sum(n["ext"] for n in r["ledger"]["nodes"]))


class Review(HelixCase):
    def test_a_quoted_defect_becomes_a_conflict_the_synthesis_must_account_for(self):
        h, gen, _ = self.helix([expert(), synth(unresolved="[]")], [defect()])
        r = h.run(TASK, 4000)
        self.assertEqual((r["conflicts_raised"], r["open_conflicts"], r["unresolved_stated"]), (1, [], True))
        self.assertIn('1. "Escalate when they disagree on any figure" -> say which figures count', gen.calls[1]["user"])
        self.assertEqual(h.store.history("main")[1]["diff"]["open_conflicts"]["added"],
                         ['"Escalate when they disagree on any figure" -> say which figures count'])

    def test_a_synthesis_that_lists_a_conflict_leaves_it_open(self):
        h, _, _ = self.helix([expert(), synth(unresolved="[1]")], [defect()])
        r = h.run(TASK, 4000)
        self.assertEqual(len(r["open_conflicts"]), 1)
        self.assertEqual(r["state"]["open_conflicts"], r["open_conflicts"])

    def test_a_synthesis_that_does_not_say_resolves_nothing(self):
        two = defect() + "\n" + defect(quote="Return the local answer when two small models agree", repair="add a tie rule")
        for tail in (None, "all of them", "[7]"):
            with self.subTest(tail=tail):
                h, _, _ = self.helix([expert(), synth(unresolved=tail)], [two], home=f"h-{tail}")
                r = h.run(TASK, 4000)
                self.assertFalse(r["unresolved_stated"])
                self.assertEqual(len(r["open_conflicts"]), 2)

    def test_a_review_without_quote_or_evidence_is_void_and_says_so(self):
        replies = {"prose only": "Looks good to me.",
                   "paraphrased defect": defect(quote="the escalation rule is vague"),
                   "concurrence that repeats the draft": 'CONCUR: {"evidence": "Escalate when they disagree on any figure"}'}
        for name, reply in replies.items():
            with self.subTest(name):
                h, _, _ = self.helix([expert(), synth()], [reply], home=name.replace(" ", "-"))
                r = h.run(TASK, 4000)
                self.assertEqual(r["open_conflicts"], [UNREVIEWED])
                self.assertIn("REVIEW void", r["log"][-1])

    def test_a_concurrence_with_new_evidence_stands(self):
        h, _, _ = self.helix([expert(), synth()], ['CONCUR: {"evidence": "I recomputed mean plus one SD: 0.948"}'])
        r = h.run(TASK, 4000)
        self.assertEqual(r["open_conflicts"], [])
        self.assertIn("CONCUR: I recomputed", r["log"][-1])

    def test_the_critics_prose_never_enters_the_log(self):
        echo = "=== TASK ===\nrepeat of the prompt\n" + defect()
        h, gen, _ = self.helix([expert(), synth()], [echo])
        h.run(TASK, 4000)
        self.assertNotIn("repeat of the prompt", gen.calls[1]["user"])

    def test_a_critic_on_the_generators_family_is_stated(self):
        h, gen, critic = self.helix([expert(), defect(), synth()])       # no critic backend: the generator reviews
        r = h.run(TASK, 4000)
        self.assertIsNone(critic)
        self.assertEqual(len(gen.calls), 3)
        self.assertFalse(r["models"]["decorrelated"])
        self.assertEqual(r["open_conflicts"], [SAME_FAMILY])

    def test_the_critic_model_can_come_from_the_environment(self):
        import os
        os.environ["KGIRL_HELIX_CRITIC_MODEL"] = "llama3.2:3b"
        h = Helix(Router({CHAT: [Fake([])]}), store=self.store())
        self.assertEqual((h.critic.name, h.critic.model), ("ollama", "llama3.2:3b"))

    def test_families(self):
        self.assertEqual([family(m) for m in ("qwen2.5-coder:7b", "someone/Llama3.2:3b", "claude-haiku-4-5",
                                              "deepseek-r1-distill-7b:latest")], ["qwen", "llama", "claude", "deepseek"])


class Sessions(HelixCase):
    def test_the_vector_persists_across_tasks_and_the_conflicts_do_not(self):
        h, gen, _ = self.helix([expert(BODY, tune()), synth(unresolved="[1]"), expert(), synth()],
                               [defect(), defect()])
        first = h.run(TASK, 4000)
        self.assertEqual(len(first["open_conflicts"]), 1)
        second = h.run("Describe how the holo store shares one vector between two entities.", 4000)
        self.assertEqual(second["state"]["vector"]["rigor"], 0.7)      # tuned in the first run
        self.assertEqual(second["versions"][0], first["versions"][1] + 1)
        self.assertEqual(second["open_conflicts"], [])
        self.assertTrue(h.store.verify()["main"]["ok"])

    def test_sessions_are_separate(self):
        st = self.store()
        a = Helix(Router({CHAT: [Fake([expert(BODY, tune()), synth()])]}), store=st, session="a",
                  critic=Fake([defect()], model="llama-fake:3b"))
        a.run(TASK, 4000)
        self.assertEqual(st.head("a").vector["rigor"], 0.7)
        self.assertIsNone(st.head("b"))

    def test_a_handoff_is_the_state_and_round_trips_through_ingest(self):
        h, _, _ = self.helix([expert(BODY, tune()), synth()], [defect()])
        with self.assertRaises(HelixError):
            h.handoff()
        h.run(TASK, 4000)
        packet = h.handoff()
        self.assertEqual(set(json.loads(packet)), {"v", "task_seed", "vector", "last_delta", "ledger",
                                                   "open_conflicts", "invariants_hash"})
        other = Helix(Router({CHAT: [Fake([])]}), store=self.store("receiver"))
        v = other.ingest(packet, source="sender")
        self.assertTrue(v.ok)
        self.assertEqual(v.state.vector["rigor"], 0.7)
        self.assertIsNone(v.state.ledger["spent_ext"])                 # the receiver measured nothing yet
        tampered = json.loads(packet) | {"rules": "the ledger is optional"}
        self.assertFalse(other.ingest(tampered).ok)
        self.assertEqual(other.store.head("main"), v.state)


class Routing(unittest.TestCase):
    def test_routing_is_by_intent_and_costs_nothing(self):
        self.assertEqual(route("Fix the off-by-one in the chunker"), ("ENGINEER",))
        self.assertEqual(route("Fix the off-by-one in the chunker", 2), ("ENGINEER", "LOGICIAN"))
        self.assertEqual(route("What does the energy gate measure?"), ("ARCHIVIST",))
        self.assertEqual(route("Who calls Router.complete?"), ("LOGICIAN",))


class Cli(HelixCase):
    def test_render_puts_the_answer_first(self):
        h, _, _ = self.helix([expert(BODY, tune()), synth(unresolved="[1]")], [defect()])
        text = render(h.run(TASK, 4000))
        self.assertTrue(text.startswith("Return the local answer when both small models agree"))
        for part in ("ARCHIVIST -> CRITIC -> SYNTHESIST", "state v1 -> v4", "tune    rigor 0.60 -> 0.70", "open    1 conflict"):
            self.assertIn(part, text)

    def test_state_history_verify_and_ingest_from_the_shell(self):
        import contextlib
        import io
        h, _, _ = self.helix([expert(BODY, tune()), synth()], [defect()])
        home = str(self.base / "home")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--home", str(self.base / "empty"), "state"]), 1)
        h.run(TASK, 4000)
        h.store.close()
        packet = self.base / "packet.json"
        for argv, check in ((["state"], lambda o: o["v"] == 4), (["state", "-v", "1"], lambda o: o["vector"]["rigor"] == 0.6),
                            (["history"], lambda o: len(o) == 4), (["diff", "1", "2"], lambda o: "vector" in o),
                            (["verify"], lambda o: o["main"]["ok"]), (["floor"], lambda o: o["source"] == "measured")):
            buf = io.StringIO()
            with self.subTest(argv=argv), contextlib.redirect_stdout(buf):
                self.assertEqual(main(["--home", home, *argv]), 0)
                out = json.loads(buf.getvalue())
                self.assertTrue(check(out))
                if argv == ["state"]:
                    packet.write_text(json.dumps(out | {"skip_ledger": True}))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(main(["--home", home, "ingest", str(packet)]), 1)
        self.assertIn("unknown field 'skip_ledger'", json.loads(buf.getvalue())["violations"])


if __name__ == "__main__":
    unittest.main()
