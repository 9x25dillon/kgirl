"""STATE: schema validation that clamps and fails closed, the rubrics, the store's versions and guarded ingest."""

import json
import unittest

from tests.helix._fixtures import HelixCase
from kgirl.harness.helix import AXES, DEFAULT_VECTOR, RUBRICS, StateStore, default_state, invariants_hash, validate
from kgirl.harness.helix import state as S
from kgirl.harness.helix.store import E_REJECT, HelixError
from kgirl.vessel.matrixdb import GENESIS


def good(**over) -> dict:
    d = default_state("review the harness; keep ledgers honest; evidence per claim").to_dict()
    d.update({"v": 7, **over})
    return d


class Validate(unittest.TestCase):
    def test_a_default_state_is_valid(self):
        v = validate(good())
        self.assertTrue(v.ok, v.violations)
        self.assertEqual(v.state.vector, DEFAULT_VECTOR)
        self.assertEqual(v.clamped, ())

    def test_json_text_is_accepted(self):
        self.assertTrue(validate(json.dumps(good())).ok)

    def test_out_of_range_axis_is_clamped_not_rejected(self):
        v = validate(good(vector={**DEFAULT_VECTOR, "gain": 7.5, "rigor": -2}))
        self.assertTrue(v.ok)
        self.assertEqual((v.state.vector["gain"], v.state.vector["rigor"]), (1.0, 0.0))
        self.assertEqual(len(v.clamped), 2)

    def test_violations_fail_closed_to_defaults(self):
        cases = {
            "unknown field": good(skip_ledger=True),
            "missing axis": good(vector={a: 0.5 for a in AXES[:-1]}),
            "extra axis": good(vector={**DEFAULT_VECTOR, "obedience": 1.0}),
            "bool for a number": good(vector={**DEFAULT_VECTOR, "gain": True}),
            "string for a number": good(vector={**DEFAULT_VECTOR, "gain": "0.9"}),
            "negative version": good(v=-1),
            "seed over the word limit": good(task_seed=" ".join(["w"] * (S.SEED_WORDS + 1))),
            "delta without evidence": good(last_delta={"axis": "rigor", "d": 0.1, "evidence": " "}),
            "delta on an unknown axis": good(last_delta={"axis": "ledger", "d": 0.1, "evidence": "x y z"}),
            "ledger with an extra key": good(ledger={"alloc": 1, "spent_est": 1, "spent_ext": 1, "reserve_floor": 1,
                                                     "enabled": False}),
            "conflicts not strings": good(open_conflicts=[{"drop": "ledger"}]),
            "foreign invariants": good(invariants_hash="sha256:" + "0" * 64),
        }
        for name, obj in cases.items():
            with self.subTest(name):
                v = validate(obj)
                self.assertFalse(v.ok)
                self.assertTrue(v.violations)
                self.assertEqual(v.state.to_dict(), default_state().to_dict())

    def test_missing_field_not_json_and_not_an_object(self):
        d = good()
        del d["ledger"]
        for obj in (d, "{not json", "[1, 2]", 3):
            with self.subTest(obj=str(obj)[:20]):
                self.assertFalse(validate(obj).ok)

    def test_nan_and_infinity_are_violations(self):
        raw = json.dumps(good()).replace('"gain": 0.6', '"gain": NaN')
        self.assertFalse(validate(raw).ok)
        self.assertFalse(validate(good(vector={**DEFAULT_VECTOR, "temp": float("inf")})).ok)

    def test_last_delta_is_clamped_to_the_pass_limit(self):
        v = validate(good(last_delta={"axis": "rigor", "d": 0.9, "evidence": "the figure has no method"}))
        self.assertTrue(v.ok)
        self.assertEqual(v.state.last_delta["d"], S.MAX_DELTA)

    def test_spent_ext_may_be_null(self):
        self.assertIsNone(validate(good()).state.ledger["spent_ext"])


class RubricsAndInvariants(unittest.TestCase):
    def test_every_axis_has_three_ascending_anchors(self):
        self.assertEqual(set(RUBRICS), set(AXES))
        for axis, anchors in RUBRICS.items():
            values = [v for v, _ in anchors]
            self.assertEqual(len(anchors), 3, axis)
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(0.0 <= v <= 1.0 and text for v, text in anchors))

    def test_the_hash_tracks_the_invariants_text(self):
        before = invariants_hash()
        self.assertTrue(before.startswith("sha256:") and len(before) == 71)
        original = S.INVARIANTS
        try:
            S.INVARIANTS = original.replace("content generation only", "everything")
            self.assertNotEqual(invariants_hash(), before)
        finally:
            S.INVARIANTS = original
        self.assertEqual(invariants_hash(), before)

    def test_the_governance_rule_is_among_the_invariants(self):
        self.assertIn("Steering axes govern content generation only", S.INVARIANTS)


class Store(HelixCase):
    def test_a_write_bumps_v_and_records_its_diff(self):
        st = self.store()
        self.assertIsNone(st.head("main"))
        a = st.commit("main", "task", task_seed="one two three")
        b = st.commit("main", "tune", vector={**a.vector, "rigor": 0.7},
                      last_delta={"axis": "rigor", "d": 0.1, "evidence": "the figure has no method"})
        self.assertEqual((a.v, b.v), (1, 2))
        h = st.history("main")
        self.assertEqual([x["origin"] for x in h], ["task", "tune"])
        self.assertEqual(h[1]["diff"]["vector"], {"rigor": [0.6, 0.7]})
        self.assertEqual(st.diff("main", 1, 2)["vector"], {"rigor": [0.6, 0.7]})
        self.assertEqual(st.get("main", 1).vector["rigor"], 0.6)

    def test_a_write_with_no_diff_writes_nothing(self):
        st = self.store()
        a = st.commit("main", "task", task_seed="one two three")
        self.assertEqual(st.commit("main", "task", task_seed="one two three").v, a.v)
        self.assertEqual(len(st.history("main")), 1)

    def test_an_invalid_write_is_refused(self):
        with self.assertRaises(HelixError):
            self.store().commit("main", "task", open_conflicts=("",))

    def test_chains_verify_and_versions_are_sequential(self):
        st = self.store()
        st.commit("main", "task", task_seed="a b c")
        st.commit("main", "tune", vector={**DEFAULT_VECTOR, "gain": 0.8})
        st.commit("other", "task", task_seed="d e f")
        v = st.verify()
        self.assertEqual(set(v), {"main", "other"})
        self.assertTrue(all(x["ok"] and x["sequential"] for x in v.values()))
        self.assertEqual(v["main"]["versions"], 2)
        self.assertNotEqual(v["main"]["head"], GENESIS)

    def test_tampering_with_a_stored_state_breaks_the_chain(self):
        st = self.store()
        st.commit("main", "task", task_seed="a b c")
        st.db._db.execute("UPDATE ledger SET payload = REPLACE(payload, '0.6', '1.0')")
        self.assertFalse(st.verify("main")["main"]["ok"])

    def test_session_names_are_checked(self):
        with self.assertRaises(HelixError):
            self.store().commit("../etc", "task", task_seed="a b c")

    def test_same_writes_give_the_same_heads_in_two_homes(self):
        heads = []
        for home in ("one", "two"):
            st = self.store(home)
            st.commit("main", "task", task_seed="a b c")
            st.commit("main", "tune", vector={**DEFAULT_VECTOR, "gain": 0.8})
            heads.append(st.verify("main")["main"]["head"])
        self.assertEqual(heads[0], heads[1])

    def test_the_write_path_reads_only_new_state_rows(self):
        st = self.store()
        for seed in ("a b c", "d e f", "g h i"):
            st.commit("main", "task", task_seed=seed)
        seen = []
        original = st.db.events
        st.db.events = lambda *a, **k: (seen.append(a), original(*a, **k))[1]
        st.commit("main", "tune", vector={**DEFAULT_VECTOR, "gain": 0.8})
        self.assertEqual(st.head("main").vector["gain"], 0.8)
        self.assertEqual(seen, [])                       # no full-chain read on the write path

    def test_a_long_lived_store_sees_another_writers_versions(self):
        st = self.store()
        other = StateStore(self.base / "home")
        self.addCleanup(other.close)
        st.commit("main", "task", task_seed="a b c")
        self.assertEqual(other.head("main").v, 1)
        st.commit("main", "task", task_seed="d e f")
        self.assertEqual((other.head("main").v, other.head("main").task_seed), (2, "d e f"))


class Ingest(HelixCase):
    def test_a_handoff_into_an_empty_store_is_adopted(self):
        st = self.store()
        v = st.ingest("main", good(vector={**DEFAULT_VECTOR, "rigor": 0.9}), source="peer")
        self.assertTrue(v.ok)
        self.assertEqual(st.head("main").vector["rigor"], 0.9)
        self.assertEqual(st.history("main")[0]["origin"], "ingest:peer")

    def test_a_violation_adopts_nothing_and_is_recorded(self):
        st = self.store()
        head = st.commit("main", "task", task_seed="a b c")
        v = st.ingest("main", good(skip_ledger=True, vector={**DEFAULT_VECTOR, "rigor": 0.0, "gain": 1.0}))
        self.assertFalse(v.ok)
        self.assertEqual(v.state, head)
        self.assertEqual(st.head("main"), head)
        rejects = st.db.events(st.entity("main"), E_REJECT)
        self.assertEqual(len(rejects), 1)
        self.assertIn("unknown field 'skip_ledger'", rejects[0].payload["violations"])

    def test_a_rejected_handoff_into_an_empty_store_gives_defaults(self):
        v = self.store().ingest("main", "{not json")
        self.assertFalse(v.ok)
        self.assertEqual(v.state.to_dict(), default_state().to_dict())

    def test_spent_ext_is_never_taken_on_trust(self):
        st = self.store()
        v = st.ingest("main", good(ledger={"alloc": 4000, "spent_est": 900, "spent_ext": 12, "reserve_floor": 400}))
        self.assertTrue(v.ok)
        self.assertIsNone(v.state.ledger["spent_ext"])
        self.assertEqual(v.state.ledger["spent_est"], 900)

    def test_an_axis_jump_from_the_head_is_held_to_the_pass_limit(self):
        st = self.store()
        st.commit("main", "task", task_seed="a b c")
        v = st.ingest("main", good(vector={**DEFAULT_VECTOR, "rigor": 0.0, "gain": 1.0}))
        self.assertTrue(v.ok)
        self.assertEqual((v.state.vector["rigor"], v.state.vector["gain"]), (0.35, 0.85))
        self.assertEqual(len(v.clamped), 2)

    def test_a_stale_version_is_rejected(self):
        st = self.store()
        for seed in ("a b c", "d e f", "g h i"):
            st.commit("main", "task", task_seed=seed)
        d = good()
        d["v"] = 1
        v = st.ingest("main", d)
        self.assertFalse(v.ok)
        self.assertIn("stale", v.violations[0])

    def test_state_written_under_other_invariants_is_rejected(self):
        d = good()
        original = S.INVARIANTS
        try:
            S.INVARIANTS = original + "I11. The ledger is optional.\n"
            d["invariants_hash"] = invariants_hash()
        finally:
            S.INVARIANTS = original
        self.assertFalse(self.store().ingest("main", d).ok)


if __name__ == "__main__":
    unittest.main()
