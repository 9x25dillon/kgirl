"""Offline tests for the holo store: entities, derived matrices, overlap, provenance, determinism.

A synthetic KGVM matrix and chunk index in a temporary home (tests/holo/_fixtures.py); no model, no
network, no live ~/.kgirl state. Randomness is ChaosRNG only.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.holo._fixtures import (ALPHA, DIM, ENTITIES, MODEL, N, RECORDS_ALL, RECORDS_AT_3, SAMPLED_AT_3, FakeAtlas,
                                  kind_of, make_index, paths)
from kgirl.harness.atlas.vectors import Matrix
from kgirl.harness.hermes.bus import Bus
from kgirl.harness.holo import HoloStore, build, status
from kgirl.harness.holo.build import locate
from kgirl.harness.holo.store import (CENTROID, CHUNK, E_BUILT, E_RECORD, EVIDENCE, ROOT, SPREAD, kind_entity,
                                      repo_entity)
from kgirl.harness.soup import Soup
from kgirl.vessel.matrixdb import GENESIS

WITH_GAMMA = ENTITIES + ["repo:gamma"]


class HoloCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ)                    # the user's own index must not leak in
        env.start()
        self.addCleanup(env.stop)
        for key in ("KGIRL_CHUNKS_DB", "KGIRL_MATRIX", "KGIRL_HOME"):
            os.environ.pop(key, None)
        self.base = Path(self.tmp.name)

    def home(self, name: str = "home") -> Path:
        home = self.base / name
        self.vecs = make_index(home)
        return home

    def open(self, home: Path) -> HoloStore:
        store = HoloStore(home)
        self.addCleanup(store.close)
        return store

    def built(self, name: str = "home", atlas=FakeAtlas(), **kw) -> tuple[HoloStore, dict]:
        home = self.home(name)
        store = self.open(home)
        kw.setdefault("per_cell", 3)
        return store, store.build_from_index(atlas, *paths(home), **kw)


class BuildTests(HoloCase):
    def setUp(self):
        super().setUp()
        self.store, self.report = self.built()
        self.db, self.kgvm = paths(self.base / "home")

    def test_one_entity_per_indexed_repository_and_per_symbol_kind(self):
        self.assertEqual(self.store.entities(), WITH_GAMMA)
        self.assertEqual(self.store.vocabulary(), WITH_GAMMA)
        self.assertEqual(self.store.entities("repo"), ["repo:alpha", "repo:beta", "repo:gamma"])
        self.assertEqual(self.report["entities"], {"kind": 3, "repo": 3})
        self.assertEqual(self.store.db.genome("repo:alpha"), {"holo": 1, "type": "repo", "name": "alpha"})
        self.assertEqual(self.store.db.genome(ROOT), {"holo": 1, "type": "holo"})

    def test_the_report_counts_what_was_built(self):
        r = self.report
        self.assertEqual((r["chunks"], r["orphans"], r["sampled"], r["records"], r["unchanged"], r["overlaps"]),
                         (N, 1, SAMPLED_AT_3, RECORDS_AT_3, 0, SAMPLED_AT_3))
        self.assertEqual(r["matrix"], {"path": str(self.kgvm.resolve()), "count": N + 1, "dim": DIM, "model": MODEL,
                                       "scale": float(Matrix(self.kgvm).scale)})
        self.assertEqual(r["bytes"]["matrices"], 2 * len(ENTITIES) + SAMPLED_AT_3)   # each vector is stored once
        self.assertEqual(r["bytes"], self.store.db.storage())
        self.assertEqual((r["per_cell"], r["seed"]), (3, 0.4123))

    def test_centroid_and_spread_come_from_the_vectors_already_there(self):
        vals = Matrix(self.kgvm).vals
        for entity, rows in ((repo_entity("alpha"), list(range(ALPHA))),
                             (repo_entity("beta"), list(range(ALPHA, N))),
                             (kind_entity("function"), [i for i in range(N) if kind_of(i) == "function"])):
            want = np.asarray(vals[np.asarray(rows)], dtype=np.float32)
            centroid, spread = self.store.matrix(entity, CENTROID), self.store.matrix(entity, SPREAD)
            self.assertEqual((centroid.dtype, centroid.shape, spread.dtype, spread.shape),
                             (np.float32, (DIM,), np.float32, (DIM,)), entity)
            np.testing.assert_array_equal(centroid, want.mean(axis=0))
            np.testing.assert_array_equal(spread, want.std(axis=0))
            self.assertEqual(self.report["digests"][entity],
                             {CENTROID: self.store.digest_of(entity, CENTROID), SPREAD: self.store.digest_of(entity, SPREAD)})
            self.assertEqual(self.store.db.latest(entity, CENTROID).meta["chunks"], len(rows))

    def test_an_indexed_repository_without_vectors_is_an_entity_without_matrices(self):
        self.assertIsNone(self.store.digest_of("repo:gamma", CENTROID))
        self.assertIsNone(self.store.matrix("repo:gamma", CENTROID))
        self.assertEqual(self.report["digests"]["repo:gamma"], {})
        self.assertEqual(self.store.db.head("repo:gamma"), GENESIS)
        self.assertIsNone(self.store.digest_of("repo:never-heard-of", CENTROID))

    def test_a_chunk_vector_is_one_matrix_with_two_owners(self):
        overlaps = self.store.overlaps()
        self.assertEqual(len(overlaps), SAMPLED_AT_3)
        vals = Matrix(self.kgvm).vals
        for g in self.store.db.history("repo:alpha", CHUNK):
            row = int(g.label) - 1
            stored = self.store.db.load(g)
            self.assertEqual(stored.dtype, np.int8)
            np.testing.assert_array_equal(stored, vals[row])                     # the KGVM row, byte for byte
            kind = kind_entity(kind_of(row))
            self.assertEqual(self.store.digest_of(kind, CHUNK, g.label), g.matrix)
            self.assertEqual(overlaps[g.matrix], [kind, "repo:alpha"])
            self.assertEqual(self.store.owners(g.matrix), [kind, "repo:alpha"])
        self.assertEqual(self.store.overlaps(CENTROID), {})                      # summaries are each entity's own

    def test_each_cell_keeps_at_most_per_cell_vectors(self):
        for repo, count in (("alpha", 9), ("beta", 9)):                          # three kinds, three vectors each
            labels = [g.label for g in self.store.db.history(repo_entity(repo), CHUNK)]
            self.assertEqual(len(labels), count)
            rows = [int(label) - 1 for label in labels]
            self.assertTrue(all((row < ALPHA) == (repo == "alpha") for row in rows))
            self.assertEqual(sorted(kind_of(row) for row in rows), sorted(["class", "function", "preamble"] * 3))

    def test_every_record_has_a_ledger_entry_and_every_chain_verifies(self):
        checks = self.store.verify()
        self.assertEqual(sorted(checks), sorted(WITH_GAMMA + [ROOT]))
        for name, check in checks.items():
            self.assertTrue(check["ok"], name)
            self.assertEqual((check["broken_at"], check["unproven"]), (None, []), name)
        events = self.store.db.events("repo:alpha")
        self.assertEqual([e.event for e in events], [E_RECORD] * 11 + [E_BUILT])  # centroid, spread, nine vectors
        recorded = {g.matrix for kind in (CENTROID, SPREAD, CHUNK) for g in self.store.db.history("repo:alpha", kind)}
        self.assertEqual({e.payload["matrix"] for e in events[:-1]}, recorded)
        self.assertEqual(events[0].payload, {"kind": CENTROID, "label": "", "dtype": "float32", "shape": [DIM],
                                             "matrix": self.store.digest_of("repo:alpha", CENTROID)})
        self.assertEqual(events[-1].payload, {"records": 11, "seed": 0.4123, "chunks": ALPHA, "sampled": 9, "head": "a1"})
        self.assertEqual(self.store.db.events("kind:class")[-1].payload,
                         {"records": 8, "seed": 0.4123, "chunks": 8, "sampled": 6})
        self.assertEqual(self.report["heads"]["repo:alpha"], events[-1].hash)

    def test_an_edited_ledger_no_longer_verifies(self):
        entity = self.store.db.entity_id("repo:alpha")
        self.store.db._db.execute("UPDATE ledger SET payload = replace(payload, 'centroid', 'centroiD')"
                                  " WHERE entity = ? AND seq = 0", (entity,))
        check = self.store.verify()["repo:alpha"]
        self.assertFalse(check["ok"])
        self.assertEqual(check["broken_at"], 0)
        self.assertTrue(self.store.verify()["repo:beta"]["ok"])
        status_now = self.store.status()
        self.assertEqual((status_now["chains_ok"], status_now["broken"]), (False, ["repo:alpha"]))

    def test_a_matrix_recorded_behind_the_stores_back_is_reported(self):
        self.store.db.record("repo:beta", CENTROID, np.ones(DIM, np.float32))     # no ledger entry
        check = self.store.verify()["repo:beta"]
        self.assertEqual(check["broken_at"], None)                               # the chain itself is intact
        self.assertEqual(len(check["unproven"]), 1)
        self.assertFalse(check["ok"])

    def test_rebuilding_an_unchanged_index_writes_nothing_new(self):
        before = self.store.db.storage()
        again = self.store.build_from_index(FakeAtlas(), self.db, self.kgvm, per_cell=3)
        self.assertEqual((again["records"], again["unchanged"]), (0, RECORDS_AT_3))
        self.assertEqual(again["heads"], self.report["heads"])                   # no entity's chain grew
        self.assertEqual(again["digests"], self.report["digests"])
        self.assertEqual(self.store.db.storage(), before)
        self.assertEqual([e.event for e in self.store.db.events(ROOT)], [E_BUILT, E_BUILT])   # the build log did

    def test_the_root_ledger_says_what_the_store_was_built_from(self):
        src = self.store.source()
        self.assertEqual(src["matrix"]["path"], str(self.kgvm.resolve()))
        self.assertEqual(src["chunks_db"], str(self.db.resolve()))
        self.assertEqual((src["matrix"]["dim"], src["per_cell"], src["sampled"], src["overlaps"]),
                         (DIM, 3, SAMPLED_AT_3, SAMPLED_AT_3))
        self.assertIsNone(self.open(self.base / "empty").source())

    def test_status(self):
        st = self.store.status()
        self.assertEqual((st["entities"], st["by_type"], st["overlaps"], st["chains_ok"], st["broken"]),
                         (len(WITH_GAMMA), {"kind": 3, "repo": 3}, SAMPLED_AT_3, True, []))
        self.assertEqual(st["built"]["records"], RECORDS_AT_3)
        self.assertEqual(st["storage"], self.report["bytes"])


class DeterminismTests(HoloCase):
    def test_the_same_index_and_seed_give_the_same_digests_and_ledger_heads(self):
        _, first = self.built("one")
        _, second = self.built("two")
        self.assertEqual(first["digests"], second["digests"])
        self.assertEqual(first["heads"], second["heads"])
        self.assertNotEqual(first["db"], second["db"])

    def test_the_seed_chooses_the_sample_and_nothing_else(self):
        _, first = self.built("one")
        _, other = self.built("two", seed=0.9)
        self.assertEqual(first["digests"], other["digests"])                     # centroids and spreads: every chunk
        self.assertEqual((other["sampled"], other["records"], other["overlaps"]),
                         (SAMPLED_AT_3, RECORDS_AT_3, SAMPLED_AT_3))

    def test_without_a_bound_every_vector_is_recorded(self):
        for per_cell in (None, 0, 100):
            store, report = self.built(f"all-{per_cell}", per_cell=per_cell)
            self.assertEqual((report["sampled"], report["records"], report["overlaps"]), (N, RECORDS_ALL, N), per_cell)
            self.assertEqual({g.label for g in store.db.history("repo:alpha", CHUNK)},
                             {str(i + 1) for i in range(ALPHA)})

    def test_without_an_atlas_the_repositories_come_from_the_chunk_index(self):
        store, report = self.built(atlas=None)
        self.assertEqual(store.entities(), ENTITIES)
        self.assertEqual(report["entities"], {"kind": 3, "repo": 2})
        self.assertEqual(store.db.events("repo:alpha")[-1].payload["head"], None)


class RecordTests(HoloCase):
    def test_unchanged_content_is_not_recorded_again(self):
        store = self.open(self.base / "home")
        self.assertTrue(store.ensure("repo:scratch"))
        self.assertFalse(store.ensure("repo:scratch"))
        first = store.record("repo:scratch", CENTROID, np.arange(4, dtype=np.float32))
        self.assertEqual((first.new, first.seq, first.label), (True, 0, ""))
        same = store.record("repo:scratch", CENTROID, np.arange(4, dtype=np.float32))
        self.assertEqual((same.new, same.seq, same.geometry, same.digest, same.hash),
                         (False, None, first.geometry, first.digest, first.hash))
        changed = store.record("repo:scratch", CENTROID, np.arange(4, dtype=np.float32) + 1)
        self.assertEqual((changed.new, changed.seq), (True, 1))
        self.assertNotEqual(changed.digest, first.digest)
        self.assertEqual(store.digest_of("repo:scratch", CENTROID), changed.digest)
        self.assertEqual(len(store.db.events("repo:scratch", E_RECORD)), 2)
        self.assertEqual(store.db.head("repo:scratch"), changed.hash)
        self.assertTrue(store.verify()["repo:scratch"]["ok"])

    def test_labels_keep_records_apart(self):
        store = self.open(self.base / "home")
        store.ensure("repo:scratch")
        vec = np.arange(4, dtype=np.int8)
        a = store.record("repo:scratch", CHUNK, vec, label="1")
        b = store.record("repo:scratch", CHUNK, vec, label="2")                  # the same vector under another label
        self.assertTrue(a.new and b.new)
        self.assertEqual(a.digest, b.digest)
        self.assertEqual(store.db.storage()["matrices"], 1)
        self.assertEqual(store.owners(a.digest), ["repo:scratch"])
        self.assertEqual(store.overlaps(), {})                                   # one owner is not an overlap


class SoupEvidenceTests(HoloCase):
    def test_active_fragments_become_one_evidence_table_per_kind(self):
        home = self.home()
        soup = Soup(home / "soup.db")
        self.addCleanup(soup.close)
        first = soup.add("the staging box is 10.0.0.7", kind="fact")
        soup.add("the build uses make", kind="fact")
        soup.add("the user may prefer tabs", kind="preference", status="staged")
        soup.credit([first], True, 2.0)
        store = self.open(home)
        report = store.build_from_index(None, *paths(home), soup=soup, per_cell=3)
        self.assertEqual(store.entities("soup"), ["soup:fact"])                  # staged fragments are left out
        self.assertEqual(report["entities"], {"kind": 3, "repo": 2, "soup": 1})
        table = store.matrix("soup:fact", EVIDENCE)
        self.assertEqual(table.dtype, np.float32)
        np.testing.assert_array_equal(table, np.array([[1, 2, 0, 0], [1, 0, 0, 0]], np.float32))
        self.assertEqual(report["digests"]["soup:fact"], {EVIDENCE: store.digest_of("soup:fact", EVIDENCE)})
        self.assertEqual(store.vocabulary(), ENTITIES)                           # Soup entities own no vectors
        self.assertEqual(store.db.events("soup:fact")[-1].payload, {"records": 1, "seed": 0.4123})
        self.assertTrue(store.verify()["soup:fact"]["ok"])


class BuilderTests(HoloCase):
    def test_build_finds_the_homes_own_index_and_publishes_the_report(self):
        home = self.home()
        bus = Bus()
        seen: list = []
        bus.subscribe("holo.*", seen.append)
        report = build(home, per_cell=3, bus=bus)
        self.assertEqual((report["records"], report["overlaps"]), (RECORDS_AT_3, SAMPLED_AT_3))
        self.assertEqual(report["entities"], {"kind": 3, "repo": 2})             # no atlas.db in this home
        self.assertEqual([(env.topic, env.sender) for env in seen], [("holo.built", "holo")])
        self.assertEqual(seen[0].payload, {key: report[key] for key in (
            "entities", "chunks", "orphans", "sampled", "records", "unchanged", "overlaps", "bytes", "per_cell", "seed")})
        st = status(home)
        self.assertEqual((st["entities"], st["chains_ok"], st["memory"]), (len(ENTITIES), True, False))
        self.assertEqual(st["built"]["records"], RECORDS_AT_3)

    def test_status_does_not_create_a_store(self):
        home = self.base / "untouched"
        home.mkdir()
        st = status(home)
        self.assertEqual((st["built"], st["entities"], st["chains_ok"], st["memory"]), (None, 0, True, False))
        self.assertEqual(list(home.iterdir()), [])

    def test_a_home_without_an_index_says_how_to_make_one(self):
        with self.assertRaises(FileNotFoundError) as caught:
            build(self.base / "nothing-here")
        self.assertIn("no chunk index at", str(caught.exception))
        self.assertFalse((self.base / "nothing-here" / "holo.db").exists())

    def test_where_the_index_is_looked_for(self):
        home = self.home()
        other = self.home("other")
        self.assertEqual(locate(home), paths(home))
        with mock.patch.dict(os.environ, {"KGIRL_CHUNKS_DB": str(paths(other)[0]), "KGIRL_MATRIX": str(paths(other)[1])}):
            self.assertEqual(locate(home), paths(other))                         # the environment over the home
            self.assertEqual(locate(home, *paths(home)), paths(home))            # arguments over both


if __name__ == "__main__":
    unittest.main()
