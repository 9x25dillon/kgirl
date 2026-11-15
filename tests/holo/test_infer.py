"""Offline tests for holo inference: KGVM search with provenance, guidance, bus events, regulation, the prior.

Queries are vectors, or text with an injected embedder: no model is called. The HoloMemory used for the
prior is a tiny fixture built here with the module's own `build` on the synthetic vectors; nothing is trained.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.holo._fixtures import ALPHA, DIM, ENTITIES, N, ORPHAN, make_index, paths
from kgirl.harness.atlas.vectors import Matrix
from kgirl.harness.hermes.bus import Bus
from kgirl.harness.holo import GateConfig, HoloError, HoloStore, query
from kgirl.harness.holo.infer import GATE_NAME, MEMORY_NAME
from kgirl.harness.holo.store import CENTROID, CHUNK, ROOT
from kgirl.harness.metabolism import Metabolism
from kgirl.tauls.holo import HoloMemory, MemoryConfig, entropy, gate
from kgirl.vessel.matrixdb import encode_matrix

HIT_KEYS = {"id", "score", "repo", "path", "symbol", "kind", "line", "cite", "entities", "digest", "stored",
            "provenance"}


class InferCase(unittest.TestCase):
    per_cell = None                                          # every vector is in the store

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"KGIRL_LIFE": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(self.tmp.name) / "home"
        self.vecs = make_index(self.home)
        self.db, self.kgvm = paths(self.home)
        self.store = HoloStore(self.home)
        self.addCleanup(self.store.close)
        self.store.build_from_index(None, self.db, self.kgvm, per_cell=self.per_cell)

    def listen(self) -> tuple[Bus, list]:
        bus, seen = Bus(), []
        bus.subscribe("holo.*", seen.append)
        return bus, seen


class QueryTests(InferCase):
    def test_a_chunks_own_vector_finds_that_chunk_first(self):
        res = query(self.store, self.vecs[3], k=5)
        self.assertEqual(res["hits"][0]["id"], 4)                # row 3 is chunk 4
        self.assertEqual(len(res["hits"]), 5)
        scores = [hit["score"] for hit in res["hits"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual((res["query"], res["k"], res["regulation"]), (None, 5, None))
        self.assertEqual(res["matrix"], {"count": N + 1, "dim": DIM, "model": "synthetic"})
        json.dumps(res)                                          # the whole result is plain JSON

    def test_a_hit_carries_its_owners_and_their_digests(self):
        top = query(self.store, self.vecs[3], k=5)["hits"][0]
        self.assertEqual(set(top), HIT_KEYS)
        self.assertEqual((top["repo"], top["path"], top["symbol"], top["kind"], top["line"], top["cite"]),
                         ("alpha", "src/m3.py", "sym3", "function", 1, "alpha:src/m3.py:1"))
        self.assertEqual(top["entities"], ["repo:alpha", "kind:function"])
        self.assertEqual(top["provenance"], [
            {"entity": name, "centroid": self.store.digest_of(name, CENTROID), "head": self.store.db.head(name)}
            for name in ("repo:alpha", "kind:function")])
        self.assertEqual(top["digest"], encode_matrix(np.array(Matrix(self.kgvm).vals[3]))[0])
        self.assertEqual(top["digest"], self.store.digest_of("repo:alpha", CHUNK, "4"))
        self.assertTrue(top["stored"])
        self.assertEqual(self.store.owners(top["digest"]), ["kind:function", "repo:alpha"])

    def test_k_bounds_the_hits(self):
        self.assertEqual(query(self.store, self.vecs[0], k=0)["hits"], [])
        self.assertEqual(len(query(self.store, self.vecs[0], k=1)["hits"]), 1)
        self.assertEqual(len(query(self.store, self.vecs[0], k=1000)["hits"]), N + 1)

    def test_a_vector_without_a_chunk_row_is_a_hit_with_no_owner(self):
        hits = query(self.store, self.vecs[0], k=N + 1)["hits"]
        (orphan,) = [hit for hit in hits if hit["id"] == ORPHAN]
        self.assertEqual((orphan["repo"], orphan["cite"], orphan["entities"], orphan["provenance"], orphan["stored"]),
                         (None, None, [], [], False))
        self.assertEqual(orphan["digest"], encode_matrix(np.array(Matrix(self.kgvm).vals[N]))[0])
        self.assertEqual(self.store.owners(orphan["digest"]), [])

    def test_a_query_of_the_wrong_dimension_is_an_error(self):
        with self.assertRaises(ValueError):
            query(self.store, np.ones(8), k=3)

    def test_text_is_embedded_with_the_query_prefix_by_the_given_embedder(self):
        asked: list = []

        def embed(texts):
            asked.append(list(texts))
            return np.asarray([self.vecs[3]], dtype=np.float32)

        res = query(self.store, "where is sym3", k=3, embed=embed)
        self.assertEqual(asked, [["search_query: where is sym3"]])
        self.assertEqual((res["query"], res["hits"][0]["id"]), ("where is sym3", 4))
        with self.assertRaises(HoloError):
            query(self.store, "   ", embed=embed)
        self.assertEqual(len(asked), 1)

    def test_a_store_that_was_never_built_says_so(self):
        empty = HoloStore(Path(self.tmp.name) / "empty")
        self.addCleanup(empty.close)
        with self.assertRaises(HoloError) as caught:
            query(empty, self.vecs[0])
        self.assertIn("has not been built", str(caught.exception))

    def test_a_rebuild_is_picked_up_by_the_next_query(self):
        query(self.store, self.vecs[0], k=1)
        before = self.store.cache["head"]
        self.assertEqual(before, self.store.db.head(ROOT))
        self.store.build_from_index(None, self.db, self.kgvm, per_cell=self.per_cell)
        self.assertEqual(self.store.cache, {})                   # the build dropped what inference had opened
        self.assertEqual(query(self.store, self.vecs[3], k=1)["hits"][0]["id"], 4)
        self.assertNotEqual(self.store.cache["head"], before)

    def test_a_missing_chunk_index_leaves_ids_scores_and_digests(self):
        self.store.drop_cache()
        self.db.unlink()
        top = query(self.store, self.vecs[3], k=2)["hits"][0]
        self.assertEqual((top["id"], top["repo"], top["entities"], top["provenance"]), (4, None, [], []))
        self.assertEqual(top["digest"], self.store.digest_of("repo:alpha", CHUNK, "4"))


class SampledStoreTests(InferCase):
    per_cell = 1                                             # six of the 24 vectors are in the store

    def test_every_hit_is_addressed_and_says_whether_the_store_holds_it(self):
        hits = [hit for hit in query(self.store, self.vecs[3], k=N + 1)["hits"] if hit["id"] != ORPHAN]
        self.assertEqual(len(hits), N)
        vals = Matrix(self.kgvm).vals
        for hit in hits:
            self.assertEqual(hit["digest"], encode_matrix(np.array(vals[hit["id"] - 1]))[0])
            in_store = self.store.digest_of(hit["entities"][0], CHUNK, str(hit["id"]))
            self.assertEqual(hit["stored"], in_store == hit["digest"])
            self.assertEqual(len(hit["provenance"]), 2)          # the owners' centroids are always there
        self.assertEqual(sum(hit["stored"] for hit in hits), 6)


class GuidanceTests(InferCase):
    def rows(self, res: dict) -> dict[str, dict]:
        return {row["entity"]: row for row in res["entities"]}

    def test_shares_follow_the_hits_and_cosines_follow_the_clusters(self):
        res = query(self.store, self.vecs[3], k=5)               # an alpha vector: its neighbours are alpha's
        rows = self.rows(res)
        self.assertEqual(sorted(rows), sorted(ENTITIES))
        self.assertEqual({hit["repo"] for hit in res["hits"]}, {"alpha"})
        self.assertAlmostEqual(rows["repo:alpha"]["p"], 0.5)     # every hit counts once for its repo, once for its kind
        self.assertEqual((rows["repo:alpha"]["hits"], rows["repo:beta"]["hits"], rows["repo:beta"]["p"]), (5, 0, 0.0))
        self.assertAlmostEqual(sum(row["p"] for row in rows.values()), 1.0)
        self.assertGreater(rows["repo:alpha"]["cosine"], rows["repo:beta"]["cosine"])
        for row in rows.values():
            self.assertEqual(row["p"], row["share"])             # no prior: guidance is the retrieval's own share
            self.assertNotIn("p_memory", row)
        self.assertEqual(sum(row["hits"] for name, row in rows.items() if name.startswith("kind:")), 5)

    def test_guidance_is_ordered_by_probability_then_name(self):
        res = query(self.store, self.vecs[ALPHA + 2], k=4)       # a beta vector
        keys = [(-row["p"], row["entity"]) for row in res["entities"]]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(ENTITIES))

    def test_without_hits_the_share_is_uniform(self):
        res = query(self.store, self.vecs[0], k=0)
        self.assertEqual([row["entity"] for row in res["entities"]], ENTITIES)   # ties fall back to the name
        for row in res["entities"]:
            self.assertAlmostEqual(row["p"], 1.0 / len(ENTITIES))
            self.assertEqual(row["hits"], 0)


class BusAndRegulationTests(InferCase):
    def test_the_bus_gets_one_small_event_per_query(self):
        bus, seen = self.listen()
        res = query(self.store, self.vecs[3], k=2, bus=bus)
        self.assertEqual([(env.topic, env.sender) for env in seen], [("holo.query", "holo")])
        payload = seen[0].payload
        self.assertEqual(set(payload), {"q", "k", "ms", "hits", "top", "prior"})
        self.assertEqual((payload["q"], payload["k"], payload["prior"], payload["ms"]), (None, 2, False, res["ms"]))
        self.assertEqual(payload["hits"], [{"id": h["id"], "score": h["score"], "cite": h["cite"],
                                            "entities": h["entities"]} for h in res["hits"]])
        self.assertEqual(payload["top"], res["entities"][0]["entity"])

    def test_text_reaches_the_bus_clipped(self):
        bus, seen = self.listen()
        query(self.store, "x" * 500, k=1, bus=bus, embed=lambda texts: np.asarray([self.vecs[0]], np.float32))
        self.assertEqual(seen[0].payload["q"], "x" * 200)

    def test_regulation_happens_only_when_a_metabolism_is_handed_over(self):
        bus, seen = self.listen()
        self.assertIsNone(query(self.store, self.vecs[3], k=2, bus=bus)["regulation"])
        self.assertEqual([env.topic for env in seen], ["holo.query"])

        organism = Metabolism()                                  # no path: nothing is written anywhere
        reserve = organism.state.F
        res = query(self.store, self.vecs[3], k=2, bus=bus, metabolism=organism)
        row = res["regulation"]
        self.assertEqual((row["kind"], row["work"], row["obs"][0], row["obs"][2:]), ("holo.query", 0.0, 1.0, [0.0, 0.0]))
        self.assertEqual(organism.stats["episodes"], 1)
        self.assertLess(organism.state.F, reserve)               # a step with no energy flowing in costs reserve
        self.assertEqual([env.topic for env in seen], ["holo.query", "holo.query", "holo.regulate"])
        regulate = seen[-1].payload
        self.assertEqual(set(regulate), {"obs", "F", "O", "C", "R", "err"})
        self.assertEqual((regulate["F"], regulate["obs"][0]), (row["F"], 1.0))

    def test_a_query_without_hits_is_observed_as_a_failure(self):
        organism = Metabolism()
        row = query(self.store, self.vecs[3], k=0, metabolism=organism)["regulation"]
        self.assertEqual(row["obs"][0], 0.0)

    def test_life_switched_off_means_no_observation(self):
        organism = Metabolism()
        bus, seen = self.listen()
        with mock.patch.dict(os.environ, {"KGIRL_LIFE": "0"}):
            res = query(self.store, self.vecs[3], k=2, bus=bus, metabolism=organism)
        self.assertIsNone(res["regulation"])
        self.assertEqual(organism.stats["episodes"], 0)
        self.assertEqual([env.topic for env in seen], ["holo.query"])


class PriorTests(InferCase):
    """The optional HoloMemory prior. The memory is a fixture: HoloMemory.build on the synthetic vectors, with
    "next token" = the index of the chunk's repository entity in the store's vocabulary."""

    def memory(self, vocab: int | None = None) -> HoloMemory:
        names = self.store.vocabulary()
        mem = HoloMemory(DIM, vocab or len(names), MemoryConfig(rank=16, branching=2, max_depth=2, min_leaf=4))
        unit = self.vecs / np.linalg.norm(self.vecs, axis=1, keepdims=True)
        nxt = np.array([names.index("repo:alpha" if i < ALPHA else "repo:beta") for i in range(N)])
        return mem.build(mem.project(unit), nxt)

    def test_without_a_memory_file_the_prior_is_off_and_says_why(self):
        res = query(self.store, self.vecs[20], k=5)
        self.assertEqual(res["prior"], {"used": False, "reason": f"no trained memory at {self.home / MEMORY_NAME}"})

    def test_the_gate_mixes_retrieval_and_memory_by_their_entropies(self):
        plain = query(self.store, self.vecs[20], k=5)
        res = query(self.store, self.vecs[20], k=5, memory=self.memory())
        prior = res["prior"]
        self.assertTrue(prior["used"])
        self.assertEqual((prior["prior"], prior["temp"]), (0.1, 1.0))
        self.assertTrue(0.0 < prior["weight"] < 1.0)
        want = gate(np.array([prior["h_retrieval"]]), np.array([prior["h_memory"]]), 0.1, 1.0)[0]
        self.assertAlmostEqual(prior["weight"], float(want), places=12)
        share = np.array([[row["share"] for row in plain["entities"]]])
        self.assertAlmostEqual(prior["h_retrieval"], float(entropy(share)[0]), places=9)
        w = prior["weight"]
        for row in res["entities"]:
            self.assertAlmostEqual(row["p"], (1 - w) * row["share"] + w * row["p_memory"], places=9)
        self.assertAlmostEqual(sum(row["p_memory"] for row in res["entities"]), 1.0, places=5)
        self.assertEqual([hit["id"] for hit in res["hits"]], [hit["id"] for hit in plain["hits"]])   # hits keep KGVM's order

    def test_a_memory_file_and_its_fitted_settings_are_picked_up(self):
        self.memory().save(self.home / MEMORY_NAME)
        (self.home / GATE_NAME).write_text(json.dumps({"prior": 0.3, "temp": 2.0, "beta": 16.0, "nll": 1.23}),
                                           encoding="utf-8")
        prior = query(self.store, self.vecs[20], k=5)["prior"]
        self.assertTrue(prior["used"])
        self.assertEqual((prior["prior"], prior["temp"]), (0.3, 2.0))
        want = gate(np.array([prior["h_retrieval"]]), np.array([prior["h_memory"]]), 0.3, 2.0)[0]
        self.assertAlmostEqual(prior["weight"], float(want), places=12)

    def test_a_memory_of_the_wrong_shape_is_skipped(self):
        wrong = HoloMemory(DIM, len(ENTITIES) + 1, MemoryConfig(rank=16))    # never built: only its shape is read
        res = query(self.store, self.vecs[20], k=5, memory=wrong)
        self.assertFalse(res["prior"]["used"])
        self.assertIn(f"{len(ENTITIES) + 1} tokens", res["prior"]["reason"])
        self.assertTrue(all(row["p"] == row["share"] for row in res["entities"]))

    def test_gate_settings_out_of_range_switch_the_prior_off(self):
        for cfg in (GateConfig(prior=1.0), GateConfig(prior=0.0), GateConfig(temp=0.0)):
            res = query(self.store, self.vecs[20], k=5, memory=self.memory(), gate_cfg=cfg)
            self.assertFalse(res["prior"]["used"], cfg)
            self.assertIn("out of range", res["prior"]["reason"])

    def test_an_unreadable_memory_file_is_skipped(self):
        (self.home / MEMORY_NAME).write_bytes(b"this is not an npz archive")
        res = query(self.store, self.vecs[20], k=5)
        self.assertFalse(res["prior"]["used"])
        self.assertIn("could not be read", res["prior"]["reason"])


class GateConfigTests(unittest.TestCase):
    def test_defaults_when_there_is_no_usable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / GATE_NAME
            self.assertEqual(GateConfig.load(path), GateConfig())
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(GateConfig.load(path), GateConfig())
            path.write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(GateConfig.load(path), GateConfig())

    def test_fitted_settings_replace_the_defaults_they_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / GATE_NAME
            path.write_text(json.dumps({"beta": 256.0, "eta": 0.25, "knn_eta": 0.5, "tau": 0.03, "prior": 0.2,
                                        "temp": 0.5, "nll": 2.5}), encoding="utf-8")
            self.assertEqual(GateConfig.load(path), GateConfig(prior=0.2, temp=0.5, beta=256.0, eta=0.25, knn_eta=0.5,
                                                               tau=0.03))
            path.write_text(json.dumps({"prior": 0.2, "temp": None}), encoding="utf-8")
            self.assertEqual(GateConfig.load(path), GateConfig(prior=0.2))


if __name__ == "__main__":
    unittest.main()
