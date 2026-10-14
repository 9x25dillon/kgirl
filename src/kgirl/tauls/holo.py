"""KGIRL memory: a geometric holographic database over a model's own hidden states.

KGIRL = Kinetic Geometric Inverted Recursive Learning. The memory is the database half; improve.py is the
recursive self-improvement loop that uses it.

    hidden h_t ─chaos projection P─▶ key k_t ∈ S^{r−1} (centred, unit) ─┐  value: next token y_t
                                                                         ▼
    fractal cell tree (recursive deterministic k-means, branching b, depth ≤ L, coarse → fine)
       every cell c stores   n_c, token counts c_c(y)         exact statistics
                             T̂_c = Σ_{t∈c} F(k_t) ⊙ F(v_{y_t})    holographic trace (HRR, Plate 1995):
                                                                  keys bound to token codes v_y, superposed
    query q: descend coarse → fine to the nearest child centroid; along the path
       p_root = unigram,   p_c = (c_c + β p_parent) / (n_c + β)      hierarchical Dirichlet back-off
       leaf vote: r = F⁻¹(conj F(q) ⊙ T̂_leaf), s_y = ⟨r, v_y⟩ ≈ Σ_{t∈leaf, y_t=y} ⟨q, k_t⟩  (+ crosstalk)
       p_mem = (1 − η) p_leaf + η · normalise(max(s, 0)) over the leaf's tokens
    mix with the LM by free energy:  w = σ(logit π + (H_LM − H_mem) / T),  p = (1 − w) p_LM + w p_mem
       (the less uncertain source gets the weight; π, T, η are fitted on held-out tokens)

Coarse-to-fine is gated by information, not by size alone. A cell splits only if its children code
their next tokens in fewer bits than the cell does by more than they cost to describe (MDL / BIC):

    gain = Σ_children Σ_y c_cy · log₂(p_c(y) / p_parent(y))    >    cost = ½ · Σ_children |tokens_c| · log₂ n_parent

Holographic crosstalk grows with the number of traces superposed in a cell (capacity ~ r per cell),
so the fine partition is also what keeps the holographic vote readable.

EFL telemetry, the mass–radius dimension of the hidden-state geometry: with N_ℓ the cells at depth ℓ and
ρ_ℓ their count-weighted RMS radius, D = −d log N_ℓ / d log ρ_ℓ (least squares over levels).

Evidence levels:
  * L0: every number here is computed (the code is the definition).
  * L1: Dirichlet back-off and kNN/cache mixing improve LM perplexity in published work. This is kNN-LM
    (Khandelwal et al. 2020) with a hierarchical index instead of flat nearest neighbours.
  * L2: whether the holographic vote, the fractal dimension, or the self-improvement loop help this model
    is measured, not assumed (`python -m kgirl.tauls memory` / `improve`).

Randomness is chaos (kgirl.tauls.chaos), so a memory built twice from the same states is identical.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import chaos

LOG2 = math.log(2.0)


@dataclass(frozen=True)
class MemoryConfig:
    rank: int = 64              # key / code dimension r
    branching: int = 8          # children per split b
    max_depth: int = 4          # deepest level L (root = 0)
    min_leaf: int = 32          # a split needs ≥ branching · min_leaf states
    beta: float = 4.0           # Dirichlet strength used by the MDL split test at build time
    query_beta: float = 256.0   # Dirichlet back-off strength at query time (fitted by `improve`/`memory`)
    store_keys: bool = True     # keep float16 keys in DFS leaf order for exact kNN (N·r·2 bytes)
    knn_scope_states: int = 4096  # kNN searches the smallest ancestor of the leaf holding at least this many
    kmeans_iters: int = 10
    mdl: bool = True            # information-gated splitting (False: split whenever size allows)
    seed: float = 0.6180

    def to_dict(self) -> dict:
        return asdict(self)


class HoloMemory:
    def __init__(self, dim: int, vocab: int, cfg: MemoryConfig = MemoryConfig()):
        self.dim, self.vocab, self.cfg = dim, vocab, cfg
        r = cfg.rank
        self.proj = (chaos.normal(dim * r, cfg.seed, chaos.stream_id("kgirl", "proj")).reshape(dim, r)
                     / math.sqrt(r)).astype(np.float32)
        self.codes = (chaos.normal(vocab * r, cfg.seed, chaos.stream_id("kgirl", "codes")).reshape(vocab, r)
                      / math.sqrt(r)).astype(np.float32)
        self._codes_f = np.fft.rfft(self.codes, axis=1)
        self.mu = np.zeros(r, np.float32)
        self.built = False

    # ------------------------------------------------------------------ keys

    def project(self, hidden: np.ndarray) -> np.ndarray:
        """Raw projections z = h·P (before centring); stored while collecting so RAM is O(N·r)."""
        return np.asarray(hidden, np.float32).reshape(-1, self.dim) @ self.proj

    def keys(self, z: np.ndarray) -> np.ndarray:
        k = np.asarray(z, np.float32) - self.mu
        return k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-12)

    # ------------------------------------------------------------------ build

    def build(self, z: np.ndarray, nxt: np.ndarray) -> "HoloMemory":
        z = np.asarray(z, np.float32)
        nxt = np.asarray(nxt, np.int64).ravel()
        if z.shape[0] != nxt.shape[0] or z.shape[0] == 0:
            raise ValueError("need one next token per state, and at least one state")
        self.mu = z.mean(axis=0)
        k = self.keys(z)
        c = self.cfg
        V = self.vocab
        self.unigram = ((np.bincount(nxt, minlength=V)[:V] + 0.5) / (len(nxt) + 0.5 * V)).astype(np.float64)
        kf = np.fft.rfft(k, axis=1)
        self.centroid, self.depth, self.parent, self.children = [], [], [], []
        self.count, self.radius, self.tok, self.cnt, self.trace = [], [], [], [], []
        self.splits_rejected = 0
        leaf_idx: dict[int, np.ndarray] = {}
        stack = [(np.arange(len(nxt)), 0, -1, self.unigram)]
        while stack:
            idx, d, par, p_par = stack.pop()
            node = len(self.count)
            toks, cnts = np.unique(nxt[idx], return_counts=True)
            n = len(idx)
            cen = k[idx].mean(axis=0)
            self.centroid.append(cen.astype(np.float32))
            self.depth.append(d)
            self.parent.append(par)
            self.children.append([])
            self.count.append(n)
            self.radius.append(float(np.sqrt(np.mean(np.sum((k[idx] - cen) ** 2, axis=1)))))
            self.tok.append(toks)
            self.cnt.append(cnts.astype(np.float64))
            self.trace.append((kf[idx] * self._codes_f[nxt[idx]]).sum(axis=0))
            if par >= 0:
                self.children[par].append(node)
            p_node = p_par * (c.beta / (n + c.beta))
            p_node[toks] += cnts / (n + c.beta)
            if d >= c.max_depth or n < c.branching * c.min_leaf:
                leaf_idx[node] = idx
                continue
            labels = self._kmeans(k[idx], c.branching)
            parts = [idx[labels == j] for j in range(c.branching)]
            small = [q for q in parts if len(q) < c.min_leaf]
            parts = [q for q in parts if len(q) >= c.min_leaf]
            if small and parts:                          # fold tiny clusters into the nearest kept part
                cen_k = np.stack([k[q].mean(0) for q in parts])
                for q in small:
                    j = int(np.argmin(((cen_k - k[q].mean(0)) ** 2).sum(1)))
                    parts[j] = np.concatenate([parts[j], q])
            if len(parts) < 2:
                leaf_idx[node] = idx
                continue
            if c.mdl and not self._split_pays(parts, nxt, p_node, n):
                self.splits_rejected += 1
                leaf_idx[node] = idx
                continue
            for q in reversed(parts):
                stack.append((q, d + 1, node, p_node))
        self.centroid = np.stack(self.centroid)
        self.count = np.asarray(self.count)
        self.depth = np.asarray(self.depth)
        self.radius = np.asarray(self.radius)
        self.trace = np.stack(self.trace)
        self._layout(leaf_idx, k, nxt)
        self.built = True
        return self

    def _layout(self, leaf_idx: dict, k: np.ndarray, nxt: np.ndarray) -> None:
        """States in DFS leaf order, so every node's descendants are the slice span[node] of key16/ytok."""
        n_nodes = len(self.children)
        self.span = np.zeros((n_nodes, 2), np.int64)
        order, pos = [], 0
        stack = [(0, False)]
        while stack:
            node, done = stack.pop()
            if done:
                self.span[node, 1] = pos
                continue
            self.span[node, 0] = pos
            stack.append((node, True))
            if self.children[node]:
                stack.extend((ch, False) for ch in reversed(self.children[node]))
            else:
                order.append(leaf_idx[node])
                pos += len(leaf_idx[node])
        perm = np.concatenate(order)
        if self.cfg.store_keys:
            self.key16 = k[perm].astype(np.float16)
            self.ytok = nxt[perm].astype(np.int32)
        else:
            self.key16 = np.zeros((0, self.cfg.rank), np.float16)
            self.ytok = np.zeros(0, np.int32)

    def _kmeans(self, x: np.ndarray, b: int) -> np.ndarray:
        """Deterministic k-means: farthest-point init from the point nearest the mean, then Lloyd."""
        cen = [x[np.argmin(np.sum((x - x.mean(0)) ** 2, axis=1))]]
        dmin = np.sum((x - cen[0]) ** 2, axis=1)
        for _ in range(1, b):
            cen.append(x[np.argmax(dmin)])
            dmin = np.minimum(dmin, np.sum((x - cen[-1]) ** 2, axis=1))
        cen = np.stack(cen)
        labels = np.zeros(len(x), np.int64)
        for _ in range(self.cfg.kmeans_iters):
            new = np.argmin((x * x).sum(1)[:, None] - 2 * x @ cen.T + (cen * cen).sum(1)[None, :], axis=1)
            if _ and np.array_equal(new, labels):
                break
            labels = new
            for j in range(b):
                m = labels == j
                if m.any():
                    cen[j] = x[m].mean(0)
        return labels

    def _split_pays(self, parts, nxt, p_node: np.ndarray, n: int) -> bool:
        beta = self.cfg.beta
        gain, params = 0.0, 0
        for q in parts:
            toks, cnts = np.unique(nxt[q], return_counts=True)
            p_child = (cnts + beta * p_node[toks]) / (len(q) + beta)
            gain += float((cnts * (np.log(p_child) - np.log(p_node[toks]))).sum()) / LOG2
            params += len(toks)
        return gain > 0.5 * params * math.log2(max(n, 2))

    # ------------------------------------------------------------------ query

    def route(self, k: np.ndarray) -> np.ndarray:
        """Node index per depth for each key (M, L+1); −1 below the leaf."""
        M = len(k)
        path = np.full((M, self.cfg.max_depth + 1), -1, np.int64)
        path[:, 0] = 0
        cur = np.zeros(M, np.int64)
        for d in range(self.cfg.max_depth):
            nxt = np.full(M, -1, np.int64)
            live = cur >= 0
            for node in np.unique(cur[live]):
                ch = self.children[node]
                if not ch:
                    continue
                rows = np.nonzero(cur == node)[0]
                dist = (k[rows] ** 2).sum(1)[:, None] - 2 * k[rows] @ self.centroid[ch].T
                nxt[rows] = np.asarray(ch)[np.argmin(dist, axis=1)]
            if (nxt < 0).all():
                break
            path[:, d + 1] = nxt
            cur = nxt
        return path

    def leaves(self, path: np.ndarray) -> np.ndarray:
        last = (path >= 0).sum(axis=1) - 1
        return path[np.arange(len(path)), last]

    def components(self, z: np.ndarray, betas=(None,), taus=(0.05,), knn_k: int = 32) -> dict[str, Any]:
        """The mixture's ingredients for raw projections z (each row a distribution over the vocabulary):

            hier[β]  hierarchical Dirichlet back-off along the routed path
            holo     holographic leaf vote normalise(max(⟨F⁻¹(conj F(q) ⊙ T̂_leaf), v_y⟩, 0)); leaf counts where
                     the vote has no positive mass
            knn[τ]   exact kNN over the leaf's parent subtree: top-k cosine neighbours, weights softmax(sim/τ)
        """
        k = self.keys(z)
        path = self.route(k)
        M = len(k)
        leaf = self.leaves(path)
        out: dict[str, Any] = {"leaf": leaf, "hier": {}, "knn": {}}
        for beta in betas:
            beta = self.cfg.query_beta if beta is None else float(beta)
            p = np.tile(self.unigram.astype(np.float32), (M, 1))
            for d in range(path.shape[1]):
                col = path[:, d]
                for node in np.unique(col[col >= 0]):
                    rows = np.nonzero(col == node)[0]
                    n = self.count[node]
                    p[rows] *= beta / (n + beta)
                    p[np.ix_(rows, self.tok[node])] += (self.cnt[node] / (n + beta)).astype(np.float32)
            out["hier"][beta] = p
        holo = np.zeros((M, self.vocab), np.float32)
        qf = np.fft.rfft(k, axis=1)
        for node in np.unique(leaf):
            rows = np.nonzero(leaf == node)[0]
            toks = self.tok[node]
            r = np.fft.irfft(np.conj(qf[rows]) * self.trace[node], n=self.cfg.rank, axis=1)
            s = np.maximum(r @ self.codes[toks].T, 0.0)
            tot = s.sum(axis=1, keepdims=True)
            counts = (self.cnt[node] / self.cnt[node].sum()).astype(np.float32)
            vote = np.where(tot > 0, s / np.maximum(tot, 1e-30), counts[None, :])
            holo[np.ix_(rows, toks)] = vote
        out["holo"] = holo
        if len(self.ytok):
            scope = np.array([self._scope(n) for n in leaf])
            for tau in taus:
                out["knn"][float(tau)] = np.zeros((M, self.vocab), np.float32)
            for node in np.unique(scope):
                rows = np.nonzero(scope == node)[0]
                a, b = self.span[node]
                keys = self.key16[a:b].astype(np.float32)
                sims = k[rows] @ keys.T
                kk = min(knn_k, b - a)
                top = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
                ts = np.take_along_axis(sims, top, axis=1)
                toks = self.ytok[a:b][top]
                for tau in taus:
                    w = np.exp((ts - ts.max(axis=1, keepdims=True)) / tau)
                    w /= w.sum(axis=1, keepdims=True)
                    pk = out["knn"][float(tau)]
                    np.add.at(pk, (np.repeat(rows, kk), toks.ravel()), w.ravel().astype(np.float32))
        return out

    def _scope(self, node: int) -> int:
        """Smallest ancestor (or the leaf itself) whose subtree holds ≥ knn_scope_states states.

        Searching only the leaf's parent made neighbour quality fall as the tree deepened (measured: depth 2
        +1.04 bits/token, depth 3 +0.65); a fixed-size scope decouples kNN from the tree's depth.
        """
        need = getattr(self.cfg, "knn_scope_states", 4096)
        while self.parent[node] >= 0 and self.span[node, 1] - self.span[node, 0] < need:
            node = self.parent[node]
        return node

    @staticmethod
    def mix(comp: dict[str, Any], beta: float | None, eta: float = 0.0, knn_eta: float = 0.0,
            tau: float = 0.05) -> np.ndarray:
        """p_mem = (1 − κ)·[(1 − η)·hier[β] + η·holo] + κ·knn[τ]."""
        hier = comp["hier"][beta] if beta in comp["hier"] else next(iter(comp["hier"].values()))
        p = hier if eta == 0 else (1 - eta) * hier + eta * comp["holo"]
        if knn_eta > 0 and comp["knn"]:
            p = (1 - knn_eta) * p + knn_eta * comp["knn"][float(tau)]
        return p

    def distribution(self, z: np.ndarray, eta: float = 0.0, beta: float | None = None, knn_eta: float = 0.0,
                     tau: float = 0.05, knn_k: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """p_mem (M, V) float32 for raw projections z under one setting, and the leaf of each query."""
        comp = self.components(z, betas=(beta,), taus=(tau,) if knn_eta > 0 else (), knn_k=knn_k)
        b = self.cfg.query_beta if beta is None else float(beta)
        return self.mix(comp, b, eta, knn_eta, tau), comp["leaf"]

    # ------------------------------------------------------------------ telemetry

    def fractal_dimension(self) -> dict[str, Any]:
        """Mass–radius dimension: slope of log N_ℓ against −log ρ_ℓ over levels with ≥ 2 cells."""
        rows = []
        for d in range(int(self.depth.max()) + 1):
            m = self.depth == d
            n = int(m.sum())
            w = self.count[m]
            rho = float(np.sqrt(np.sum(w * self.radius[m] ** 2) / w.sum()))
            rows.append({"level": d, "cells": n, "rms_radius": rho, "mean_cell_states": float(w.mean())})
        pts = [(math.log(r["rms_radius"]), math.log(r["cells"])) for r in rows if r["rms_radius"] > 0]
        out: dict[str, Any] = {"levels": rows, "dimension": None, "r2": None}
        if len(pts) >= 2:
            x, y = np.array(pts).T
            if np.ptp(x) > 0:
                slope, icpt = np.polyfit(x, y, 1)
                pred = slope * x + icpt
                ss = float(((y - y.mean()) ** 2).sum())
                out["dimension"] = float(-slope)
                out["r2"] = float(1 - ((y - pred) ** 2).sum() / ss) if ss > 0 else 1.0
        return out

    def summary(self) -> dict[str, Any]:
        leaves = [i for i, ch in enumerate(self.children) if not ch]
        return {"states": int(self.count[0]), "cells": int(len(self.count)), "leaves": len(leaves),
                "max_depth_reached": int(self.depth.max()), "splits_rejected_by_mdl": int(self.splits_rejected),
                "mean_leaf_states": float(np.mean(self.count[leaves])), "config": self.cfg.to_dict(),
                "fractal": self.fractal_dimension()}

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        offs = np.cumsum([0] + [len(t) for t in self.tok])
        np.savez_compressed(path, centroid=self.centroid, depth=self.depth, parent=np.asarray(self.parent),
                            count=self.count, radius=self.radius, trace=self.trace, mu=self.mu,
                            unigram=self.unigram, tok=np.concatenate(self.tok), cnt=np.concatenate(self.cnt),
                            span=self.span, key16=self.key16, ytok=self.ytok,
                            offs=offs, meta=np.frombuffer(json.dumps(
                                {"dim": self.dim, "vocab": self.vocab, "cfg": self.cfg.to_dict(),
                                 "splits_rejected": self.splits_rejected}).encode(), np.uint8))

    @classmethod
    def load(cls, path: str | Path) -> "HoloMemory":
        a = np.load(path)
        meta = json.loads(bytes(a["meta"]).decode())
        m = cls(meta["dim"], meta["vocab"], MemoryConfig(**meta["cfg"]))
        m.centroid, m.depth, m.count, m.radius = a["centroid"], a["depth"], a["count"], a["radius"]
        m.trace, m.mu, m.unigram = a["trace"], a["mu"], a["unigram"]
        m.parent = a["parent"].tolist()
        m.span, m.key16, m.ytok = a["span"], a["key16"], a["ytok"]
        offs = a["offs"]
        m.tok = [a["tok"][offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        m.cnt = [a["cnt"][offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
        m.children = [[] for _ in m.parent]
        for i, p in enumerate(m.parent):
            if p >= 0:
                m.children[p].append(i)
        m.splits_rejected = meta["splits_rejected"]
        m.built = True
        return m


# ---------------------------------------------------------------------- free-energy gate

def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-30, None)
    return -(p * np.log(q)).sum(axis=1)


def gate(h_lm: np.ndarray, h_mem: np.ndarray, prior: float, temp: float) -> np.ndarray:
    """w_mem = σ(logit π + (H_LM − H_mem) / T); T = inf gives the constant weight π."""
    z = np.full(np.shape(h_lm), math.log(prior / (1 - prior)))
    if not math.isinf(temp):
        z = z + (h_lm - h_mem) / temp
    return 1.0 / (1.0 + np.exp(-z))


PRIORS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
TEMPS = (0.25, 0.5, 1.0, 2.0, math.inf)
BETAS = (16.0, 256.0, 4096.0)
ETAS = (0.0, 0.25)
KNN = ((0.0, 0.05), (0.5, 0.03), (0.5, 0.1), (0.8, 0.03), (0.8, 0.1))   # (κ, τ)
SETTINGS = tuple((b, e, kn, t) for b in BETAS for e in ETAS for kn, t in KNN)


def mixed_nll(p_lm_t: np.ndarray, p_mem_t: np.ndarray, w: np.ndarray) -> np.ndarray:
    return -np.log(np.clip((1 - w) * p_lm_t + w * p_mem_t, 1e-30, None))


def fit_gate(stats: dict[tuple, dict[str, np.ndarray]], fit: np.ndarray) -> dict[str, float]:
    """Grid search of the memory setting (β, η, κ, τ) and the gate (π, T) minimising mean NLL on `fit` rows.

    stats[setting] holds per-token arrays: p_lm_t, p_mem_t (target probabilities), h_lm, h_mem (nats)."""
    best = None
    for (beta, eta, knn_eta, tau), s in stats.items():
        for prior in PRIORS:
            for temp in TEMPS:
                w = gate(s["h_lm"][fit], s["h_mem"][fit], prior, temp)
                nll = float(mixed_nll(s["p_lm_t"][fit], s["p_mem_t"][fit], w).mean())
                if best is None or nll < best["nll"]:
                    best = {"beta": beta, "eta": eta, "knn_eta": knn_eta, "tau": tau,
                            "prior": prior, "temp": temp, "nll": nll}
    return best
