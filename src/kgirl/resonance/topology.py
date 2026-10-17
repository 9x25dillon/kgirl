"""Exact 0-dimensional persistent homology of a point cloud or dissimilarity matrix. Replaces RVSE's
PersistentHomology, whose births and deaths were random draws.

For the Vietoris–Rips (or single-linkage) filtration every point is born at 0, and the H₀ deaths are exactly
the edge weights of a minimum spanning tree: when the filtration reaches an MST edge, two components merge.
Prim's algorithm on a dense matrix gives them in O(n²) time and O(n) extra memory, with no simplicial complex.

    barcode      n − 1 finite bars [0, d_i) plus one infinite bar
    β₀(t)        components alive at scale t = 1 + #{d_i > t}
    entropy      persistence entropy of the finite bars, E = −Σ pᵢ ln pᵢ with pᵢ = dᵢ / Σd
                 (Chintakunta et al. 2015; the original computed −Σ d ln d without normalizing)
    clusters(t)  single-linkage labels at scale t (union–find over MST edges ≤ t)
    gap          largest jump between consecutive deaths: the most persistent cluster split
    sorted_death_distance   L∞ between sorted death vectors: an upper bound on the H₀ bottleneck distance

Higher homology (H₁ loops, H₂ voids) needs the full Rips complex and boundary-matrix reduction. It is not
implemented, and nothing here reports a β₁ or β₂.

Gram input: distances_from_gram turns a (normalized centered) Gram into Euclidean distances
d²ᵢⱼ = Gᵢᵢ + Gⱼⱼ − 2Gᵢⱼ. These inherit the Gram's invariance group, so the barcode of a vessel's probe cloud
is as substrate-independent as its CKA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def distances_from_gram(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G, dtype=np.float64)
    d = np.diag(G)
    D2 = d[:, None] + d[None, :] - 2.0 * G
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(np.maximum(D2, 0.0))


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0))


@dataclass(frozen=True)
class H0:
    deaths: np.ndarray                 # ascending, length n − 1
    edges: tuple                       # MST edges (i, j, w) in the order Prim added them

    @property
    def n(self) -> int:
        return len(self.deaths) + 1

    def betti0(self, t: float) -> int:
        return 1 + int(np.sum(self.deaths > t))

    def total(self) -> float:
        return float(self.deaths.sum())

    def entropy(self) -> float:
        d = self.deaths[self.deaths > 0]
        if d.size == 0:
            return 0.0
        p = d / d.sum()
        return float(-np.sum(p * np.log(p)))

    def normalized_entropy(self) -> float:
        """Entropy / ln(#bars): 1 for equal bars, toward 0 when one bar dominates."""
        k = int(np.sum(self.deaths > 0))
        return float(self.entropy() / np.log(k)) if k > 1 else 0.0

    def gap(self) -> tuple[float, float]:
        """(largest jump between consecutive deaths, the scale just below it)."""
        if len(self.deaths) < 2:
            return 0.0, 0.0
        jumps = np.diff(self.deaths)
        i = int(np.argmax(jumps))
        return float(jumps[i]), float(self.deaths[i])

    def clusters(self, t: float) -> np.ndarray:
        parent = np.arange(self.n)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j, w in self.edges:
            if w <= t:
                parent[find(i)] = find(j)
        roots = np.array([find(a) for a in range(self.n)])
        _, labels = np.unique(roots, return_inverse=True)
        return labels

    def summary(self) -> dict:
        jump, at = self.gap()
        return {"points": self.n, "total": round(self.total(), 6), "entropy": round(self.entropy(), 6),
                "normalized_entropy": round(self.normalized_entropy(), 6),
                "median_death": round(float(np.median(self.deaths)), 6) if len(self.deaths) else 0.0,
                "max_death": round(float(self.deaths[-1]), 6) if len(self.deaths) else 0.0,
                "gap": round(jump, 6), "gap_at": round(at, 6)}


def h0(D: np.ndarray) -> H0:
    """H₀ barcode of a symmetric dissimilarity matrix (Prim's MST, O(n²))."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.ndim != 2 or D.shape[1] != n:
        raise ValueError("need a square dissimilarity matrix")
    if n == 0:
        raise ValueError("empty point set")
    if not np.allclose(D, D.T, atol=1e-9):
        raise ValueError("dissimilarity matrix must be symmetric")
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = D[0].copy()
    parent = np.zeros(n, dtype=int)
    edges = []
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        j = int(np.argmin(cand))
        edges.append((int(parent[j]), j, float(cand[j])))
        in_tree[j] = True
        closer = (D[j] < best) & ~in_tree
        best[closer] = D[j][closer]
        parent[closer] = j
    deaths = np.sort(np.array([w for _, _, w in edges], dtype=np.float64))
    return H0(deaths, tuple(edges))


def sorted_death_distance(a: H0, b: H0) -> float:
    """max |sorted deaths(a) − sorted deaths(b)| for diagrams with the same number of bars.

    Matching bars in sorted order is a valid matching, so this is an *upper bound* on the H₀ bottleneck
    distance. It is tight when every bar is longer than twice the difference, because then no bar is better
    sent to the diagonal. An exact bottleneck needs bipartite matching with diagonal copies and is not
    implemented here."""
    if len(a.deaths) != len(b.deaths):
        raise ValueError("sorted death distance needs diagrams with equal bar counts (same probe set)")
    return float(np.max(np.abs(a.deaths - b.deaths))) if len(a.deaths) else 0.0
