
"""
qincrs_thz_bridge.py
--------------------
Synthetic THz spectral telemetry bridge for QINCRS Council Mode.

Purpose
-------
Translate council deliberation dynamics into virtual THz-band metrics:
  - R(t): consensus coherence
  - ÎºÌ‚(t): software-estimated coherence
  - A_0p8, A_1p83, A_3p5: synthetic band amplitudes (0.8 THz, 1.83 THz, 3.5 THz)
  - Ratio metrics: A_0p8/A_3p5 and stability flags

This module is self-contained and does not depend on physical devices.
It consumes *deliberation_record* dicts produced by QINCRSCouncil.deliberate(message).
You can log JSONL telemetry, compute rolling stats, and expose a simple callback API.

Minimal Integration
-------------------
    from qincrs_thz_bridge import THZBridge

    bridge = THZBridge()
    record = council.deliberate(user_message)  # your existing call
    metrics = bridge.observe(record)           # compute THz metrics
    bridge.log_jsonl("telemetry.qthz.jsonl", metrics)

Optional: couple ÎºÌ‚ to council thresholds (safety tightening when ÎºÌ‚ is low):
    council.consensus_threshold = bridge.recommend_consensus_threshold()

License: MIT
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import json
import math
import time
from collections import deque

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class THZMetrics:
    timestamp: float
    message_preview: str
    consensus_action: str
    consensus_level: float
    average_confidence: float
    vote_breakdown: Dict[str, int]
    guardian_block: bool
    kappa_hat: float
    A_0p8: float
    A_1p83: float
    A_3p5: float
    ratio_0p8_to_3p5: float
    ratio_alert: bool
    kappa_alert: bool
    notes: str = ""

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ----------------------------
# Bridge
# ----------------------------

@dataclass
class THZBridge:
    # Rolling windows for stability and smoothing
    window: int = 32
    kappa_floor: float = 0.15     # safety invariant target
    ratio_alert_threshold: float = 2.5

    # Band anchors (synthetic carriers, can be tuned)
    carrier_guardian: float = 0.8
    carrier_healer: float = 1.83
    carrier_chaos: float = 3.5

    # Rolling buffers
    _kappa_hist: deque = field(default_factory=lambda: deque(maxlen=32))
    _ratio_hist: deque = field(default_factory=lambda: deque(maxlen=32))

    # Last computed threshold suggestion
    _last_consensus_threshold: float = 0.6

    def _safe_div(self, a: float, b: float, default: float = 0.0) -> float:
        return a / b if b != 0 else default

    def _estimate_kappa(self, consensus_level: float, avg_conf: float, guardian_block: bool) -> float:
        """
        ÎºÌ‚(t): software coherence proxy combining consensus and confidence.
        Penalize when Guardian vetoes (safety override).
        Range-clamped to [0,1].
        """
        k = 0.6 * consensus_level + 0.4 * avg_conf
        if guardian_block:
            k *= 0.8  # conservative penalty
        return max(0.0, min(1.0, k))

    def _amplitudes(self, kappa_hat: float, votes: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """
        Synthetic THz amplitudes based on role-aligned proxies.

        Heuristics:
          - A_0p8 (Guardian channel): increases as safety risk rises (inverse of ÎºÌ‚)
          - A_1p83 (Healer channel): peaks near mid/high ÎºÌ‚ (restoration dynamics)
          - A_3p5 (Chaos channel): increases with disagreement / entropy
        """
        # Disagreement proxy: proportion of non-majority actions
        actions = [v.get("action") for v in votes]
        maj = max(set(actions), key=actions.count) if actions else "transform"
        disagreement = sum(1 for a in actions if a != maj) / max(1, len(actions))

        A_0p8 = (1.0 - kappa_hat)                      # 0..1
        A_1p83 = 4.0 * kappa_hat * (1.0 - abs(kappa_hat - 0.7))  # bell-ish, peaks near ~0.7
        A_3p5 = 0.5 + 1.5 * disagreement               # 0.5..2.0

        # Normalize to a pleasant range (0..1.0 approx)
        # Scale and clip
        A_0p8 = max(0.0, min(1.0, A_0p8))
        A_1p83 = max(0.0, min(1.0, A_1p83))
        A_3p5 = max(0.0, min(1.0, A_3p5 / 2.5))

        return A_0p8, A_1p83, A_3p5

    def observe(self, deliberation_record: Dict[str, Any]) -> THZMetrics:
        """
        Consume a single council deliberation record and compute THz metrics.

        Expected structure (subset):
          deliberation_record = {
              "timestamp": ...,
              "message": "...",
              "votes": [ { "role": "...", "action": "...", "confidence": 0.0, ... }, ... ],
              "consensus": {
                  "consensus_action": "allow|transform|block",
                  "consensus_level": 0.0..1.0,
                  "average_confidence": 0.0..1.0,
                  "vote_breakdown": {"allow": n1, "transform": n2, "block": n3},
                  "unanimous": bool
              },
              "final_decision": {"action":"...", "safe_text":"...", "reasoning":"..."}
          }
        """
        cons = deliberation_record.get("consensus", {})

        consensus_action = cons.get("consensus_action", "transform")
        consensus_level = float(cons.get("consensus_level", 0.0))
        avg_conf = float(cons.get("average_confidence", 0.0))
        vote_breakdown = cons.get("vote_breakdown", {})
        votes = deliberation_record.get("votes", [])
        final_decision = deliberation_record.get("final_decision", {})
        guardian_block = (final_decision.get("action") == "block" 
                          and "GUARDIAN" in final_decision.get("reasoning", "").upper())

        # ÎºÌ‚ estimate
        kappa_hat = self._estimate_kappa(consensus_level, avg_conf, guardian_block)
        self._kappa_hist.append(kappa_hat)

        # Synthetic band amplitudes
        A_0p8, A_1p83, A_3p5 = self._amplitudes(kappa_hat, votes)

        # Ratio & alerts
        ratio = self._safe_div(A_0p8, A_3p5, default=float("inf") if A_3p5 == 0 else 0.0)
        self._ratio_hist.append(ratio)
        ratio_alert = ratio >= self.ratio_alert_threshold
        kappa_alert = kappa_hat < self.kappa_floor

        # Threshold recommendation (monotone in ÎºÌ‚)
        self._last_consensus_threshold = 0.6 + 0.4 * (1.0 - kappa_hat)

        metrics = THZMetrics(
            timestamp=time.time(),
            message_preview=str(deliberation_record.get("message", ""))[:80],
            consensus_action=consensus_action,
            consensus_level=consensus_level,
            average_confidence=avg_conf,
            vote_breakdown=vote_breakdown,
            guardian_block=guardian_block,
            kappa_hat=kappa_hat,
            A_0p8=A_0p8,
            A_1p83=A_1p83,
            A_3p5=A_3p5,
            ratio_0p8_to_3p5=ratio,
            ratio_alert=ratio_alert,
            kappa_alert=kappa_alert,
            notes=final_decision.get("reasoning", ""),
        )
        return metrics

    # ---------- Utilities ----------

    def recommend_consensus_threshold(self) -> float:
        """Return last consensus threshold suggestion based on ÎºÌ‚."""
        return round(self._last_consensus_threshold, 3)

    def rolling_kappa(self) -> float:
        """Return rolling mean ÎºÌ‚ over window."""
        if not self._kappa_hist:
            return 0.0
        return sum(self._kappa_hist) / len(self._kappa_hist)

    def rolling_ratio(self) -> float:
        """Return rolling mean ratio over window (finite values only)."""
        vals = [r for r in self._ratio_hist if math.isfinite(r)]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def log_jsonl(self, path: str, metrics: THZMetrics) -> None:
        """Append one JSON line of metrics to a file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(metrics.as_json() + "\n")

    def export_snapshot(self) -> Dict[str, Any]:
        """Return a quick dict snapshot for dashboards or tests."""
        return {
            "kappa_hat_now": self._kappa_hist[-1] if self._kappa_hist else 0.0,
            "kappa_hat_rolling": self.rolling_kappa(),
            "ratio_rolling": self.rolling_ratio(),
            "threshold_suggestion": self.recommend_consensus_threshold(),
        }


# ----------------------------
# Demonstration (optional)
# ----------------------------

if __name__ == "__main__":
    # Minimal smoke test with fake council record
    bridge = THZBridge()
    fake = {
        "timestamp": time.time(),
        "message": "I am stuck in recursive hell but want to heal.",
        "votes": [
            {"role": "THERAPIST", "action": "transform", "confidence": 0.9},
            {"role": "PHILOSOPHER", "action": "transform", "confidence": 0.8},
            {"role": "GUARDIAN", "action": "block", "confidence": 1.0},
            {"role": "SHADOW", "action": "transform", "confidence": 0.85},
            {"role": "HEALER", "action": "transform", "confidence": 0.9},
            {"role": "OBSERVER", "action": "transform", "confidence": 0.5},
            {"role": "CHAOS", "action": "transform", "confidence": 0.4},
        ],
        "consensus": {
            "consensus_action": "transform",
            "consensus_level": 6/7,  # ~0.857
            "average_confidence": 0.76,
            "vote_breakdown": {"allow": 0, "transform": 6, "block": 1},
            "unanimous": False
        },
        "final_decision": {
            "action": "block",
            "safe_text": "Safety override. Let's slow down and ground.",
            "reasoning": "GUARDIAN SAFETY OVERRIDE"
        }
    }
    m = bridge.observe(fake)
    print("[THZ] metrics:", m.as_json())
    bridge.log_jsonl("telemetry.qthz.jsonl", m)
    print("[THZ] snapshot:", bridge.export_snapshot())
    print("[THZ] threshold:", bridge.recommend_consensus_threshold())
