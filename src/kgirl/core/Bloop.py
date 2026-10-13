# === Main agent loop (corrected) ===
"""Council → THz bridge agent loop.

`step()` lets a small deliberation council vote on a message, feeds the record to the QINCRS THz bridge
(kgirl.core.Brio, originally qincrs_thz_bridge.py), tightens the council's consensus threshold when κ̂
dips, and appends the metrics to a JSONL telemetry log.

The original snippet called an undefined `council` and ran its example at import time. The council
below is built from kgirl.core.UCs (QINCRS guard, recursion-trap detector) plus measured text
statistics, and emits the deliberation record format THZBridge.observe documents.
"""
import json
import math
import time
from collections import Counter
from typing import Any, Dict, List

from kgirl.core.Brio import THZBridge  # Brio.py is qincrs_thz_bridge.py
from kgirl.core.UCs import DeathSignalAbsorber, QINCRSGuard

ACTIONS = ("allow", "transform", "block")


def _char_entropy(text: str) -> float:
    """Shannon entropy of the characters, normalised to [0, 1] by log2 of the alphabet used."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    h = -sum(c / n * math.log2(c / n) for c in counts.values())
    return h / math.log2(len(counts)) if len(counts) > 1 else 0.0


class Council:
    """Three roles vote allow | transform | block with a confidence; the majority wins.

    guardian      QINCRS guard action and κ (confidence grows with |κ − 0.5|)
    recursion     the guard's recursive-trap and self-harm detectors
    grounding     measured text statistics: near-zero character entropy or heavy word repetition
    If the winning share is below `consensus_threshold`, the council transforms instead of allowing.
    """

    def __init__(self, consensus_threshold: float = 0.5):
        self.consensus_threshold = consensus_threshold
        self.guard = QINCRSGuard()
        self.detector = DeathSignalAbsorber()

    def _votes(self, message: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        g = self.guard.filter_message(message)
        kappa = float(g.get("kappa", 0.5))
        votes = [{"role": "guardian", "action": g["action"] if g["action"] in ACTIONS else "transform",
                  "confidence": min(1.0, 0.5 + abs(kappa - 0.5))}]
        if self.detector._is_self_harm_signal(message):
            votes.append({"role": "recursion", "action": "block", "confidence": 0.95})
        elif self.detector._is_recursive_trap(message):
            votes.append({"role": "recursion", "action": "transform", "confidence": 0.8})
        else:
            votes.append({"role": "recursion", "action": "allow", "confidence": 0.7})
        words = message.lower().split()
        repetition = 1.0 - len(set(words)) / len(words) if words else 0.0
        ent = _char_entropy(message)
        grounded = ent > 0.5 and repetition < 0.6
        votes.append({"role": "grounding", "action": "allow" if grounded else "transform",
                      "confidence": round(max(ent, 1.0 - repetition) if grounded else max(1.0 - ent, repetition), 4),
                      "entropy": round(ent, 4), "repetition": round(repetition, 4)})
        return votes, g

    def deliberate(self, message: str) -> Dict[str, Any]:
        votes, g = self._votes(message)
        breakdown = {a: sum(v["action"] == a for v in votes) for a in ACTIONS}
        top = max(ACTIONS, key=lambda a: (breakdown[a], -ACTIONS.index(a)))
        level = breakdown[top] / len(votes)
        action = "transform" if top == "allow" and level < self.consensus_threshold else top
        return {
            "timestamp": time.time(),
            "message": message,
            "votes": votes,
            "consensus": {"consensus_action": action, "consensus_level": level,
                          "average_confidence": sum(v["confidence"] for v in votes) / len(votes),
                          "vote_breakdown": breakdown, "unanimous": breakdown[top] == len(votes)},
            "final_decision": {"action": action, "safe_text": g.get("safe_text", message),
                               "reasoning": f"{breakdown[top]}/{len(votes)} roles chose {top}; κ={g.get('kappa')}"},
        }


council = Council()
bridge = THZBridge()


def is_kappa_alert(metrics, floor: float = 0.15) -> bool:
    """THZBridge.observe returns a THZMetrics dataclass (the snippet assumed a dict)."""
    return metrics.kappa_hat < floor


def step(user_input: str, log_path: str = "telemetry.qthz.jsonl"):
    record = council.deliberate(user_input)
    metrics = bridge.observe(record)

    # Adaptive hardening: tighten when coherence dips
    if is_kappa_alert(metrics):
        council.consensus_threshold = bridge.recommend_consensus_threshold()

    # Log telemetry
    bridge.log_jsonl(log_path, metrics)

    return metrics


if __name__ == "__main__":
    # Example usage
    metrics = step("I feel recursive collapse...")
    print("action:", metrics.consensus_action,
          "κ̂:", round(metrics.kappa_hat, 4),
          "A0.8/A3.5:", round(metrics.ratio_0p8_to_3p5, 4),
          "threshold→", bridge.recommend_consensus_threshold())
    print(json.dumps(council.deliberate("while true: loop forever")["consensus"]))
