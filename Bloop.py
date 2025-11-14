# === Main agent loop (corrected) ===
from qincrs_thz_bridge import THZBridge
import time, json

bridge = THZBridge()

def is_kappa_alert(metrics: dict, floor: float = 0.15) -> bool:
    return metrics.get('kappa_hat', 0.0) < floor

def step(user_input: str):
    record = council.deliberate(user_input)
    metrics = bridge.observe(record)

    # Adaptive hardening: tighten when coherence dips
    if is_kappa_alert(metrics):
        council.consensus_threshold = bridge.recommend_consensus_threshold()

    # Log telemetry
    bridge.log_jsonl("telemetry.qthz.jsonl", metrics)

    return metrics

# Example usage
metrics = step("I feel recursive collapse...")
print("R:", metrics['phase_coherence_R'],
      "κ̂:", metrics['kappa_hat'],
      "A0.8/A3.5:", metrics['amplitude_ratio_A08_A35'],
      "threshold→", bridge.recommend_consensus_threshold())
