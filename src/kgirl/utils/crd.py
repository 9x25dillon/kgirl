# crd.py — (condensed, production-ready)
from kgirl.utils.crd_engine import CRD  # ← our full engine (was never committed; now crd_engine.py)
import numpy as np

class UnifiedCoherenceRecovery:
    def __init__(self, sfreq=250):
        self.name = "UnifiedCoherenceRecovery"
        self.crd = CRD(
            sfreq=sfreq,
            bands={'delta':(1,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,45)},
            eta={b:1/5 for b in "delta theta alpha beta gamma".split()},
            tau=30.0, theta=0.35, T0=2.0,
            rho={b:0.7 for b in "delta theta alpha beta gamma".split()},
            kappa_base={b:0.3 for b in "delta theta alpha beta gamma".split()},
            noise_std={b:0.01 for b in "delta theta alpha beta gamma".split()},
            window_s=2.0, hop_s=0.25
        )
        self.recovery_count = 0

    async def process(self, kappa_dict: dict, timestamp=None):
        """One step of CRD recovery dynamics on band coherences κ (was a fixed "+0.3" placeholder)."""
        recovered = self.crd.update(kappa_dict)
        recovered.pop("global", None)
        if any(recovered[b] > kappa_dict.get(b, recovered[b]) for b in recovered):
            self.recovery_count += 1
        return recovered

    def process_eeg(self, eeg):
        """Raw EEG (channels × samples at sfreq) → per-window recovered κ, via the CRD engine."""
        return self.crd.process_eeg(eeg)

    def get_statistics(self):
        return {
            'successful_recoveries': self.recovery_count,
            'system': 'CRD',
            **self.crd.statistics(),
        }