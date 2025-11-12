# qincrs_thz_bridge.py (hardened)

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json, math, time
import numpy as np

class THZBridge:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.COUNCIL_WEIGHTS = {
            'Guardian': 2.0,
            'Healer': 1.3,
            'Therapist': 1.5,
            'Chaos': 0.7,
            'Philosopher': 1.0,
            'Observer': 1.0,
            'Shadow': 1.2,
        }
        self.CARRIER_FREQS = {  # Hz
            'Guardian':   0.80e12,
            'Healer':     1.83e12,
            'Therapist':  1.20e12,
            'Chaos':      3.50e12,
            'Philosopher':1.10e12,
            'Observer':   0.95e12,
            'Shadow':     1.40e12,
        }
        self.base_linewidth = 150e9    # 150 GHz
        self.ratio_alert_threshold = 2.5

    # ---------- public API ----------

    def observe(self, council_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps a deliberation record (either minimal or full) to THz metrics.
        Minimal format:
            {'roles': [...], 'votes': [...], 'stress': float, 'transmutation_triggered': bool}
        Full council format from QINCRSCouncil.deliberate(message):
            {'votes': [{'role': 'GUARDIAN', 'action': 'block', 'confidence': 1.0, ...}, ...],
             'consensus': {...}, 'timestamp': ..., 'message': ...}
        """
        roles, votes, stress, tflag, ts = self._extract_minimal_fields(council_record)

        # Safety: align roles/votes, clamp to [0,1]
        roles, votes = self._sanitize_roles_votes(roles, votes)
        if not roles:
            # empty input → neutral metrics
            m = self._neutral_metrics(ts)
            self.history.append(m)
            return m

        # Phase seed: incorporate carriers so each role contributes a distinct drift
        phases = self._phases_from_votes_and_carriers(roles, votes)

        # Kuramoto order parameter R (coherence 0..1)
        R = self._kuramoto_order_parameter(phases)

        # Amplitudes (Guardian ~ “risk”, Healer ~ mid/high κ̂, Chaos ~ disagreement)
        kappa_hat = self._kappa_hat_from_votes_R(votes, R, tflag)
        A08, A183, A35 = self._synthetic_amplitudes(roles, votes, kappa_hat)

        # A(0.8)/A(3.5) with safe divide
        ratio = A08 / (A35 if A35 > 1e-12 else 1e-12)
        ratio_alert = bool(ratio >= self.ratio_alert_threshold)

        # Linewidth broadening from stress (clamped)
        Gamma = self.base_linewidth * (1.0 + 0.3 * min(max(stress, 0.0) / 4.0, 1.0))

        # Proper entropy of normalized activations
        council_entropy = self._entropy_from_votes(votes)

        metrics = {
            'time_unix': ts,
            'phase_coherence_R': float(R),
            'amplitude_ratio_A08_A35': float(ratio),
            'predicted_linewidth_Hz': float(Gamma),
            'transmutation_active': bool(tflag),
            'stress_level': float(stress),
            'A_0p8': float(A08),
            'A_1p83': float(A183),
            'A_3p5': float(A35),
            'kappa_hat': float(kappa_hat),
            'council_entropy': float(council_entropy),
        }
        self.history.append(metrics)
        return metrics

    def recommend_consensus_threshold(self) -> float:
        """Adaptive threshold: higher R → stricter consensus."""
        if not self.history:
            return 0.75
        recent = self.history[-5:]
        avg_R = sum(m['phase_coherence_R'] for m in recent) / max(len(recent), 1)
        return float(np.clip(0.70 + 0.25 * avg_R, 0.60, 0.90))

    def log_jsonl(self, filepath: str, metrics: Dict[str, Any]) -> None:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + '\n')

    # ---------- adapters & helpers ----------

    def _extract_minimal_fields(self, rec: Dict[str, Any]) -> Tuple[List[str], List[float], float, bool, float]:
        """Support both minimal and full council records."""
        ts = float(rec.get('timestamp', time.time()))
        # Full council record?
        if 'votes' in rec and isinstance(rec['votes'], list) and rec['votes'] and isinstance(rec['votes'][0], dict):
            # roles mapped from NodeRole enum names or strings
            roles = []
            votes = []
            for v in rec['votes']:
                role = str(v.get('role', '')).title()  # e.g., GUARDIAN -> Guardian
                conf = float(v.get('confidence', 0.0))
                roles.append(role)
                votes.append(conf)
            stress = float(rec.get('consensus', {}).get('entropy', 0.0)) if isinstance(rec.get('consensus'), dict) else 0.0
            tflag = 'OVERRIDE' in str(rec.get('final_decision', {}).get('reasoning', '')).upper()
            return roles, votes, stress, tflag, ts

        # Minimal format
        roles = list(rec.get('roles', []))
        votes = [float(x) for x in rec.get('votes', [])]
        stress = float(rec.get('stress', 0.0))
        tflag = bool(rec.get('transmutation_triggered', False))
        return roles, votes, stress, tflag, ts

    def _sanitize_roles_votes(self, roles: List[str], votes: List[float]) -> Tuple[List[str], List[float]]:
        n = min(len(roles), len(votes))
        roles = [str(r).title() for r in roles[:n]]
        votes = [float(max(0.0, min(1.0, v))) for v in votes[:n]]
        return roles, votes

    def _phases_from_votes_and_carriers(self, roles: List[str], votes: List[float]) -> np.ndarray:
        # Normalize carriers to [0,1] to seed distinct angular velocities
        carriers = [self.CARRIER_FREQS.get(r, 1.0) for r in roles]
        c_min, c_max = (min(carriers), max(carriers)) if carriers else (1.0, 1.0)
        if c_max == c_min:
            scaled = [0.5] * len(carriers)
        else:
            scaled = [(c - c_min) / (c_max - c_min) for c in carriers]
        # Phase = 2π*( vote_weight * 0.7 + carrier_scaled * 0.3 )
        phases = [2*math.pi*(0.7*v + 0.3*s) for v, s in zip(votes, scaled)]
        return np.array(phases, dtype=float)

    def _kuramoto_order_parameter(self, phases: np.ndarray) -> float:
        if phases.size == 0:
            return 0.0
        z = np.exp(1j * phases)
        R = np.abs(np.mean(z))
        return float(max(0.0, min(1.0, R)))

    def _kappa_hat_from_votes_R(self, votes: List[float], R: float, guardian_block: bool) -> float:
        # votes mean ~ “confidence”; combine with R
        v_mean = sum(votes) / max(len(votes), 1)
        k = 0.6 * R + 0.4 * v_mean
        if guardian_block:
            k *= 0.8
        return float(max(0.0, min(1.0, k)))

    def _synthetic_amplitudes(self, roles: List[str], votes: List[float], kappa_hat: float) -> Tuple[float, float, float]:
        # Disagreement proxy
        # Here we approximate: dispersion of votes ~ disagreement
        v_mean = sum(votes) / max(len(votes), 1)
        disp = math.sqrt(sum((v - v_mean)**2 for v in votes) / max(len(votes), 1))

        # Guardian / Chaos activations (weighted)
        g = self._role_activation('Guardian', roles, votes)
        c = self._role_activation('Chaos', roles, votes)

        # A_0.8 rises when κ low and Guardian activation high
        A08 = max(0.0, min(1.0, (1.0 - kappa_hat)*0.6 + g*0.6))
        # A_1.83 peaks mid/high κ (simple bell shape)
        A183 = max(0.0, min(1.0, 4.0 * kappa_hat * (1.0 - abs(kappa_hat - 0.7))))
        # A_3.5 rises with disagreement and Chaos activation
        A35 = max(0.0, min(1.0, 0.4 + 0.8*disp + 0.6*c))
        return A08, A183, A35

    def _role_activation(self, role: str, roles: List[str], votes: List[float]) -> float:
        if role not in roles:
            return 0.0
        idx = roles.index(role)
        w = self.COUNCIL_WEIGHTS.get(role, 1.0)
        return float(max(0.0, min(1.0, w * votes[idx])))

    def _entropy_from_votes(self, votes: List[float]) -> float:
        if not votes:
            return 0.0
        v = np.array(votes, dtype=float)
        s = v.sum()
        if s <= 0:
            return 0.0
        p = np.clip(v / s, 1e-9, 1.0)
        p /= p.sum()
        return float(-np.sum(p * np.log(p)))

    def _neutral_metrics(self, ts: float) -> Dict[str, Any]:
        return {
            'time_unix': ts,
            'phase_coherence_R': 0.0,
            'amplitude_ratio_A08_A35': 1.0,
            'predicted_linewidth_Hz': float(self.base_linewidth),
            'transmutation_active': False,
            'stress_level': 0.0,
            'A_0p8': 0.0,
            'A_1p83': 0.0,
            'A_3p5': 0.0,
            'kappa_hat': 0.0,
            'council_entropy': 0.0,
        }
