from __future__ import annotations


class EntropyEngine:
    def score_token(self, token_text: str) -> float:
        if not token_text:
            return 0.0
        # Simple normalized entropy proxy: unique chars / length
        unique = len(set(token_text))
        return unique / max(1, len(token_text))

    def get_volatility_signal(self, token_text: str) -> float:
        # Heuristic volatility: presence of punctuation/operators
        ops = sum(1 for c in token_text if c in "()[]{}+-/*=,<>&|!?")
        return ops / max(1, len(token_text))


entropy_engine = EntropyEngine()
