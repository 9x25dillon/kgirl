import asyncio
import math
from typing import Dict, Any

try:
    from tauls_model import TAULSEvaluator, TAULSRunner
    HAS_TAULS = True
except Exception:
    HAS_TAULS = False


async def evaluate_text(text: str) -> Dict[str, Any]:
    """Compute evaluation metrics for a candidate text.

    Returns a dict like: { 'entropy': float, 'stability': float, 'length': int }
    """
    if HAS_TAULS:
        try:
            # prefer model-based runner when available
            runner = TAULSRunner()
            return await asyncio.get_event_loop().run_in_executor(None, runner.score_text, text)
        except Exception:
            try:
                ev = TAULSEvaluator()
                return await asyncio.get_event_loop().run_in_executor(None, ev.score, text)
            except Exception:
                pass

    # fallback: simple character-level entropy estimator and stability proxy
    cnt = {}
    for ch in text:
        cnt[ch] = cnt.get(ch, 0) + 1
    total = len(text) or 1
    entropy = -sum((v/total) * math.log((v/total)+1e-12, 2) for v in cnt.values())
    # normalize entropy to [0,1] by dividing by log2(unique_chars)
    unique = max(1, len(cnt))
    max_ent = math.log(unique, 2) if unique > 1 else 1.0
    entropy_norm = entropy / max_ent if max_ent > 0 else 0.0

    # stability proxy: inverse of punctuation density (simple heuristic)
    punct = sum(1 for c in text if c in '.!?')
    stability = 1.0 - min(1.0, punct / (total / 10 + 1e-12))

    return {'entropy': entropy_norm, 'stability': stability, 'length': total}
