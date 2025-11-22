import asyncio
import json
from typing import Dict, Any

from llm_adapters import LLMAdapter, LocalStubAdapter, RemoteHTTPAdapter
from llm_eval import evaluate_text


class DualLLMOrchestrator:
    """Orchestrates a primary LLM and a critic LLM (secondary) to produce and evaluate outputs."""
    def __init__(self, primary: LLMAdapter = None, critic: LLMAdapter = None):
        from llm_adapters import create_adapter_from_env
        self.primary = primary or create_adapter_from_env()
        self.critic = critic or create_adapter_from_env()

    async def generate_and_critique(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        # Generate candidate from primary
        candidate = await self.primary.generate(prompt, max_tokens=max_tokens, temperature=0.7)
        # Build improved critic prompt that asks for JSON output
        critic_prompt = {
            "task": "critique",
            "prompt": prompt,
            "answer": candidate,
            "format": "json",
            "fields": ["score","explanation","categories"]
        }

        # Ask critic for structured JSON
        critic_resp = await self.critic.generate(json.dumps(critic_prompt), max_tokens=256, temperature=0.0, stream=True)

        # If critic_resp is an async iterator (streaming), consume it incrementally
        critic_raw = ''
        parsed = None
        score = None
        try:
            if hasattr(critic_resp, '__aiter__') or hasattr(critic_resp, '__iter__') and not isinstance(critic_resp, str):
                # it's an iterator — iterate and concatenate
                async for chunk in critic_resp:
                    critic_raw += chunk
            else:
                critic_raw = critic_resp

            # attempt parse as before (tolerant JSON or SCORE patterns)
            try:
                parsed = json.loads(critic_raw)
            except Exception:
                import re
                m = re.search(r"\{.*\}", critic_raw, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None
                if parsed is None:
                    m2 = re.search(r"SCORE[:\s]*([0-9]*\.?[0-9]+)", critic_raw, re.IGNORECASE)
                    if m2:
                        try:
                            score = float(m2.group(1))
                            score = max(0.0, min(1.0, score))
                        except Exception:
                            score = None

            if parsed is not None:
                try:
                    score = float(parsed.get('score', 0.0))
                    score = max(0.0, min(1.0, score))
                except Exception:
                    pass
        except Exception:
            critic_raw = ''

        # Objective evaluation via TAULS (or fallback)
        eval_metrics = await evaluate_text(candidate)

        return {
            'candidate': candidate,
            'critic_raw': critic_raw,
            'critic_parsed': parsed,
            'score': score,
            'eval': eval_metrics
        }


async def demo():
    orch = DualLLMOrchestrator()
    res = await orch.generate_and_critique('Explain entropy in one sentence')
    print('Primary:', res['candidate'][:200])
    print('Critic raw:', res['critic_raw'][:200])
    print('Score:', res['score'])


if __name__ == '__main__':
    asyncio.run(demo())
