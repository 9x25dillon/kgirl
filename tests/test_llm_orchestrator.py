import asyncio
from llm_orchestrator import DualLLMOrchestrator


def test_orchestrator_stub():
    orch = DualLLMOrchestrator()
    res = asyncio.get_event_loop().run_until_complete(orch.generate_and_critique('Hello test'))
    assert 'candidate' in res
    assert 'critic_raw' in res
    assert 'eval' in res
