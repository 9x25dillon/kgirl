from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

from .services.qgi import api_suggest, api_suggest_async
from .services.retrieval import ingest_texts, search  # noqa: F401 (kept for future use)
from .services.unitary_mixer import route_mixture, choose_route
from .services.al_uls import al_uls
from .services.numbskull import numbskull
from .services.newthought import newthought_service

app = FastAPI(title="Chaos LLM MVP", version="0.5.0")


class SuggestRequest(BaseModel):
    prefix: str = ""
    state: str = "S0"
    use_semantic: bool = True
    async_eval: bool = False


class SuggestResponse(BaseModel):
    suggestions: List[str]
    qgi: Dict[str, Any]
    mixture: Dict[str, float]
    route: str


class BatchSymbolicRequest(BaseModel):
    calls: List[Dict[str, Any]]


class NSKInvokeRequest(BaseModel):
    name: str
    args: List[str] = []


class NSKBatchRequest(BaseModel):
    calls: List[Dict[str, Any]]


# NewThought request/response models
class NewThoughtGenerateRequest(BaseModel):
    seed_text: str
    depth: int = 3
    store_in_memory: bool = True


class NewThoughtRecallRequest(BaseModel):
    query_text: str
    top_k: int = 5
    similarity_threshold: float = 0.5


class NewThoughtSuperposeRequest(BaseModel):
    thought_texts: List[str]
    weights: List[float] = None


class NewThoughtEntanglementRequest(BaseModel):
    thought_text_a: str
    thought_text_b: str


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "service": "Chaos LLM MVP", "version": "0.4.0"}
    return {"ok": True, "service": app.title, "version": app.version}


@app.get("/symbolic/status")
async def symbolic_status() -> Dict[str, Any]:
    return await al_uls.health()


@app.post("/batch_symbolic")
async def batch_symbolic(payload: BatchSymbolicRequest) -> Dict[str, Any]:
    results = await al_uls.batch_eval_symbolic_calls(payload.calls)
    return {"results": results}


@app.post("/suggest", response_model=SuggestResponse)
async def suggest(payload: SuggestRequest) -> SuggestResponse:
    result = (
        await api_suggest_async(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic)
        if payload.async_eval
        else api_suggest(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic)
    )
    mixture = route_mixture(result["qgi"])  # type: ignore[arg-type]
    route = choose_route(mixture)
    result["qgi"].setdefault("retrieval_routes", []).append(route)
    return SuggestResponse(suggestions=result["suggestions"], qgi=result["qgi"], mixture=mixture, route=route)


# Numbskull integration endpoints (optional, enabled by env URLs)
@app.get("/numbskull/status")
async def numbskull_status() -> Dict[str, Any]:
    return await numbskull.health()


@app.get("/numbskull/tools")
async def numbskull_tools() -> Dict[str, Any]:
    return await numbskull.list_tools()


@app.post("/numbskull/invoke")
async def numbskull_invoke(payload: NSKInvokeRequest) -> Dict[str, Any]:
    return await numbskull.invoke(payload.name, payload.args)


@app.post("/numbskull/batch")
async def numbskull_batch(payload: NSKBatchRequest) -> Dict[str, Any]:
    results = await numbskull.batch_invoke(payload.calls)
    return {"results": results}


# NewThought endpoints - Quantum-Inspired Neural Coherence Recovery
@app.get("/newthought/status")
async def newthought_status() -> Dict[str, Any]:
    """Health check for NewThought service."""
    return newthought_service.health_check()


@app.get("/newthought/stats")
async def newthought_stats() -> Dict[str, Any]:
    """Get comprehensive NewThought service statistics."""
    return newthought_service.get_statistics()


@app.post("/newthought/generate")
async def newthought_generate(payload: NewThoughtGenerateRequest) -> Dict[str, Any]:
    """
    Generate a new thought cascade using quantum-inspired coherence recovery.

    Implements recursive thought generation with:
    - Spatial encoding with locality preservation
    - Quantum coherence recovery
    - Integrity validation
    - Holographic memory storage
    """
    result = await newthought_service.generate_new_thought(
        seed_text=payload.seed_text,
        depth=payload.depth,
        store_in_memory=payload.store_in_memory,
    )
    return result


@app.post("/newthought/recall")
async def newthought_recall(payload: NewThoughtRecallRequest) -> Dict[str, Any]:
    """
    Recall similar thoughts from holographic memory.

    Uses content-addressable storage with associative recall based on
    semantic similarity and spatial locality.
    """
    results = await newthought_service.recall_similar_thoughts(
        query_text=payload.query_text,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
    )
    return {"similar_thoughts": results, "query": payload.query_text}


@app.post("/newthought/superpose")
async def newthought_superpose(payload: NewThoughtSuperposeRequest) -> Dict[str, Any]:
    """
    Create quantum superposition of multiple thoughts.

    Combines multiple thought states with amplitude weighting to create
    a coherent superposition following quantum mechanics principles.
    """
    result = await newthought_service.quantum_superpose_thoughts(
        thought_texts=payload.thought_texts,
        weights=payload.weights,
    )
    return result


@app.post("/newthought/entanglement")
async def newthought_entanglement(payload: NewThoughtEntanglementRequest) -> Dict[str, Any]:
    """
    Measure quantum entanglement between two thoughts.

    Uses von Neumann entropy to quantify correlations and entanglement
    between thought states.
    """
    result = await newthought_service.measure_thought_entanglement(
        thought_text_a=payload.thought_text_a,
        thought_text_b=payload.thought_text_b,
    )
    return result

