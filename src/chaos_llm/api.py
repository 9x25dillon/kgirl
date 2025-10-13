from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

from .services.qgi import api_suggest, api_suggest_async
from .services.retrieval import ingest_texts, search  # noqa: F401 (kept for future use)
from .services.unitary_mixer import route_mixture, choose_route
from .services.al_uls import al_uls
from .services.numbskull import numbskull

app = FastAPI(title="Chaos LLM MVP", version="0.4.0")


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

