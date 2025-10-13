from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List
from .services.qgi import api_suggest, api_suggest_async
from .services.retrieval import ingest_texts, search
from .services.unitary_mixer import route_mixture, choose_route

from .services.al_uls import al_uls

app = FastAPI(title="Chaos LLM MVP", version="0.4.0")

class SuggestRequest(BaseModel):
    prefix: str = ""
    state: str = "S0"
    use_semantic: bool = True
    async_eval: bool = False

class SuggestResponse(BaseModel):
    suggestions: List[str]
    qgi: Dict[str, Any]
cursor/bc-f408c7bd-bc2a-48a4-bc8d-0989f628ad52-ef2e

    mixture: Dict[str, float]
    route: str

class IngestRequest(BaseModel):
    docs: List[str]
    namespace: str = "default"

class SearchRequest(BaseModel):
    query: str
    namespace: str = "default"
    top_k: int = 5


class BatchSymbolicRequest(BaseModel):
    calls: List[Dict[str, Any]]

@app.get("/")
async def root() -> Dict[str, Any]:
    return {"ok": True, "service": "Chaos LLM MVP", "version": "0.4.0"}


@app.get("/symbolic/status")
async def symbolic_status() -> Dict[str, Any]:
    return await al_uls.health()

@app.post("/batch_symbolic")
async def batch_symbolic(payload: BatchSymbolicRequest) -> Dict[str, Any]:
    results = await al_uls.batch_eval_symbolic_calls(payload.calls)
    return {"results": results}

@app.post("/suggest", response_model=SuggestResponse)
async def suggest(payload: SuggestRequest) -> SuggestResponse:
    result = await api_suggest_async(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic) if payload.async_eval \
             else api_suggest(prefix=payload.prefix, state=payload.state, use_semantic=payload.use_semantic)
    mixture = route_mixture(result["qgi"]) ; route = choose_route(mixture)
    result["qgi"].setdefault("retrieval_routes", []).append(route)
    return SuggestResponse(suggestions=result["suggestions"], qgi=result["qgi"], mixture=mixture, route=route)

