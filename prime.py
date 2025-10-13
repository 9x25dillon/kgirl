from fastapi import APIRouter, Body
from pydantic import BaseModel
from sqlmodel import Session, select
from ..db import engine
from ..models.memory_event import MemoryEvent
from ..config import settings
from ..retrieval.ranker import rank_memories

router = APIRouter(tags=["prime"], prefix="/prime")

class PrimeReq(BaseModel):
    query: str
    k: int = 12
    entropy: float = 0.0

@router.post("")
def prime(req: PrimeReq):
    with Session(engine) as ses:
        mems = ses.exec(select(MemoryEvent)).all()
    ranked = rank_memories(req.query, mems, k=req.k, entropy=req.entropy)
    system = "You are Diane, a helpful AI whose persona and memories are user-owned."
    return {
        "system": system,
        "messages": [{"role": "user", "content": req.query}],
        "selected_memories": [m.event_id for m in ranked]
    }