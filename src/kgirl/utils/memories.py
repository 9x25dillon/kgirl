from fastapi import APIRouter, Body
from typing import List
from sqlmodel import Session
from ..db import engine
from ..models.memory_event import MemoryEvent

router = APIRouter(tags=["memories"], prefix="/memories")

@router.post("")
def add_memories(events: List[dict] = Body(...)):
    items = []
    with Session(engine) as ses:
        for e in events:
            me = MemoryEvent(**e)
            ses.add(me)
            items.append(me)
        ses.commit()
    return {"ok": True, "count": len(items)}

@router.get("")
def list_memories(since: str | None = None, limit: int = 200):
    from sqlmodel import select
    with Session(engine) as ses:
        stmt = select(MemoryEvent).order_by(MemoryEvent.ts.desc()).limit(limit)
        res = ses.exec(stmt).all()
        return [m.model_dump() for m in res]