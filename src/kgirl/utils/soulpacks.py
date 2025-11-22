from fastapi import APIRouter, Body, HTTPException
from sqlmodel import Session
from ..db import engine
from ..models.soulpack_meta import SoulpackMeta
from ..core.soulpack import Soulpack

router = APIRouter(tags=["soulpacks"], prefix="/soulpacks")

@router.post("/import")
def import_soulpack(payload: dict = Body(...)):
    sp = Soulpack(**payload)
    with Session(engine) as ses:
        ses.add(SoulpackMeta(
            version=sp.version,
            persona_name=sp.persona.name,
            pubkey=sp.signing.get("pubkey", ""),
            checksum=sp.signing.get("checksum", ""),
            raw=sp.model_dump_json()
        ))
        ses.commit()
    return {"ok": True}

@router.get("/export")
def export_soulpack():
    from sqlmodel import select
    with Session(engine) as ses:
        row = ses.exec(select(SoulpackMeta).order_by(SoulpackMeta.id.desc())).first()
        if not row:
            raise HTTPException(404, "No soulpack available")
        return row.raw