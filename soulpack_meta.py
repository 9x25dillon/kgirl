from typing import Optional
from sqlmodel import SQLModel, Field

class SoulpackMeta(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    version: str
    persona_name: str
    pubkey: str
    checksum: str
    raw: str  # the JSON soulpack