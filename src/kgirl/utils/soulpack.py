from pydantic import BaseModel
from typing import List, Dict, Optional

class Persona(BaseModel):
    name: str
    aliases: List[str] = []
    voice: Dict[str, List[str]] = {}
    values: List[str] = []
    boundaries: List[str] = []
    expertise_tags: List[str] = []

class Pointers(BaseModel):
    vector: str
    graph: str
    memory_feed: Optional[str] = None
    tools_manifest: Optional[str] = None
    consent_policy: Optional[str] = None

class Soulpack(BaseModel):
    version: str = "0.1.0"
    persona: Persona
    preferences: Dict[str, str] = {}
    pointers: Pointers
    signing: Dict[str, str]