from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

class MemoryEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str
    ts: datetime
    type: str
    subject: str
    data: str  # JSON string for portability
    sensitivity: str = "low"
    consent: str = "retain"
    hash: str
    # retrieval helpers
    recency_score: float = 0.0
    graph_degree: float = 0.0