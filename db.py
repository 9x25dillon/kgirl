from sqlmodel import SQLModel, create_engine
from .config import settings

engine = create_engine(settings.db_url, echo=False)

def init_db():
    from .models.memory_event import MemoryEvent
    from .models.soulpack_meta import SoulpackMeta
    SQLModel.metadata.create_all(engine)