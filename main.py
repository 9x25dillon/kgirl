from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, soulpacks, memories, prime, tools
from .config import settings

app = FastAPI(title="CarryOn API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "name": "carryon", "version": "0.1.0"}

app.include_router(health.router, prefix="/v1")
app.include_router(soulpacks.router, prefix="/v1")
app.include_router(memories.router, prefix="/v1")
app.include_router(prime.router, prefix="/v1")
app.include_router(tools.router, prefix="/v1")