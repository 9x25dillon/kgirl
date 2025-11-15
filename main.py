from __future__ import annotations

import os
import sys
from typing import List, Optional, Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# ----- CTH import path wiring -----
load_dotenv(override=True)
CTH_PATH = os.getenv("CTH_PATH")
if CTH_PATH and CTH_PATH not in sys.path:
    sys.path.append(CTH_PATH)
try:
    from CTH.topological_consciousness import TopologicalConsciousness  # type: ignore
except Exception:  # allow running without CTH for fallback demos
    TopologicalConsciousness = None  # type: ignore

# ----- Optional embedder for Anthropic fallback -----
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # optional dependency
    SentenceTransformer = None  # type: ignore

# ----- Ollama client for local LLM -----
try:
    import ollama
except Exception:  # optional dependency
    ollama = None  # type: ignore


# ----- Adapters -----
class BaseAdapter:
    name: str = "base"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def get_hidden_states(self, prompt: str) -> np.ndarray:
        # Fallback: use embedding of prompt as a proxy state
        return self.embed(prompt)


class OpenAIAdapter(BaseAdapter):
    def __init__(self, chat_model: str, embed_model: str, api_key: Optional[str] = None):
        from openai import OpenAI  # lazy import

        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.name = f"openai:{chat_model}"

    def generate(self, prompt: str) -> str:
        r = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return r.choices[0].message.content or ""

    def embed(self, text: str) -> np.ndarray:
        r = self.client.embeddings.create(model=self.embed_model, input=text)
        return np.array(r.data[0].embedding, dtype=np.float64)


class AnthropicAdapter(BaseAdapter):
    def __init__(
        self,
        chat_model: str,
        api_key: Optional[str] = None,
        embedder: Optional[Any] = None,
        openai_embed_fallback: Optional[OpenAIAdapter] = None,
    ):
        from anthropic import Anthropic  # lazy import

        self.client = Anthropic(api_key=api_key)
        self.chat_model = chat_model
        self.embedder = embedder
        self.openai_embed_fallback = openai_embed_fallback
        self.name = f"anthropic:{chat_model}"

    def generate(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        # anthropic returns content blocks
        return "".join(getattr(c, "text", "") for c in msg.content)

    def embed(self, text: str) -> np.ndarray:
        if self.openai_embed_fallback is not None:
            return self.openai_embed_fallback.embed(text)
        if self.embedder is not None:
            return np.asarray(self.embedder.encode(text), dtype=np.float64)
        raise RuntimeError("No embedder available for AnthropicAdapter")


class OllamaAdapter(BaseAdapter):
    """Local LLM adapter using Ollama - no API keys required!"""

    def __init__(
        self,
        chat_model: str = "qwen2.5:3b",
        embed_model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        embedder: Optional[Any] = None,
    ):
        if ollama is None:
            raise RuntimeError("ollama package not installed. Install with: pip install ollama")

        self.chat_model = chat_model
        self.embed_model = embed_model
        self.host = host
        self.embedder = embedder  # fallback to sentence-transformers if Ollama embedding fails
        self.name = f"ollama:{chat_model}"

        # Configure ollama client with custom host if needed
        if host != "http://localhost:11434":
            import os
            os.environ["OLLAMA_HOST"] = host

    def generate(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2}
            )
            return response["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}. Is Ollama running?")

    def embed(self, text: str) -> np.ndarray:
        # Try Ollama embedding first
        try:
            response = ollama.embeddings(model=self.embed_model, prompt=text)
            return np.array(response["embedding"], dtype=np.float64)
        except Exception:
            # Fallback to sentence-transformers if available
            if self.embedder is not None:
                return np.asarray(self.embedder.encode(text), dtype=np.float64)
            # If no embedder available, raise error
            raise RuntimeError(
                f"Ollama embedding failed and no fallback embedder available. "
                f"Pull the embedding model with: ollama pull {self.embed_model}"
            )


# ----- Coherence & Energy helpers -----

def _l2n(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        n = np.linalg.norm(X) + 1e-12
        return X / n
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def spectral_weights(embs: np.ndarray) -> np.ndarray:
    """Principal-eigenvector weights on cosine kernel (fallback)."""
    E = _l2n(embs)
    K = E @ E.T  # symmetric
    # Use eigh (for symmetric matrices) to get real eigenpairs
    vals, vecs = np.linalg.eigh(K + 1e-9 * np.eye(K.shape[0]))
    w = np.real(vecs[:, -1])  # principal eigenvector
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / (s + 1e-12)


class PhaseCoherence:
    def __init__(self, n_anyons: int, central_charge: int = 627):
        self.use_cth = TopologicalConsciousness is not None
        if self.use_cth:
            self.tc = TopologicalConsciousness(  # type: ignore
                n_anyons=n_anyons, central_charge=central_charge
            )
        else:
            self.tc = None

    def weights(self, embeddings: np.ndarray) -> np.ndarray:
        if self.use_cth and hasattr(self.tc, "calculate_modular_invariance"):
            try:
                w = np.asarray(self.tc.calculate_modular_invariance(embeddings), dtype=np.float64)
                return w / (w.sum() + 1e-12)
            except Exception:
                pass
        return spectral_weights(embeddings)

    def scalar(self, embeddings: np.ndarray) -> float:
        w = self.weights(embeddings)
        n = len(w)
        return float((w.max() - 1.0 / n) / (1.0 - 1.0 / n + 1e-12))


class CardyEnergy:
    def __init__(self, n_anyons: int = 27, central_charge: int = 627):
        self.use_cth = TopologicalConsciousness is not None
        if self.use_cth:
            self.tc = TopologicalConsciousness(  # type: ignore
                n_anyons=n_anyons, central_charge=central_charge
            )
            self.max_energy = float(getattr(self.tc, "max_energy", 1.0))
        else:
            self.tc = None
            self.max_energy = 1.0

    def boundary(self, state: Any) -> float:
        if self.use_cth and hasattr(self.tc, "cardy_boundary_energy"):
            try:
                return float(self.tc.cardy_boundary_energy(state))
            except Exception:
                pass
        # heuristic fallback: dispersion proxy
        s = _l2n(np.asarray(state, dtype=np.float64).ravel())
        return float(1.0 - np.abs(s).mean())


# ----- Trinary quantization -----

def quantize_trinary(vec: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    out = np.zeros_like(v, dtype=np.int8)
    out[v > threshold] = 1
    out[v < -threshold] = -1
    return out


# ----- Settings & App -----
class Settings(BaseSettings):
    # API Keys (optional - only needed for cloud providers)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Cloud LLM Models
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    anthropic_chat_model: str = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-5-sonnet-latest")

    # Ollama (Local) Configuration
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Use local LLMs by default (set to "true" to use Ollama, "false" for cloud APIs)
    use_local_llm: bool = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"

    # Multi-Model Pool Configuration
    # Default to Ollama for local-first operation
    # To use cloud providers: MODELS="openai:chat=gpt-4o-mini,embed=text-embedding-3-large|anthropic:chat=claude-3-5-sonnet-latest"
    models: str = os.getenv(
        "MODELS", "ollama:chat=qwen2.5:3b,embed=nomic-embed-text"
    )

    # Topological Parameters
    central_charge: int = int(os.getenv("CENTRAL_CHARGE", "627"))
    n_anyons: int = int(os.getenv("N_ANYONS", "5"))


settings = Settings()

app = FastAPI(title="Topological Consensus API", version="0.1.0")

# ----- Build model pool at startup -----
g_model_pool: List[BaseAdapter] = []


@app.on_event("startup")
def _startup() -> None:
    global g_model_pool
    g_model_pool = []

    # Load sentence-transformers for embedding fallback
    st = None
    if SentenceTransformer:
        try:
            st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("✓ Loaded sentence-transformers for embedding fallback")
        except Exception as e:
            print(f"⚠ Failed to load sentence-transformers: {e}")

    # parse MODELS var like: "ollama:chat=...,embed=...|openai:chat=...|anthropic:chat=..."
    groups = [p.strip() for p in settings.models.split("|") if p.strip()]
    openai_embed_adapter: Optional[OpenAIAdapter] = None

    for g in groups:
        if ":" not in g:
            continue
        kind, spec = g.split(":", 1)
        kv = dict(item.split("=", 1) for item in spec.split(",") if "=" in item)

        if kind == "ollama":
            # LOCAL LLM - No API keys needed!
            chat = kv.get("chat", settings.ollama_chat_model)
            emb = kv.get("embed", settings.ollama_embed_model)
            host = kv.get("host", settings.ollama_host)
            try:
                ol = OllamaAdapter(
                    chat_model=chat,
                    embed_model=emb,
                    host=host,
                    embedder=st,
                )
                g_model_pool.append(ol)
                print(f"✓ Loaded Ollama adapter: {chat} (local, no API key required)")
            except Exception as e:
                print(f"⚠ Failed to initialize Ollama adapter: {e}")
                print(f"  Make sure Ollama is running: ollama serve")
                print(f"  And models are pulled: ollama pull {chat} && ollama pull {emb}")

        elif kind == "openai":
            # Cloud API - requires API key
            if not settings.openai_api_key:
                print(f"⚠ Skipping OpenAI adapter: OPENAI_API_KEY not set")
                continue
            chat = kv.get("chat", settings.openai_chat_model)
            emb = kv.get("embed", settings.openai_embed_model)
            try:
                oa = OpenAIAdapter(chat_model=chat, embed_model=emb, api_key=settings.openai_api_key)
                g_model_pool.append(oa)
                openai_embed_adapter = oa
                print(f"✓ Loaded OpenAI adapter: {chat}")
            except Exception as e:
                print(f"⚠ Failed to initialize OpenAI adapter: {e}")

        elif kind == "anthropic":
            # Cloud API - requires API key
            if not settings.anthropic_api_key:
                print(f"⚠ Skipping Anthropic adapter: ANTHROPIC_API_KEY not set")
                continue
            chat = kv.get("chat", settings.anthropic_chat_model)
            try:
                an = AnthropicAdapter(
                    chat_model=chat,
                    api_key=settings.anthropic_api_key,
                    embedder=st,
                    openai_embed_fallback=openai_embed_adapter,
                )
                g_model_pool.append(an)
                print(f"✓ Loaded Anthropic adapter: {chat}")
            except Exception as e:
                print(f"⚠ Failed to initialize Anthropic adapter: {e}")

    if not g_model_pool:
        print("\n" + "="*60)
        print("⚠ WARNING: No models configured!")
        print("="*60)
        print("\nTo use LOCAL models (recommended, no API keys needed):")
        print("  1. Install Ollama: https://ollama.ai")
        print("  2. Start Ollama: ollama serve")
        print("  3. Pull models:")
        print("     ollama pull qwen2.5:3b")
        print("     ollama pull nomic-embed-text")
        print("  4. Set in .env: MODELS=ollama:chat=qwen2.5:3b,embed=nomic-embed-text")
        print("\nTo use CLOUD APIs (requires API keys):")
        print("  1. Set API keys in .env:")
        print("     OPENAI_API_KEY=sk-...")
        print("     ANTHROPIC_API_KEY=sk-ant-...")
        print("  2. Set MODELS in .env (example):")
        print("     MODELS=openai:chat=gpt-4o-mini,embed=text-embedding-3-large")
        print("="*60 + "\n")
    else:
        print(f"\n✓ Successfully loaded {len(g_model_pool)} model adapter(s)")
        print(f"  Models: {[m.name for m in g_model_pool]}\n")


# ----- Schemas -----
class AskRequest(BaseModel):
    prompt: str
    min_coherence: float = 0.80
    max_energy: float = 0.30
    return_all: bool = False


class AskResponse(BaseModel):
    answer: Optional[str]
    decision: str
    coherence: float
    energy: float
    weights: List[float]
    model_names: List[str]
    all_outputs: Optional[List[str]] = None


class Doc(BaseModel):
    id: str
    text: str
    embedding: Optional[List[float]] = None


class RerankRequest(BaseModel):
    query: str
    docs: List[Doc]
    trinary_threshold: float = 0.25
    alpha: float = 0.7  # similarity weight
    beta: float = 0.3  # coherence weight


class RerankResponse(BaseModel):
    ranked_ids: List[str]
    scores: List[float]


# ----- Endpoints -----
@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "models": [getattr(m, "name", "?") for m in g_model_pool],
        "cth": TopologicalConsciousness is not None,
    }


@app.get("/config")
def config() -> dict:
    return {
        "central_charge": settings.central_charge,
        "n_anyons": settings.n_anyons,
        "models": [getattr(m, "name", "?") for m in g_model_pool],
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    assert g_model_pool, "no models configured"

    # 1) generate outputs
    outputs: List[str] = [m.generate(req.prompt) for m in g_model_pool]

    # 2) embed outputs (consensus on semantics of answers)
    embs = np.stack([m.embed(o) for o in outputs], axis=0)

    # 3) phase coherence
    pc = PhaseCoherence(
        n_anyons=max(len(g_model_pool), settings.n_anyons),
        central_charge=settings.central_charge,
    )
    w = pc.weights(embs)
    coh = pc.scalar(embs)

    # 4) energy (hallucination risk) — use first model's hidden state as proxy
    energy = CardyEnergy(n_anyons=27, central_charge=settings.central_charge).boundary(
        g_model_pool[0].get_hidden_states(req.prompt)
    )

    # 5) decision logic
    decision: str
    answer: Optional[str] = None
    best = int(np.argmax(w))
    if coh >= req.min_coherence and energy <= req.max_energy:
        decision = "auto"
        answer = outputs[best]
    elif coh >= 0.5 and energy <= 0.5:
        decision = "needs_citations"
        answer = outputs[best]
    else:
        decision = "escalate"
        answer = None

    return AskResponse(
        answer=answer,
        decision=decision,
        coherence=float(coh),
        energy=float(energy),
        weights=[float(x) for x in w.tolist()],
        model_names=[m.name for m in g_model_pool],
        all_outputs=outputs if req.return_all else None,
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    assert g_model_pool, "no models configured"

    # choose an embedder (first model)
    emb_model = g_model_pool[0]

    # query embedding
    q = emb_model.embed(req.query)
    q = _l2n(q)

    # doc embeddings (compute if missing)
    D: List[np.ndarray] = []
    for d in req.docs:
        if d.embedding is not None:
            vec = np.asarray(d.embedding, dtype=np.float64)
        else:
            vec = emb_model.embed(d.text)
        D.append(vec)
    E = _l2n(np.stack(D, axis=0))

    # similarity scores
    sims = (E @ q).reshape(-1)

    # trinary quantization (for logging / potential downstream)
    _ = [quantize_trinary(v, threshold=req.trinary_threshold) for v in E]

    # coherence weights among docs (who clusters together)
    w = spectral_weights(E)

    # combined score
    scores = req.alpha * sims + req.beta * w
    order = np.argsort(-scores)

    return RerankResponse(
        ranked_ids=[req.docs[i].id for i in order],
        scores=[float(scores[i]) for i in order],
    )


# ----- If run as a script -----
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MAIN_API_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
