"""Model backends. Nothing here decides *which* model to use — see hermes.router."""

from .anthropic_backend import DEFAULT_SCOUT_MODEL, AnthropicBackend
from .base import Backend, BackendError, Completion, Message
from .ollama_backend import OllamaBackend
from .scripted import ScriptedBackend

__all__ = ["Backend", "BackendError", "Completion", "Message", "AnthropicBackend", "OllamaBackend",
           "ScriptedBackend", "DEFAULT_SCOUT_MODEL"]
