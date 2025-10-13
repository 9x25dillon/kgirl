import asyncio
import os
from typing import Optional
import requests
import json
import threading


class LLMAdapter:
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, stream: bool = False):
        raise NotImplementedError()


class LocalStubAdapter(LLMAdapter):
    def __init__(self, name: str = 'local-stub'):
        self.name = name

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, stream: bool = False):
        out = f"[STUB:{self.name}] " + prompt
        if len(out) > max_tokens * 4:
            out = out[:max_tokens * 4] + '...'
        if stream:
            async def gen():
                for i in range(0, len(out), 64):
                    await asyncio.sleep(0)
                    yield out[i:i+64]
            return gen()
        return out


class RemoteHTTPAdapter(LLMAdapter):
    def __init__(self, endpoint: str, api_key: Optional[str] = None, timeout: float = 10.0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, stream: bool = False):
        payload = {'prompt': prompt, 'max_tokens': max_tokens, 'temperature': temperature}
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        loop = asyncio.get_event_loop()

        def _post(streaming=False):
            return requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout, stream=streaming)

        if stream:
            q: 'asyncio.Queue[Optional[str]]' = asyncio.Queue()

            def _reader():
                try:
                    r = _post(streaming=True)
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024):
                        if not chunk:
                            continue
                        try:
                            text = chunk.decode('utf-8', errors='ignore')
                        except Exception:
                            text = str(chunk)
                        loop.call_soon_threadsafe(q.put_nowait, text)
                    loop.call_soon_threadsafe(q.put_nowait, None)
                except Exception:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()

            async def _aiter():
                parts = []
                while True:
                    chunk = await q.get()
                    if chunk is None:
                        break
                    parts.append(chunk)
                    yield chunk
                # finished streaming

            return _aiter()

        # non-streaming
        r = await loop.run_in_executor(None, _post, False)
        r.raise_for_status()
        try:
            resp = r.json()
        except Exception:
            resp = {'text': r.text}

        if isinstance(resp, dict):
            txt = resp.get('text') or resp.get('result')
            if not txt:
                for k in ('output', 'generation', 'data'):
                    if k in resp:
                        txt = resp[k]
                        break
            if isinstance(txt, list):
                txt = ' '.join(str(x) for x in txt)
            if txt:
                return txt
            return json.dumps(resp)
        return str(resp)


def create_adapter_from_env():
    endpoint = os.environ.get('LLM_ENDPOINT')
    if endpoint:
        api_key = os.environ.get('LLM_API_KEY')
        return RemoteHTTPAdapter(endpoint, api_key)
    return LocalStubAdapter('auto-stub')


# Provider-specific adapters
class OpenAIAdapter(LLMAdapter):
    """Simple OpenAI-style REST adapter (completions-like)."""
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None, timeout: float = 15.0):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.endpoint = endpoint or os.environ.get('OPENAI_ENDPOINT', 'https://api.openai.com/v1/completions')
        self.timeout = timeout

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, stream: bool = False):
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'} if self.api_key else {'Content-Type': 'application/json'}
        payload = {'model': 'gpt-like', 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': temperature}

        loop = asyncio.get_event_loop()

        def _call(streaming=False):
            return requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout, stream=streaming)

        if stream:
            q = asyncio.Queue()

            def _reader():
                try:
                    r = _call(streaming=True)
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024):
                        if not chunk:
                            continue
                        try:
                            s = chunk.decode('utf-8')
                        except Exception:
                            s = str(chunk)
                        loop.call_soon_threadsafe(q.put_nowait, s)
                    loop.call_soon_threadsafe(q.put_nowait, None)
                except Exception:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()

            async def _aiter():
                while True:
                    chunk = await q.get()
                    if chunk is None:
                        break
                    yield chunk

            return _aiter()

        r = await loop.run_in_executor(None, _call, False)
        r.raise_for_status()
        try:
            out = r.json()
            # try expected shapes
            if isinstance(out, dict):
                if 'choices' in out and isinstance(out['choices'], list):
                    txt = out['choices'][0].get('text')
                    if txt:
                        return txt
                if 'text' in out:
                    return out['text']
        except Exception:
            pass
        return r.text


class LocalSubprocessAdapter(LLMAdapter):
    """Lightweight example adapter that shells out to a local command (simulates ggml/llama.cpp).

    This is intentionally minimal: production adapters should handle pipes, timeouts and streaming.
    """
    def __init__(self, cmd: str = 'echo', timeout: float = 10.0):
        self.cmd = cmd
        self.timeout = timeout

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, stream: bool = False):
        # Use run_in_executor to avoid blocking event loop
        loop = asyncio.get_event_loop()

        def _run():
            import subprocess
            try:
                # use echo for safe default; real ggml adapter would call binary with args
                p = subprocess.run([self.cmd, prompt], capture_output=True, text=True, timeout=self.timeout)
                return p.stdout or p.stderr
            except Exception as e:
                return str(e)

        if stream:
            # no streaming support here — return an async iterator that yields once
            async def _aiter():
                out = await loop.run_in_executor(None, _run)
                yield out
            return _aiter()

        out = await loop.run_in_executor(None, _run)
        return out


def create_provider_adapter(provider: str = None):
    provider = provider or os.environ.get('LLM_PROVIDER')
    if provider == 'openai' or os.environ.get('OPENAI_API_KEY'):
        return OpenAIAdapter()
    if provider == 'local':
        return LocalSubprocessAdapter()
    return create_adapter_from_env()
