import os
import time
import asyncio
from typing import Dict, Any, List, Tuple
import httpx

JULIA_SERVER_URL = os.environ.get("JULIA_SERVER_URL", "http://localhost:8088")
CACHE_TTL_SECONDS = float(os.environ.get("ALULS_HTTP_TTL", 30))

class TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, Any]]] = {}
        self.hits = 0
        self.misses = 0

    def _now(self) -> float:
        return time.monotonic()

    def _key(self, name: str, args: List[str]) -> Tuple[str, Tuple[str, ...]]:
        return (name.upper(), tuple(args))

    def get(self, name: str, args: List[str]) -> Dict[str, Any] | None:
        k = self._key(name, args)
        v = self._store.get(k)
        if not v:
            self.misses += 1
            return None
        ts, data = v
        if self._now() - ts <= self.ttl:
            self.hits += 1
            return data
        self._store.pop(k, None)
        self.misses += 1
        return None

    def set(self, name: str, args: List[str], value: Dict[str, Any]) -> None:
        self._store[self._key(name, args)] = (self._now(), value)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._store), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}

class ALULSClient:
    def __init__(self, base_url: str | None = None):
        self.base = base_url or JULIA_SERVER_URL
        self.client = httpx.AsyncClient(timeout=10)
        self.cache = TTLCache(CACHE_TTL_SECONDS)

    async def health(self) -> Dict[str, Any]:
        try:
            r = await self.client.get(f"{self.base}/health")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def parse(self, text: str) -> Dict[str, Any]:
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/parse", json={"text": text})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def eval(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        try:
            r = await self.client.post(f"{self.base}/v1/symbolic/eval", json={"name": name, "args": args})
            r.raise_for_status()
            data = r.json()
            if data.get("ok"):
                self.cache.set(name, args, data)
            return data
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch_eval(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Use cache per-call; run only misses concurrently
        to_run: List[Tuple[int, Dict[str, Any]]] = []
        results: List[Dict[str, Any]] = [{} for _ in calls]
        for i, c in enumerate(calls):
            name = c.get("name", "").upper(); args = c.get("args", [])
            cached = self.cache.get(name, args)
            if cached is not None:
                results[i] = {**cached, "_cached": True}
            else:
                to_run.append((i, {"name": name, "args": args}))
        tasks = [self.eval(c["name"], c["args"]) for _, c in to_run]
        outs = await asyncio.gather(*tasks, return_exceptions=True)
        for (i, _), out in zip(to_run, outs):
            results[i] = out if not isinstance(out, Exception) else {"ok": False, "error": str(out)}
        return results

al_uls_client = ALULSClient()
