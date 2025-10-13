import os
import re
import json
import asyncio
import time
from typing import Any, Dict, List, Tuple
import httpx
import websockets

NS_HTTP_URL = os.environ.get("NUMBSKULL_HTTP_URL", "http://localhost:9090")
NS_WS_URL = os.environ.get("NUMBSKULL_WS_URL", "")
NS_PREFER_WS = os.environ.get("NUMBSKULL_PREFER_WS", "1") in {"1", "true", "TRUE", "yes"}
NS_TTL = float(os.environ.get("NUMBSKULL_TTL", 30))

TOOL_RE = re.compile(r"\bTOOL\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*(.*?))?\s*\)$")


class TTLCache:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self._store: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, Any]]] = {}

    def _now(self) -> float:
        return time.monotonic()

    def _key(self, name: str, args: List[str]) -> Tuple[str, Tuple[str, ...]]:
        return (name.upper(), tuple(args))

    def get(self, name: str, args: List[str]) -> Dict[str, Any] | None:
        k = self._key(name, args)
        v = self._store.get(k)
        if not v:
            return None
        ts, data = v
        if self._now() - ts <= self.ttl:
            return data
        self._store.pop(k, None)
        return None

    def set(self, name: str, args: List[str], value: Dict[str, Any]) -> None:
        self._store[self._key(name, args)] = (self._now(), value)


class NumbskullHTTP:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.client = httpx.AsyncClient(timeout=10)
        self.cache = TTLCache(NS_TTL)

    async def health(self) -> Dict[str, Any]:
        try:
            r = await self.client.get(f"{self.base}/health")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def list_tools(self) -> Dict[str, Any]:
        try:
            r = await self.client.get(f"{self.base}/v1/tools")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def invoke(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        try:
            r = await self.client.post(f"{self.base}/v1/invoke", json={"name": name, "args": args})
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("ok"):
                self.cache.set(name, args, data)
            return data
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def batch(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = [{} for _ in calls]
        to_run: List[Tuple[int, Dict[str, Any]]] = []
        for i, c in enumerate(calls):
            nm = c.get("name", "").upper(); ag = c.get("args", [])
            cached = self.cache.get(nm, ag)
            if cached is not None:
                results[i] = {**cached, "_cached": True}
            else:
                to_run.append((i, {"name": nm, "args": ag}))
        outs = await asyncio.gather(*[self.invoke(c["name"], c["args"]) for _, c in to_run], return_exceptions=True)
        for (i, _), out in zip(to_run, outs):
            results[i] = out if not isinstance(out, Exception) else {"ok": False, "error": str(out)}
        return results


class NumbskullWS:
    def __init__(self, url: str):
        self.url = url
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.cache = TTLCache(NS_TTL)

    async def connect(self):
        if (self.ws is None) or self.ws.closed:
            self.ws = await websockets.connect(self.url)
        return self.ws

    async def _rt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ws = await self.connect()
            await ws.send(json.dumps(payload))
            resp = await ws.recv()
            return json.loads(resp)
        except Exception as e:
            try:
                if self.ws:
                    await self.ws.close()
            finally:
                self.ws = None
            return {"ok": False, "error": str(e)}

    async def health(self) -> Dict[str, Any]:
        return await self._rt({"type": "health"})

    async def list_tools(self) -> Dict[str, Any]:
        return await self._rt({"type": "tools"})

    async def invoke(self, name: str, args: List[str]) -> Dict[str, Any]:
        cached = self.cache.get(name, args)
        if cached is not None:
            return {**cached, "_cached": True}
        res = await self._rt({"type": "invoke", "name": name, "args": args})
        if isinstance(res, dict) and res.get("ok"):
            self.cache.set(name, args, res)
        return res

    async def batch(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        res = await self._rt({"type": "batch", "calls": calls})
        if isinstance(res, dict) and isinstance(res.get("results"), list):
            out: List[Dict[str, Any]] = []
            for c, r in zip(calls, res["results"]):
                if isinstance(r, dict) and r.get("ok"):
                    self.cache.set(c.get("name", ""), c.get("args", []), r)
                out.append(r if isinstance(r, dict) else {"ok": False, "error": "invalid item"})
            return out
        return [await self.invoke(c.get("name", ""), c.get("args", [])) for c in calls]


class Numbskull:
    def __init__(self):
        self.http = NumbskullHTTP(NS_HTTP_URL)
        self.ws = NumbskullWS(NS_WS_URL) if NS_WS_URL else None

    def is_tool_call(self, text: str) -> bool:
        return bool(TOOL_RE.search((text or "").strip()))

    def parse_tool_call(self, text: str) -> Dict[str, Any]:
        m = TOOL_RE.search((text or "").strip())
        if not m:
            return {"name": None, "args": []}
        name = m.group(1)
        argstr = (m.group(2) or "").strip()
        args = [a.strip() for a in argstr.split(",") if a.strip()] if argstr else []
        return {"name": name.upper(), "args": args}

    async def health(self) -> Dict[str, Any]:
        if NS_PREFER_WS and self.ws is not None:
            res = await self.ws.health()
            if isinstance(res, dict) and (res.get("ok") or res.get("_cached")):
                return res
        return await self.http.health()

    async def list_tools(self) -> Dict[str, Any]:
        if NS_PREFER_WS and self.ws is not None:
            res = await self.ws.list_tools()
            if isinstance(res, dict) and (res.get("ok") or res.get("tools")):
                return res
        return await self.http.list_tools()

    async def invoke(self, name: str, args: List[str]) -> Dict[str, Any]:
        if NS_PREFER_WS and self.ws is not None:
            res = await self.ws.invoke(name, args)
            if isinstance(res, dict) and (res.get("ok") or res.get("_cached")):
                return res
        return await self.http.invoke(name, args)

    async def batch_invoke(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if NS_PREFER_WS and self.ws is not None:
            res = await self.ws.batch(calls)
            if isinstance(res, list) and any(isinstance(r, dict) for r in res):
                return res
        return await self.http.batch(calls)


numbskull = Numbskull()
