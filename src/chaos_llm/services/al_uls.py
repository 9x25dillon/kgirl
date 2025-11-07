import os
import re
from typing import Any, Dict, List
from .al_uls_client import al_uls_client
from .al_uls_ws_client import al_uls_ws_client

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)$")
PREFER_WS = os.environ.get("ALULS_PREFER_WS", "1") in {"1", "true", "TRUE", "yes"}


class ALULS:
    def is_symbolic_call(self, text: str) -> bool:
        return bool(CALL_RE.search((text or "").strip()))

    def parse_symbolic_call(self, text: str) -> Dict[str, Any]:
        m = CALL_RE.search((text or "").strip())
        if not m:
            return {"name": None, "args": []}
        name, argstr = m.group(1), m.group(2)
        args = [a.strip() for a in argstr.split(",") if a.strip()]
        return {"name": name.upper(), "args": args}

    async def health(self) -> Dict[str, Any]:
        # Only HTTP has /health; use it as liveness check
        return await al_uls_client.health()

    async def eval_symbolic_call_async(self, call: Dict[str, Any]) -> Dict[str, Any]:
        name = call.get("name", "")
        args = call.get("args", [])
        if PREFER_WS:
            res = await al_uls_ws_client.eval(name, args)
            if isinstance(res, dict) and (res.get("ok") or res.get("_cached")):
                return res
        return await al_uls_client.eval(name, args)

    async def batch_eval_symbolic_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if PREFER_WS:
            res = await al_uls_ws_client.batch_eval(calls)
            if isinstance(res, list) and any(isinstance(r, dict) for r in res):
                return res
        return await al_uls_client.batch_eval(calls)


al_uls = ALULS()
