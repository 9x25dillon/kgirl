cursor/bc-f408c7bd-bc2a-48a4-bc8d-0989f628ad52-ef2e
class MotifEngine:
    def detect_tags(self, token: str):
        token = (token or "").lower()
        tags = []
        for kw in ("sum", "mean", "var", "diff", "simplify"):
            if kw in token:
                tags.append(f"{kw.upper()}_HINT")

from __future__ import annotations
from typing import List

class MotifEngine:
    def detect_tags(self, token_text: str) -> List[str]:
        t = (token_text or "").upper()
        tags = []
        if "SUM(" in t or "MEAN(" in t or "VAR(" in t:
            tags.append("SYMBOLIC")
        if any(k in t for k in ("SELECT", "WHERE", "GROUP", "ORDER")):
            tags.append("QUERY")

        return tags

motif_engine = MotifEngine()
