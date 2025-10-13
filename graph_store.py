# Placeholder: compute naive degree from co-occurrence in subjects
from collections import Counter
from typing import List

def degree(subjects: List[str]) -> dict[str, float]:
    c = Counter(subjects)
    total = sum(c.values()) or 1
    return {k: v/total for k, v in c.items()}