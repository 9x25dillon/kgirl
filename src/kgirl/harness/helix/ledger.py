"""Ledger: token accounting done by code, from the counts the provider reports.

A model cannot count its own tokens, so nothing here asks one to. Each call is charged with two numbers:

    ext   the provider's count (`Completion.output_tokens`); None when the backend reported none
    est   a proxy from the text, with the conversion stated: max(words x 1.35, characters / 3.5)

`ext` is the truth and `est` is labelled an estimate everywhere it is shown. Their ratio is reported, so the
proxy can be judged against the deployment's own tokenizer instead of trusted.

The budget B is in generated tokens. Reserve is defined once, at L1, and split so tuning cannot starve the
answer:

    steering pool    5% of B   tuning trailers and calibration calls
    synthesis floor  10% of B  spendable only by the synthesis
    content          the rest  split across the expert blocks and the critic by weight

Targets are stated to the model, not enforced by cutting it off. When closed nodes have spent more than they
were allocated, the next node's target shrinks by that overrun ratio. Compaction has no allocation of its
own: it is spend where it happens, so it shows as overrun and compresses what follows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

STEER_SHARE = 0.05
SYNTH_SHARE = 0.10
MAX_OVERHEAD = 0.20          # the harness is not worth running when its fixed scaffold exceeds this share of B
MIN_CONTENT = 300            # nor when the answer itself is expected to be shorter than this
EXPERT, CRITIC, SYNTHESIS, STEER, COMPACTION, BYPASS = "expert", "critic", "synthesis", "steer", "compaction", "bypass"
_CONTENT = (EXPERT, CRITIC, COMPACTION)


def estimate(text: str) -> int:
    """The stated proxy: max(words x 1.35, characters / 3.5). An estimate; never shown as a measurement."""
    return int(round(max(len(text.split()) * 1.35, len(text) / 3.5)))


def words(tokens: int) -> int:
    """A token target as the word count a model can actually aim for (tokens / 1.35)."""
    return max(1, int(round(tokens / 1.35)))


def bypass(B: int, expect: int | None = None, floor: int = 500) -> str | None:
    """The BYPASS predicate. Returns the reason to answer directly, or None to run the harness.

    The scaffold floor is fixed, so its share of the budget is floor / B: bypass when that exceeds
    MAX_OVERHEAD, or when the caller expects less than MIN_CONTENT tokens of answer."""
    if B * MAX_OVERHEAD < floor:
        return (f"budget {B} is under {int(-(-floor // MAX_OVERHEAD))}: a scaffold floor of {floor} tokens "
                f"would be {floor / B:.0%} of it (limit {MAX_OVERHEAD:.0%})")
    if expect is not None and expect < MIN_CONTENT:
        return f"expected answer of {expect} tokens is under {MIN_CONTENT}"
    return None


@dataclass
class Entry:
    node: str
    kind: str
    alloc: int               # the node's share of its pool (0 for spend that has no allocation)
    target: int              # what the model was asked to aim for, after any compression
    est: int
    ext: int | None          # provider-counted generated tokens
    steer_ext: int           # the part of this call attributed to its trailers, charged to the steering pool
    in_ext: int | None       # provider-counted prompt tokens
    model: str = ""
    backend: str = ""
    stop: str = ""

    @property
    def spent(self) -> int:
        return self.ext if self.ext is not None else self.est


class Ledger:
    def __init__(self, B: int, weights: dict[str, float] | None = None):
        weights = weights or {}
        if B <= 0:
            raise ValueError("the budget B must be a positive number of tokens")
        if sum(weights.values()) > 1.0 + 1e-9:
            raise ValueError("sibling weights must sum to at most 1.0")
        self.B = int(B)
        self.steer_pool = int(round(self.B * STEER_SHARE))
        self.synth_floor = int(round(self.B * SYNTH_SHARE))
        self.content_pool = self.B - self.steer_pool - self.synth_floor
        self.alloc = {node: int(round(self.content_pool * w)) for node, w in weights.items()}
        self.entries: list[Entry] = []

    # ------------------------------------------------------------------------------------------- pools

    def _content_spent(self) -> int:
        return sum(e.spent - e.steer_ext for e in self.entries if e.kind in _CONTENT)

    def steer_spent(self) -> int:
        return sum(e.spent if e.kind == STEER else e.steer_ext for e in self.entries)

    def steer_left(self) -> int:
        return self.steer_pool - self.steer_spent()

    def ratio(self) -> float:
        """Allocated over spent across the content nodes closed so far; 1.0 while they are within budget."""
        spent = self._content_spent()
        alloc = sum(e.alloc for e in self.entries if e.kind in _CONTENT)
        return alloc / spent if spent > alloc > 0 else 1.0

    def target(self, node: str) -> int:
        return max(1, int(round(self.alloc.get(node, 0) * self.ratio())))

    def synth_target(self) -> int:
        return self.synth_floor + max(0, self.content_pool - self._content_spent())

    # ------------------------------------------------------------------------------------------- charging

    def charge(self, node: str, kind: str, text: str, completion, target: int, trailer_chars: int = 0) -> Entry:
        """Record one call. `completion` is the backend's Completion; a zero token count beside non-empty text
        means the backend did not count, and is kept as None rather than as a measured zero.

        `trailer_chars` is how much of `text` was trailer lines: that share of the call is attributed to the
        steering pool (an attribution of a measured total, by characters)."""
        est = estimate(text)
        out = getattr(completion, "output_tokens", 0) or 0
        inp = getattr(completion, "input_tokens", 0) or 0
        ext = int(out) if out > 0 or not text else None
        spent = ext if ext is not None else est
        steer = int(round(spent * trailer_chars / len(text))) if text and trailer_chars and kind != STEER else 0
        alloc = self.alloc.get(node, 0) if kind in (EXPERT, CRITIC) else 0
        if kind == SYNTHESIS:
            alloc = self.synth_floor
        entry = Entry(node, kind, alloc, int(target), est, ext, steer, int(inp) if inp > 0 else None,
                      getattr(completion, "model", ""), getattr(completion, "backend", ""),
                      getattr(completion, "stop_reason", ""))
        self.entries.append(entry)
        return entry

    # ------------------------------------------------------------------------------------------- report

    def report(self) -> dict:
        uncounted = sum(1 for e in self.entries if e.ext is None)
        spent_ext = sum(e.ext for e in self.entries if e.ext is not None) if not uncounted else None
        spent_est = sum(e.est for e in self.entries)
        spent = sum(e.spent for e in self.entries)
        inputs = [e.in_ext for e in self.entries]
        answer = sum(e.spent - e.steer_ext for e in self.entries if e.kind in (SYNTHESIS, BYPASS))
        scaffold = self.steer_spent() + sum(e.spent - e.steer_ext for e in self.entries if e.kind in (CRITIC, COMPACTION))
        counted = [e for e in self.entries if e.ext]
        return {
            "B": self.B,
            "pools": {"content": self.content_pool, "steer": self.steer_pool, "synth_floor": self.synth_floor},
            "nodes": [asdict(e) for e in self.entries],
            "spent_est": spent_est,
            "spent_ext": spent_ext,                       # null unless every call was counted by its provider
            "uncounted_calls": uncounted,
            "input_ext": sum(inputs) if inputs and all(i is not None for i in inputs) else None,
            "steer_spent": self.steer_spent(),
            "scaffold_ext": scaffold if not uncounted else None,
            "overrun": spent > self.B,
            "efficiency": round(answer / spent, 3) if spent else None,     # answer tokens per generated token
            "overhead": round(scaffold / spent, 3) if spent else None,
            "est_over_ext": round(sum(e.est for e in counted) / sum(e.ext for e in counted), 3) if counted else None,
        }
