"""Prompt layout: what is stable sits first, what changes sits last.

    system   STATIC: doctrine, the expert lattice, the axis rubrics, the trailer formats, the invariants.
             Byte-identical on every call of every run, so a provider's prefix cache (and Ollama's reuse of
             an evaluated prefix) keeps it.
    user     TASK       fixed for the run
             LOG        append-only flight log: blocks, accepted re-steers, rejected trailers
             STATE      the one mutable block, always the newest version, always at the tail
             TURN       who speaks now and what they owe

The vector is never written into STATIC, and a re-steer logs only the axis that changed: the full vector
exists in one place, the STATE tail. Tags are ASCII.
"""

from __future__ import annotations

from .ledger import words
from .state import AXES, INVARIANTS, RUBRICS, State

# name -> (what it is for, how it fails, the compensation it owes)
LATTICE: dict[str, tuple[str, str, str]] = {
    "ARCHIVIST": ("grounding, definitions, retrieval of known facts", "confabulating citations",
                  "hedge any source you cannot verify; never invent a reference"),
    "LOGICIAN": ("formal reasoning, proofs, edge-case enumeration", "pedantry that misses the practical point",
                 "end with what the reasoning changes in practice"),
    "ENGINEER": ("implementation, systems design, tradeoff analysis", "over-engineering",
                 "name the simplest viable version first"),
    "ORACLE": ("calibration, uncertainty, what would change the answer", "infinite hedging",
               "commit to a number, with its method and its falsifier"),
    "CRITIC": ("adversarial review, failure-mode hunting", "nihilism, or agreeing with the author",
               "every objection quotes its target and proposes a repair"),
    "SYNTHESIST": ("integration, compression, the final prose", "smoothing over unresolved conflict",
                   "list what stays unresolved instead of hiding it"),
}
GENERATORS = ("ARCHIVIST", "LOGICIAN", "ENGINEER", "ORACLE")


def _static() -> str:
    lattice = "\n".join(f"  {name:<10} {use}. Fails by: {fail}. Owes: {owe}."
                        for name, (use, fail, owe) in LATTICE.items())
    rubrics = "\n".join(f"  {axis}\n" + "\n".join(f"    {value:.1f}  {text}" for value, text in RUBRICS[axis])
                        for axis in AXES)
    return f"""\
You are one stage of HELIX, an orchestration harness. Harness code routes the stages, counts every token and
keeps the control state; you write one block when it is your turn. The message has four parts in this order:
TASK, LOG (what earlier stages wrote; history), STATE (the current control state), TURN (your instructions).

EXPERTS
{lattice}

STEERING AXES
STATE holds a vector of six values from 0.0 to 1.0. Read each value against its anchors and write the way
the nearest anchor describes; a value between two anchors sits between their behaviours.
{rubrics}

TRAILERS
A turn may ask you to end with trailer lines. Each is a tag and one JSON value on a single line. A quote
must be copied word for word, at least three words, from the text it is about. Numbers carry no plus sign.
  TUNE: {{"quote": "<span of your own block>", "axis": "<axis>", "d": <-0.25 to 0.25>, "because": "<what in the quote shows the axis is set wrong>"}}
  TUNE: none
  DEFECT: {{"quote": "<span of the draft>", "repair": "<the change that fixes it>"}}
  CONCUR: {{"evidence": "<a fact or check that is not already in the draft>"}}
  CAL: {{"p": <0 to 1>, "method": "<how you got the number>", "falsifier": "<what would prove it wrong>", "drops_below": <the value it would fall under>}}
  UNRESOLVED: [<numbers of the open conflicts you could not resolve>]

INVARIANTS
{INVARIANTS}"""


STATIC = _static()
BYPASS_SYSTEM = "Answer the question directly and briefly."


def assemble(task: str, log: list[str], state: State, turn: str) -> str:
    """The user message. Everything before STATE only ever grows by appending."""
    return (f"=== TASK ===\n{task.strip()}\n\n=== LOG ===\n" + "".join(entry + "\n\n" for entry in log) +
            f"=== STATE ===\n{state.to_json()}\n\n=== TURN ===\n{turn}")


def header(name: str, state: State, completion) -> str:
    return f"--- {name} | v{state.v} | {getattr(completion, 'backend', '?')}:{getattr(completion, 'model', '?')} ---"


def expert_turn(name: str, target: int) -> str:
    _, fail, owe = LATTICE[name]
    return (f"You are {name}. Write your block for the TASK, steered by the vector in STATE. Plain prose and "
            "lists; no heading for the block or for the trailer.\n"
            f"Aim for about {words(target)} words ({target} tokens; the harness measures the real count).\n"
            f"You tend to fail by {fail}, so: {owe}.\n"
            "Then end with exactly one TUNE trailer. Quote the weakest sentence of your own block and name the "
            "one axis whose setting caused it, or write `TUNE: none` if no sentence is weak.")


def critic_turn(target: int, conflicts: int) -> str:
    return ("You are CRITIC. Review the expert blocks in the LOG against the TASK. Do not rewrite them and do "
            "not repeat this message.\n"
            f"Reply with trailer lines only, at most {words(target)} words in total.\n"
            "Write one DEFECT trailer per defect, at most five, each quoting the faulty span of a block word for "
            "word and giving its repair. If you find no defect you may write one CONCUR trailer, but only with "
            "evidence that is not already in the LOG. A review with neither is void."
            + (f"\nSTATE already lists {conflicts} open conflict(s); do not repeat them." if conflicts else ""))


def synth_turn(target: int, conflicts: tuple) -> str:
    numbered = "\n".join(f"  {i}. {c}" for i, c in enumerate(conflicts, 1))
    return ("You are SYNTHESIST. Write the final answer to the TASK in plain prose, for a reader who will see "
            "nothing else: no tags, no expert names, nothing about this harness.\n"
            f"Aim for about {words(target)} words.\n"
            + (f"Open conflicts, numbered:\n{numbered}\nApply each repair you can. " if conflicts else "")
            + "End with one UNRESOLVED trailer listing the numbers of the conflicts you could not resolve, "
              "`UNRESOLVED: []` if none. A conflict you did not address must be listed. The trailer carries the "
              "list: do not describe the conflicts in the prose.")


def oracle_turn(axis: str, direction: str, quotes: list[str]) -> str:
    spans = "\n".join(f'  "{q}"' for q in quotes)
    return (f"You are ORACLE. Two tuning probes in a row named the same weakness: {axis} set too "
            f"{'low' if direction == '+' else 'high'}. The quoted spans:\n{spans}\n"
            "Give one CAL trailer for the claim \"the quoted spans are correct as written\": a probability, "
            "the method that produced it, what would falsify it, and the value it would then drop below. "
            "One sentence of prose before the trailer at most.")


def compact_turn(target: int) -> str:
    return ("You are SYNTHESIST, compacting. Rewrite the LOG as notes of about "
            f"{words(target)} words. Keep every figure, every decision, every accepted re-steer and every "
            "defect with its repair. Drop wording, not facts. No trailer.")
