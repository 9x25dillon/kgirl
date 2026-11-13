"""EFL / IWME integrated validation on cascades: an isolated research package hosted in KGIRL.

Layout
    host.py          the one module that touches kgirl (cascade generators, attention memory, repaired core, ChaosRNG)
    streams.py       every random stream, by purpose, with provenance (ChaosRNG as registered, PCG64 for replication)
    models/          cascade strings, the baseline families, the attention–memory family, the EFL chain model
    metrics/         coherence, closure, description length, interventions
    experiments/     Part 1 partial observability (registered Experiment A), Part 2 layered coupling, measured data
    checks.py        Level 0 claims this package relies on, recomputed
    configs/         the registered designs and their reduced and replication variants
    results/         outputs of the registered runs
    tests/           unit tests

Nothing under src/ is changed by this package, and nothing under src/ imports it.

Run from the `research/` directory:

    python -m efl_iwme_cascade checks
    python -m efl_iwme_cascade part1 --config part1_registered --jobs 2
    python -m efl_iwme_cascade part2 --config part2_registered
    python -m unittest discover -s efl_iwme_cascade/tests -t .
"""

__version__ = "0.1.0"
