"""EFL / IWME integrated validation on cascades: an isolated research package hosted in KGIRL.

Layout
    host.py          the one module that touches kgirl (cascade generators, attention memory, repaired core, ChaosRNG)
    streams.py       every random stream, by purpose, with provenance (ChaosRNG as registered, PCG64 for replication)
    models/          cascade strings, the baseline families, the attention–memory family, the EFL chain model
    metrics/         coherence, closure, description length, interventions
    experiments/     Part 1 partial observability (registered Experiment A), Part 2 layered coupling, measured data
    checks.py        Level 0 claims this package relies on, recomputed
    limits.py        exploratory, not registered: where the registered tests stop working
    configs/         the registered designs and their reduced and replication variants
    results/         program output of the registered, replication and exploratory runs
    tests/           unit tests

Nothing under src/ is changed by this package, and nothing under src/ imports it. README.md is the
specification as built; RESULTS.md holds the verdicts, the failed replication of A5 and the limits.

Run from the `research/` directory:

    python -m efl_iwme_cascade checks
    python -m efl_iwme_cascade part1 --config part1_registered --jobs 2
    python -m efl_iwme_cascade part2 --config part2_registered
    python -m efl_iwme_cascade limits
    python -m unittest discover -s efl_iwme_cascade/tests -t .
"""

__version__ = "0.1.0"
