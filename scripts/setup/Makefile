# Source-tree harness helpers. Invoke with -f from the repo, or -C scripts/setup.
.DEFAULT_GOAL := help
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
VENV ?= $(ROOT)/.venv
PY ?= $(VENV)/bin/python
BOOTSTRAP_PYTHON ?= python3
KGIRL ?= kgirl
PORT ?= 8765

.PHONY: help install dev test smoke fmt lint

help:
	@printf '%s\n' 'kgirl harness engineering helpers (all actions are opt-in)' \
	  'install  create a local venv and install NumPy, Black and Ruff' \
	  'dev      launch this worktree console with a fresh temporary state directory' \
	  'test     run offline harness (including mind), helix and holo suites' \
	  'smoke    run the offline mind smoke script' \
	  'fmt      format engineering Python scripts only (120 columns)' \
	  'lint     lint engineering Python scripts only' \
	  'Override PY=/path/to/python or VENV=/path/to/venv as needed.'

install:
	"$(BOOTSTRAP_PYTHON)" -m venv "$(VENV)"
	"$(PY)" -m pip install numpy black ruff

dev:
	@state_dir=$$(mktemp -d /tmp/kgirl-console.XXXXXX) || exit 1; \
	printf 'Console scratch state: %s (retained after exit)\n' "$$state_dir"; \
	env KGIRL_CODE="$(ROOT)" KGIRL_HOME="$$state_dir" KGIRL_VENV="$(VENV)" \
	  KGIRL_CHUNKS_DB= KGIRL_MATRIX= KGIRL_ROUTER_SSH= KGIRL_LOCAL_ONLY=1 \
	  KGIRL_QINCRS_GATE=0 KGIRL_VERIFY_ALLOWLIST= "$(KGIRL)" console --port "$(PORT)"

test:
	@status=0; for suite in harness helix holo; do \
	  "$(PY)" "$(ROOT)/scripts/smoke/suite_check.py" "$$suite" || status=1; \
	done; exit "$$status"

smoke:
	"$(PY)" "$(ROOT)/scripts/smoke/mind_smoke.py"

fmt:
	cd "$(ROOT)" && "$(PY)" -m black --line-length 120 scripts/smoke

lint:
	cd "$(ROOT)" && "$(PY)" -m ruff check --line-length 120 scripts/smoke
