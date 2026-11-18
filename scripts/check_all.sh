#!/usr/bin/env bash
# Run deliberately on check day; never installs dependencies or invokes a model.
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${KGIRL_CHECK_PYTHON:-python3}" "$script_dir/smoke/check_all.py" "$@"
