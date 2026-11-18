#!/usr/bin/env bash
# Review outgoing history locally. This script never pushes or fetches.
set -euo pipefail
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${KGIRL_CHECK_PYTHON:-python3}" "$script_dir/smoke/prepush_scan.py" "$@"
