#!/usr/bin/env bash
# Wrapper that ensures lob_sim is importable regardless of shell VIRTUAL_ENV state.
# Usage: ./run.sh <script_or_module> [args...]
# Examples:
#   ./run.sh notebooks/experiment_runner.py
#   ./run.sh -m lob_sim.itch_parser data/raw/file.gz MSFT data/raw

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO/.venv/bin/python3"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON" "$@"
