#!/usr/bin/env bash
# =============================================================================
# run_linux.sh - run the whole project on Linux
#
# Settings: run_config.yaml (repository root). Every option is passed on to
# run_all.py, for instance:
#     bash run_linux.sh                          run every step switched on
#     bash run_linux.sh --dry-run                print the commands, run nothing
#     bash run_linux.sh --only pose analysis     run these steps only
#     bash run_linux.sh --config my_run.yaml     another settings file
# To keep a long run alive after closing the terminal:
#     nohup bash run_linux.sh > run.out 2>&1 &
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# The virtual environment of the README, else the Python of the PATH.
if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# Servers usually have no display: matplotlib must not look for one.
export MPLBACKEND=Agg

exec "$PYTHON" run_all.py --config run_config.yaml "$@"
