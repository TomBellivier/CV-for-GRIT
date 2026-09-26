#!/usr/bin/env bash
# =============================================================================
# run_macos.sh - run the whole project on macOS
#
# Settings: run_config.yaml (repository root). Every option is passed on to
# run_all.py, for instance:
#     bash run_macos.sh                          run every step switched on
#     bash run_macos.sh --dry-run                print the commands, run nothing
#     bash run_macos.sh --only pose analysis     run these steps only
#     bash run_macos.sh --config my_run.yaml     another settings file
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# The virtual environment of the README, else the Python of the PATH (python3 on macOS).
if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# Apple GPU (mps): the operations PyTorch does not implement on it yet run on the CPU
# instead of stopping the training.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# No window for matplotlib: every figure is written to a file.
export MPLBACKEND=Agg

exec "$PYTHON" run_all.py --config run_config.yaml "$@"
