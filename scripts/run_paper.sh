#!/usr/bin/env bash
# Pass-through to scripts/paper_runs.py (paper grids and one-off runs).
#   bash scripts/run_paper.sh --one --seed 1 --lam 0.25
#   bash scripts/run_paper.sh --grid comparison --list
#   bash scripts/run_paper.sh --grid paper --wandb-offline
# On Windows cmd / Anaconda Prompt, use scripts\run_paper.bat instead.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ -x "${CONDA_PREFIX}/python.exe" ]]; then
      PYTHON="${CONDA_PREFIX}/python.exe"
    elif [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
      PYTHON="${CONDA_PREFIX}/bin/python"
    fi
  fi
fi
PYTHON="${PYTHON:-python}"

exec "$PYTHON" scripts/paper_runs.py "$@"
