#!/usr/bin/env bash
# Smoke-test that simulation training starts. This is not a paper-length run.
#
# Linux / macOS / Git Bash (not Windows system32\bash.exe, which launches WSL):
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --all
# On Windows cmd or Anaconda Prompt, use scripts\smoke_test.bat instead.
#
# Override the interpreter if conda is not activated:
#   PYTHON=/path/to/drones/python bash scripts/smoke_test.sh
#
# Expect several minutes: main_pyb_train.py always evaluates 10 hover
# episodes before the training loop. Logs go to log/ (gitignored).
# Paper training grids are scripts/paper_runs.py (run_paper.sh / run_paper.bat).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ALL=0
for arg in "$@"; do
  case "$arg" in
    --all) RUN_ALL=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

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

if [[ ! -d env/gym_pybullet_drones ]]; then
  echo "Missing env/gym_pybullet_drones. Run: git submodule update --init --recursive" >&2
  exit 1
fi

echo "Using PYTHON=$PYTHON"
"$PYTHON" - <<'PY'
import torch
from train.utils.env import TunableRewardAviary
print("imports ok;", "cuda" if torch.cuda.is_available() else "cpu")
print("obs", TunableRewardAviary(gui=False, record=False).observation_space.shape)
PY

SMOKE_STEPS="${SMOKE_STEPS:-8}"
SMOKE_IDX="${SMOKE_IDX:-9999}"

run_one() {
  local alg="$1"
  local env_name="$2"
  echo
  echo "=== smoke: --alg $alg --env $env_name ($SMOKE_STEPS env steps) ==="
  "$PYTHON" main_pyb_train.py \
    --alg "$alg" \
    --env "$env_name" \
    --seed 1 \
    --max_timesteps "$SMOKE_STEPS" \
    --eval_freq 1000000 \
    --exp_idx "$SMOKE_IDX" \
    --wandb_offline
}

run_one sac_predictor_IB tunable-reward

if [[ "$RUN_ALL" -eq 1 ]]; then
  run_one spederv3 tunable-reward
  run_one domain-randomization domain-randomization
fi

echo
echo "Smoke test passed."
