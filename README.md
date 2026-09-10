# Predictor-Aligned SAC for Quadrotor Sim-to-Real Transfer

Code for **Predictor-Aligned Soft Actor–Critic for Zero-Shot Quadrotor Sim-to-Real Transfer**.

Xuankun Cai, Ahmed Hamidalddin, Andrea Lecchini-Visintin, Chris Freeman, and Matthew C. Turner. School of Electronics and Computer Science, University of Southampton, SO17 1BJ, UK.

The policy mean is a differentiable Crazyflie Mellinger / DSLPID controller. SAC learns the PID gains (with a feasible-gain projection) and a squashed-Gaussian scale. Training is in PyBullet (`HoverAviary`, Crazyflie CF2X, 240 Hz, 5 s hover).

## Paper methods vs this repository

| Paper name | Command | Agent |
|---|---|---|
| **Predictor** (nHSIC-IB) | `--alg sac_predictor_IB --env tunable-reward` | `SPEDERAgent_PredictorIB` |
| **Vanilla SAC** | `--alg spederv3 --env tunable-reward` | `SPEDERAgentV3Mel` |
| **Domain randomization (DR)** | `--alg domain-randomization --env domain-randomization` | `SPEDERAgentDomainRandomization` |
| **STEADY** (full) | stage 1: `--alg spederv3`, then `main_online_STEADY.py` | `TransferAgent` |

`--alg sac` is **not** a paper experiment. Vanilla SAC in the tables is `spederv3`: the same Mellinger actor as Predictor. Its auxiliary network `feature_mu` does **not** train the actor; it only exists so STEADY stage 2 can freeze simulator skills.

Predictor trains an nHSIC information bottleneck \(Z=\varphi(s,a)\) and aligns the critic feature \(\varphi_Q\) to \(Z\). The nHSIC estimator is the ridge/CCA form from Ma et al., *HSIC Bottleneck* (AAAI 2020), not a new dependence measure.

## Installation

```bash
git clone --recurse-submodules https://github.com/Shoaken/PredictorSACMellinger.git
cd PredictorSACMellinger
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

The simulator is the git submodule `env/` ([mahaitongdae/gym-pybullet-drones](https://github.com/mahaitongdae/gym-pybullet-drones.git)), pinned to commit `7856f34`. Do not edit files under `env/gym_pybullet_drones/`. Paper wrappers live in `train/utils/env.py`.

Create the conda environment from the export used for the paper runs (`drones`, Python 3.10, PyTorch 2.5.1, CUDA 12.4):

```bash
conda env create -f environment.yml
conda activate drones
pip install -e env
```

`environment.yml` was exported on **Windows** with `--no-builds`. On Linux or CPU-only machines, create a Python 3.10 env with matching `pytorch`, `gymnasium`, `pybullet`, `numpy`, `scipy`, `pandas`, and `wandb`, then `pip install -e env`.

Run commands from the **repository root** so `import train` and `import env.gym_pybullet_drones` resolve.

Weights & Biases is imported by the training scripts. Use `--wandb-offline` (wrappers) or `--wandb_offline` (`main_pyb_train.py`) on machines without internet.

### Check the environment

`scripts/smoke_test` runs a few environment steps so you can confirm PyBullet, the Mellinger actor, and CUDA/CPU import. It is **not** a paper-length run (the training loop still starts with 10 eval episodes).

Activate `drones` first. On Windows Anaconda Prompt / cmd, `bash` is often WSL and may fail — use the `.bat` file:

```bat
scripts\smoke_test.bat
scripts\smoke_test.bat --all
```

On Linux, macOS, or Git Bash:

```bash
bash scripts/smoke_test.sh
bash scripts/smoke_test.sh --all
```

Git Bash does not load conda by default. Either source Anaconda's `etc/profile.d/conda.sh` and `conda activate drones`, or set `PYTHON` to the `drones` interpreter:

```bash
PYTHON=/path/to/drones/python bash scripts/smoke_test.sh
```

## Simulation training

Paper jobs are launched from `scripts/paper_runs.py`. Defaults are the `PAPER` dict at the top of that file: \(B=256\), `feature_dim=512`, `hidden_dim=256`, \(\lambda=0.25\), \(\beta=1\), \(N_\mathrm{Pred}=2\) (`extra_feature_steps=1`), \(10^6\) environment steps, evaluation every \(10^4\) steps, seeds `(1, 42, 123, 456, 789)`, reward weights position \(2.5\), roll/pitch \(1.5\), linear/angular velocity \(0.05\), action \(0.1\). Actor lr \(3\times 10^{-5}\) and critic / Predictor lr \(10^{-4}\) stay in `main_pyb_train.py` / the agents. Edit `PAPER`, or pass flags; you do not need to change `main_pyb_train.py` for the usual sweeps.

Wrappers (same flags as `paper_runs.py`):

```bat
scripts\run_paper.bat --one --seed 1
```

```bash
bash scripts/run_paper.sh --one --seed 1
```

Or call Python directly: `python scripts/paper_runs.py ...`.

**One run.** Predictor unless you set `--alg`. Override any hyperparameter on the command line (`--lam`, `--beta`, `--batch-size`, `--seed`, `--max-timesteps`, …):

```bash
python scripts/paper_runs.py --one --seed 1 --lam 0.25 --beta 1 --batch-size 256
python scripts/paper_runs.py --one --alg spederv3 --seed 1
python scripts/paper_runs.py --one --alg domain-randomization --seed 1 --wandb-offline
```

**Paper tables.** `--grid paper` is the unique union (55 jobs at the paper step budget). DR pairs `--alg` and `--env` as `domain-randomization`.

| `--grid` | What it runs |
|---|---|
| `comparison` | Predictor \(\lambda=0.25,\beta=1\), Vanilla SAC, DR (5 seeds) |
| `lambda` | Predictor \(\lambda\in\{0.10,0.25,0.50\}\), \(\beta=1\), \(B=256\) |
| `beta` | Predictor \(\beta\in\{1,2,4\}\), \(\lambda=0.25\), \(B=256\) |
| `batch` | Predictor \(B\in\{64,128,256\}\), \(\lambda=0.25\), \(\beta=1\) |
| `ablation` | \((\lambda,\beta)=(0.25,1)\), \((0,1)\), \((0.25,0)\) |
| `paper` | Unique union of the rows above |

```bash
python scripts/paper_runs.py --list --grid comparison
python scripts/paper_runs.py --grid comparison --dry-run
python scripts/paper_runs.py --grid paper --wandb-offline
python scripts/paper_runs.py --grid lambda --from-index 6 --keep-going
```

`--list` prints jobs and exits. `--dry-run` prints the `main_pyb_train.py` commands. `--from-index N` resumes a grid (1-based). `--seeds 1 42` replaces the paper seed list. `--keep-going` continues after a failed job. `main_pyb_train.py` still works if you prefer to call it directly.

Logs and checkpoints go to

```
log/<env>/<alg>/batch_<B>_featureDim_<d>_seed_<s>_lambda_<lam>_beta_<beta>_extraFeatureStep_<n>_experiment_<idx>/
```

Best-eval files: `best_actor.pth`, `best_critic.pth`, and either `best_predictor.pth` (Predictor) or `best_feature_mu.pth` (`spederv3`).

### STEADY stage 2

Complete STEADY is **two stages**. `spederv3` (`--grid comparison` Vanilla SAC jobs, or `--one --alg spederv3`) is stage 1 only. Stage 2 is not in `paper_runs.py` because it needs a user radio CSV.

```bash
python main_online_STEADY.py \
  --agent_path log/tunable-reward/spederv3/<run-directory> \
  --log_path path/to/user_flight.csv \
  --seed 1
```

`--agent_path` must contain `best_actor.pth`, `best_critic.pth`, and `best_feature_mu.pth` from stage 1. `--log_path` is a user-provided radio CSV (or a directory of CSVs). This repository does not ship flight logs. `--feature_dim` must match stage 1 (paper: 512). `max_timesteps` here is **gradient steps** on the offline buffer, not simulator steps.

## MATLAB Simulink evaluation (`deploy/sim_to_sim`)

Requires MATLAB with Simulink. In MATLAB, set the current folder to `deploy/sim_to_sim/` so `parameter/` and `quadrotor_mellinger.slx` resolve.

Typical pipeline: export gains from a trained actor, optionally inspect one task, write `.mat` files, then batch-evaluate eight tasks.

1. **Export gains** with `scripts/pth_reader.py` from `best_actor.pth` / `last_actor.pth` (under the `log/...` run directory):

   ```bash
   python scripts/pth_reader.py --pth path/to/best_actor.pth
   ```

   Copy the printed `Kp_lin` … `Ki_rot` block. Replace `path/to/best_actor.pth` with your local checkpoint. The actor stores PID values directly; the reader does not decode old sigmoid-`_raw` checkpoints.

2. **`param_set.m`** — paste or uncomment one gain set and set `init_pos` / `goal_pos` (and optional sensor-bias / pulse flags). Then run `quadrotor_mellinger.slx` in Simulink to inspect **that** controller on **that** task. This script does not load `.mat` files and does not report the eight-task scores.

3. **`generate_mat_files.m`** — paste the same MATLAB blocks into `raw_data` (a `% FileName` comment, then the six gain lines). Running it writes `parameter/<FileName>.mat` for `main2.m` to load. The repository ships `parameter/` empty; generate the files locally.

4. **`main2.m`** — set `param_file` to a stem in `parameter/` (without `.mat`). It loads those controller gains and runs **eight** Simulink tasks (hover, takeoff, 3-D step, pulse; each with and without sensor bias) and prints task-success and convergence for each.

## Plotting (`scripts/`)

Pass **your** paths; nothing is hardcoded to a machine.

```bash
# Mellinger gains from an actor checkpoint (for param_set.m / generate_mat_files.m)
python scripts/pth_reader.py --pth path/to/best_actor.pth

# Sim-to-sim bar charts: fill RESULTS in plot_validation.py from main2.m, then
python scripts/plot_validation.py --output-dir path/to/output_figures

# Evaluation curves from W&B-exported evaluation CSVs
python scripts/plot_evaluation.py --root path/to/wandb_evaluation_csvs --param beta

# Training-reward curves from W&B-exported reward CSVs
python scripts/plot_smooth.py --data-dir path/to/wandb_reward_csvs --param batch
```

## Hardware deployment

Waiting update.

Firmware, radio CSV schema, and onboard evaluation protocol will be documented here.

## Layout

```
main_pyb_train.py             # Predictor, Vanilla SAC / STEADY stage 1, DR
main_online_STEADY.py         # STEADY stage 2
train/agent/                  # SAC, Mellinger actor, Predictor, STEADY, DR
train/math/hsic.py            # nHSIC (Ma et al. 2020)
train/networks/features.py
train/utils/env.py            # paper reward + DR wrapper
train/utils/buffer.py         # sim replay + radio CSV loader
deploy/sim_to_sim/            # MATLAB/Simulink 8-task evaluation
scripts/paper_runs.py         # paper grids and one-off training
scripts/run_paper.sh|.bat     # wrappers for paper_runs.py
scripts/smoke_test.sh|.bat    # environment check (not a paper run)
scripts/pth_reader.py         # export PID gains from a .pth
scripts/plot_validation.py    # sim-to-sim success/convergence bars
scripts/plot_evaluation.py    # W&B evaluation CSVs
scripts/plot_smooth.py        # W&B reward CSVs
env/                          # gym-pybullet-drones submodule
environment.yml
```

## License

This repository is released under the [MIT License](LICENSE). Third-party code retains its original notices (see Acknowledgements).

## Acknowledgements

This repository adapts third-party code. The Predictor method and remaining training stack are ours; the following implementations are modified from the cited sources.

- **STEADY baseline.** Skill discovery, residual features, and the two-stage transfer loop in `train/agent/feature_sac/feature_sac_agent.py` (`SPEDERAgentV3Mel`, `TransferAgent`) and `main_online_STEADY.py` are adapted from [mahaitongdae/steady_sim_to_real](https://github.com/mahaitongdae/steady_sim_to_real/tree/published) (`published` branch), the official code for Ma et al., “Skill Transfer and Discovery for Sim-to-Real Learning: A Representation-Based Viewpoint,” [arXiv:2404.05051](https://arxiv.org/abs/2404.05051).
- **nHSIC estimator.** `train/math/hsic.py` (`hsic_normalized_cca_fast` and the median-heuristic RBF kernel) is adapted from [choasma/HSIC-bottleneck](https://github.com/choasma/HSIC-bottleneck) (MIT), the official code for Ma, Lewis, and Kleijn, “The HSIC Bottleneck: Deep Learning without Back-Propagation,” AAAI 2020.
- **Simulator.** [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) (MIT); this repo vendors the [mahaitongdae](https://github.com/mahaitongdae/gym-pybullet-drones) fork as the `env/` submodule.
- **SAC backbone.** Actor/critic update code is adapted from [pytorch_sac](https://github.com/denisyarats/pytorch_sac).
