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

Run scripts from the **repository root** so `import train` and `import env.gym_pybullet_drones` resolve.

Weights & Biases is imported by the training scripts. Use `--wandb_offline` on machines without internet.

## Simulation training

Paper defaults: \(B=256\), \(\gamma=0.99\), \(\tau=0.005\), `feature_dim=512`, `hidden_dim=256`, actor lr \(3\times 10^{-5}\), critic / Predictor lr \(10^{-4}\), \(\lambda=0.25\), \(\beta=1\), \(N_\mathrm{Pred}=2\) (`--extra_feature_steps 1`). Reward weights: position \(2.5\), roll/pitch \(1.5\), linear/angular velocity \(0.05\), action \(0.1\), plus a \(+2\) offset. Episodes are \(10^6\) environment steps; evaluation every \(10^4\) steps.

**Predictor**

```bash
python main_pyb_train.py --alg sac_predictor_IB --env tunable-reward \
  --seed 1 --batch_size 256 --feature_dim 512 --lam 0.25 --beta 1.0 \
  --max_timesteps 1000000 --eval_freq 10000
```

**Vanilla SAC** (and STEADY stage 1)

```bash
python main_pyb_train.py --alg spederv3 --env tunable-reward --seed 1
```

**DR** (`--alg` and `--env` must both be `domain-randomization`; evaluation uses the unrandomized hover task)

```bash
python main_pyb_train.py --alg domain-randomization --env domain-randomization --seed 1
```

Logs and checkpoints go to

```
log/<env>/<alg>/batch_<B>_featureDim_<d>_seed_<s>_lambda_<lam>_beta_<beta>_extraFeatureStep_<n>_experiment_<idx>/
```

Best-eval files: `best_actor.pth`, `best_critic.pth`, and either `best_predictor.pth` (Predictor) or `best_feature_mu.pth` (`spederv3`).

### STEADY stage 2

Complete STEADY is **two stages**. `spederv3` alone is Vanilla SAC, not STEADY.

```bash
python main_online_STEADY.py \
  --agent_path log/tunable-reward/spederv3/<run-directory> \
  --log_path path/to/user_flight.csv \
  --seed 1
```

`--agent_path` must contain `best_actor.pth`, `best_critic.pth`, and `best_feature_mu.pth` from stage 1. `--log_path` is a user-provided radio CSV (or a directory of CSVs). This repository does not ship flight logs. `--feature_dim` must match stage 1 (paper: 512). `max_timesteps` here is **gradient steps** on the offline buffer, not simulator steps.

## Hardware deployment

Waiting update.

Firmware, radio CSV schema, and onboard evaluation protocol will be documented here.

## Layout

```
main_pyb_train.py          # Predictor, Vanilla SAC / STEADY stage 1, DR
main_online_STEADY.py      # STEADY stage 2
train/agent/               # SAC, Mellinger actor, Predictor, STEADY, DR
train/math/hsic.py         # nHSIC (Ma et al. 2020)
train/networks/features.py
train/utils/env.py         # paper reward + DR wrapper
train/utils/buffer.py      # sim replay + radio CSV loader
env/                       # gym-pybullet-drones submodule
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
