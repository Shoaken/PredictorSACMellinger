"""Launch paper simulation jobs (or one custom run).

Edit the PAPER dict below for defaults. Override per call with flags;
you do not need to edit main_pyb_train.py.

Examples (from the repository root, drones env active)::

    python scripts/paper_runs.py --list --grid paper
    python scripts/paper_runs.py --one --seed 1 --lam 0.25 --beta 1
    python scripts/paper_runs.py --one --alg spederv3 --seed 42
    python scripts/paper_runs.py --grid comparison --wandb-offline
    python scripts/paper_runs.py --grid paper --dry-run

Windows::

    scripts\\run_paper.bat --one --lam 0.5 --seed 1

Git Bash / Linux::

    bash scripts/run_paper.sh --grid lambda

STEADY stage 2 is not in these grids (needs a user radio CSV).
Use main_online_STEADY.py after spederv3.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Paper defaults. Change these to retune every job, or pass CLI flags.
# ---------------------------------------------------------------------------
PAPER = {
    "max_timesteps": 1_000_000,
    "eval_freq": 10_000,
    "feature_dim": 512,
    "hidden_dim": 256,
    "extra_feature_steps": 1,  # N_Pred = extra_feature_steps + 1 = 2
    "exp_idx": 0,
    "seeds": (1, 42, 123, 456, 789),
    "w_pos": 2.5,
    "w_rpy": 1.5,
    "w_lin_vel": 0.05,
    "w_ang_vel": 0.05,
    "w_action": 0.1,
}

GRIDS = ("comparison", "lambda", "beta", "batch", "ablation", "paper")


@dataclass(frozen=True)
class Job:
    name: str
    alg: str
    env: str
    seed: int
    batch_size: int
    lam: float
    beta: float

    @property
    def key(self) -> tuple:
        return (self.alg, self.env, self.seed, self.batch_size, self.lam, self.beta)


def _predictor(seed: int, batch_size: int, lam: float, beta: float) -> Job:
    return Job(
        name="Predictor",
        alg="sac_predictor_IB",
        env="tunable-reward",
        seed=seed,
        batch_size=batch_size,
        lam=lam,
        beta=beta,
    )


def _vanilla(seed: int) -> Job:
    return Job(
        name="Vanilla SAC",
        alg="spederv3",
        env="tunable-reward",
        seed=seed,
        batch_size=256,
        lam=0.25,
        beta=1.0,
    )


def _dr(seed: int) -> Job:
    return Job(
        name="DR",
        alg="domain-randomization",
        env="domain-randomization",
        seed=seed,
        batch_size=256,
        lam=0.25,
        beta=1.0,
    )


def grid_jobs(grid: str, seeds: Sequence[int]) -> list[Job]:
    """Paper sim-to-sim training table (constrained / published setting)."""
    seeds = tuple(seeds)

    def pred_main() -> list[Job]:
        return [_predictor(s, 256, 0.25, 1.0) for s in seeds]

    comparison = pred_main() + [_vanilla(s) for s in seeds] + [_dr(s) for s in seeds]
    lambda_sweep = [
        _predictor(s, 256, lam, 1.0) for lam in (0.10, 0.25, 0.50) for s in seeds
    ]
    beta_sweep = [
        _predictor(s, 256, 0.25, beta) for beta in (1.0, 2.0, 4.0) for s in seeds
    ]
    batch_sweep = [
        _predictor(s, batch, 0.25, 1.0) for batch in (64, 128, 256) for s in seeds
    ]
    ablation = pred_main() + [
        _predictor(s, 256, 0.0, 1.0) for s in seeds
    ] + [
        _predictor(s, 256, 0.25, 0.0) for s in seeds
    ]

    tables = {
        "comparison": comparison,
        "lambda": lambda_sweep,
        "beta": beta_sweep,
        "batch": batch_sweep,
        "ablation": ablation,
    }
    if grid == "paper":
        seen: set[tuple] = set()
        merged: list[Job] = []
        for job in comparison + lambda_sweep + beta_sweep + batch_sweep + ablation:
            if job.key in seen:
                continue
            seen.add(job.key)
            merged.append(job)
        return merged
    return tables[grid]


def job_argv(
    job: Job,
    *,
    max_timesteps: int,
    eval_freq: int,
    feature_dim: int,
    hidden_dim: int,
    extra_feature_steps: int,
    exp_idx: int,
    wandb_offline: bool,
    w_pos: float,
    w_rpy: float,
    w_lin_vel: float,
    w_ang_vel: float,
    w_action: float,
    device: str | None,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "main_pyb_train.py"),
        "--alg",
        job.alg,
        "--env",
        job.env,
        "--seed",
        str(job.seed),
        "--batch_size",
        str(job.batch_size),
        "--lam",
        str(job.lam),
        "--beta",
        str(job.beta),
        "--max_timesteps",
        str(max_timesteps),
        "--eval_freq",
        str(eval_freq),
        "--feature_dim",
        str(feature_dim),
        "--hidden_dim",
        str(hidden_dim),
        "--extra_feature_steps",
        str(extra_feature_steps),
        "--exp_idx",
        str(exp_idx),
        "--w_pos",
        str(w_pos),
        "--w_rpy",
        str(w_rpy),
        "--w_lin_vel",
        str(w_lin_vel),
        "--w_ang_vel",
        str(w_ang_vel),
        "--w_action",
        str(w_action),
    ]
    if device:
        argv.extend(["--device", device])
    if wandb_offline:
        argv.append("--wandb_offline")
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper simulation jobs. Edit PAPER in this file or pass flags."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--one",
        action="store_true",
        help="Single run. Override any hyperparameter below.",
    )
    mode.add_argument(
        "--grid",
        choices=GRIDS,
        help="Named paper table: comparison, lambda, beta, batch, ablation, or all unique jobs (paper).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print jobs and exit (use with --grid or --one).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching training.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue the grid after a failed job.",
    )
    parser.add_argument("--from-index", type=int, default=1, help="1-based job index to start from.")

    parser.add_argument("--alg", choices=["sac_predictor_IB", "spederv3", "domain-randomization"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", help="Replace the paper seed list for --grid.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--extra-feature-steps", type=int)
    parser.add_argument("--exp-idx", type=int)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--w-pos", type=float)
    parser.add_argument("--w-rpy", type=float)
    parser.add_argument("--w-lin-vel", type=float)
    parser.add_argument("--w-ang-vel", type=float)
    parser.add_argument("--w-action", type=float)
    return parser.parse_args()


def common_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        max_timesteps=args.max_timesteps if args.max_timesteps is not None else PAPER["max_timesteps"],
        eval_freq=args.eval_freq if args.eval_freq is not None else PAPER["eval_freq"],
        feature_dim=args.feature_dim if args.feature_dim is not None else PAPER["feature_dim"],
        hidden_dim=args.hidden_dim if args.hidden_dim is not None else PAPER["hidden_dim"],
        extra_feature_steps=(
            args.extra_feature_steps
            if args.extra_feature_steps is not None
            else PAPER["extra_feature_steps"]
        ),
        exp_idx=args.exp_idx if args.exp_idx is not None else PAPER["exp_idx"],
        wandb_offline=args.wandb_offline,
        w_pos=args.w_pos if args.w_pos is not None else PAPER["w_pos"],
        w_rpy=args.w_rpy if args.w_rpy is not None else PAPER["w_rpy"],
        w_lin_vel=args.w_lin_vel if args.w_lin_vel is not None else PAPER["w_lin_vel"],
        w_ang_vel=args.w_ang_vel if args.w_ang_vel is not None else PAPER["w_ang_vel"],
        w_action=args.w_action if args.w_action is not None else PAPER["w_action"],
        device=args.device,
    )


def one_job(args: argparse.Namespace) -> Job:
    alg = args.alg or "sac_predictor_IB"
    if alg == "domain-randomization":
        env = "domain-randomization"
    else:
        env = "tunable-reward"
    return Job(
        name={"sac_predictor_IB": "Predictor", "spederv3": "Vanilla SAC"}.get(alg, "DR"),
        alg=alg,
        env=env,
        seed=args.seed if args.seed is not None else 1,
        batch_size=args.batch_size if args.batch_size is not None else 256,
        lam=args.lam if args.lam is not None else 0.25,
        beta=args.beta if args.beta is not None else 1.0,
    )


def print_jobs(jobs: Sequence[Job]) -> None:
    print(f"{len(jobs)} job(s):")
    for i, job in enumerate(jobs, start=1):
        print(
            f"  {i:3d}  {job.name:<12} alg={job.alg:<24} "
            f"seed={job.seed:<4} B={job.batch_size:<4} "
            f"lam={job.lam:<5} beta={job.beta}"
        )


def run_jobs(jobs: Sequence[Job], args: argparse.Namespace) -> int:
    shared = common_kwargs(args)
    start = max(args.from_index, 1)
    if start > len(jobs):
        print(f"--from-index {start} is past {len(jobs)} jobs.", file=sys.stderr)
        return 2

    failed = 0
    for i, job in enumerate(jobs, start=1):
        if i < start:
            continue
        argv = job_argv(job, **shared)
        print(f"\n=== [{i}/{len(jobs)}] {job.name}  seed={job.seed} B={job.batch_size} "
              f"lam={job.lam} beta={job.beta} ===")
        print(" ".join(argv))
        if args.dry_run:
            continue
        completed = subprocess.run(argv, cwd=REPO_ROOT)
        if completed.returncode != 0:
            failed += 1
            print(f"Job {i} failed with code {completed.returncode}", file=sys.stderr)
            if not args.keep_going:
                return completed.returncode
    if args.dry_run:
        print("\nDry run only; nothing launched.")
    return 1 if failed else 0


def main() -> int:
    if not (REPO_ROOT / "env" / "gym_pybullet_drones").is_dir():
        print(
            "Missing env/gym_pybullet_drones. Run: git submodule update --init --recursive",
            file=sys.stderr,
        )
        return 1

    args = parse_args()
    seeds = tuple(args.seeds) if args.seeds else PAPER["seeds"]
    if args.one:
        jobs = [one_job(args)]
    else:
        jobs = grid_jobs(args.grid, seeds)

    print_jobs(jobs)
    if args.list:
        return 0
    return run_jobs(jobs, args)


if __name__ == "__main__":
    raise SystemExit(main())
