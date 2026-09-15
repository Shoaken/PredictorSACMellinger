"""
STEADY stage 2: residual-skill transfer on a Crazyflie radio CSV.

Adapted from https://github.com/mahaitongdae/steady_sim_to_real/tree/published
(Ma et al., arXiv:2404.05051). This script is the paper STEADY baseline,
not Predictor.

Pipeline
--------
1. Train simulator skills::

       python main_pyb_train.py --alg spederv3 --env tunable-reward --seed 1

   Saves ``best_actor.pth``, ``best_critic.pth``, ``best_feature_mu.pth``.
   That actor *is* Vanilla SAC; ``feature_mu`` is only for this stage.

2. Provide a radio/ground-station CSV (see RealDataBuffer.CSV_REQUIRED_FIELDS).
   ``ctrlMel_pos_error_*`` / ``ctrlMel_i_err_m*`` may be omitted; they are
   reconstructed. Do not use SD-card USD logs.

3. Run this script::

       python main_online_STEADY.py \\
           --agent_path log/tunable-reward/spederv3/<run> \\
           --log_path path/to/flight.csv --seed 1

What this stage does
--------------------
Freeze phi^circ (critic feature), mu^circ, pi^circ. Learn residual phi, mu
with Hilbert-Schmidt skill loss plus orthogonality to phi^circ. Improve
the Mellinger PID with Q = w^T [phi^circ, phi] and KL(pi || pi^circ).
The log_std trunk is frozen so only firmware-like gains move.

Defaults: 28-D state, 4-D PWM, feature_dim must match stage 1 (512),
aug_feature_dim=128 (TransferAgent default), extra_feature_steps=3,
tau_kl=0.1, discovery_lambda=1.0, max_timesteps=1e3 gradient steps
(offline; there is no simulator roll-out here).
"""
import numpy as np
import torch
import argparse
import os
import pickle as pkl

import wandb
from gymnasium.spaces import Box

from train.utils import util, buffer
from train.agent.feature_sac import feature_sac_agent

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="STEADY stage 2: residual skills from a radio CSV (requires spederv3 checkpoints)."
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--agent_path", required=True,
                        help="Stage-1 directory with best_actor.pth, best_critic.pth, best_feature_mu.pth")
    parser.add_argument("--log_path", required=True,
                        help="One CSV file or a directory of Crazyflie radio CSVs (not USD)")
    parser.add_argument("--output_dir", default=os.path.join("log", "transfer", "steady"),
                        help="Directory for online-stage outputs")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--max_timesteps", default=1e3, type=float,
                        help="Gradient steps on the offline CSV buffer (not environment steps).")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--feature_dim", default=512, type=int,
                        help="Must match stage-1 --feature_dim (paper 512)")
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--extra_feature_steps", default=3, type=int,
                        help="Residual phi/mu updates per critic/actor step (N = extra_feature_steps + 1).")
    parser.add_argument("--wandb_offline", action="store_true",
                        help="Run wandb in offline mode (useful for HPC compute nodes without internet)")
    args = parser.parse_args()

    exp_name = f"steady_online_seed_{args.seed}"
    log_path = args.output_dir
    os.makedirs(log_path, exist_ok=True)

    wandb_mode = "offline" if args.wandb_offline else "online"
    wandb.init(
        project="Predictor_quadrotor_transfer",
        name=exp_name,
        config=vars(args),
        dir=log_path,
        mode=wandb_mode,
    )

    kwargs_args = vars(args)
    with open(os.path.join(log_path, 'train_params.pkl'), 'wb') as fp:
        pkl.dump(kwargs_args, fp)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Observation/action match HoverAviary + Mellinger PWM (see mellinger_pybullet.py).
    kwargs = {
        "state_dim": 28,
        "action_dim": 4,
        "action_space": Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
        "feature_dim": args.feature_dim,
        "extra_feature_steps": args.extra_feature_steps,
        "device": args.device,
        "agent_path": args.agent_path,
    }

    agent = feature_sac_agent.TransferAgent(**kwargs)

    replay_buffer = buffer.RealDataBuffer(device=args.device)
    replay_buffer.load_all_data(args.log_path)
    print(f'Loaded {replay_buffer.size} real transitions from {args.log_path}')
    xyz = replay_buffer.state[:, 0:3]
    print(f'obs xyz mean={xyz.mean(0)}, std={xyz.std(0)}')
    with torch.no_grad():
        sample = replay_buffer.sample(min(256, replay_buffer.size))
        control = agent.actor.mellinger_control(sample.state)
        min_pwm = agent.actor.MIN_PWM / agent.actor.MAX_PWM
        clip_frac = ((control <= min_pwm + 1e-4) | (control >= 1.0 - 1e-4)).float().mean().item()
        pos_e_z = (agent.actor.goal.to(sample.state.device) - sample.state[:, 0:3])[:, 2]
        print(f'Mellinger pwm mean={control.mean().item():.3f} clip_frac={clip_frac:.3f} '
              f'pos_e_z mean={pos_e_z.mean().item():.3f}')
        if clip_frac > 0.2:
            print('WARNING: PWM is saturating; PID gradients will be ~0. Check that obs[0:3] is world xyz near (0,0,1).')
    timer = util.Timer()

    for t in range(int(args.max_timesteps)):
        info = agent.train(replay_buffer, batch_size=args.batch_size)

        steps_per_sec = timer.steps_per_sec(t + 1)
        if (t + 1) % 50 == 0:
            wandb.log({
                **{f"train/{k}": v for k, v in info.items()},
                "train/steps_per_sec": steps_per_sec,
            }, step=t + 1)

        if t >= int(args.max_timesteps) - 5:
            torch.save(agent.actor.state_dict(), os.path.join(log_path, f'terminal_actor_{t}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(log_path, f'terminal_critic_{t}.pth'))
            torch.save(agent.feature_mu.state_dict(), os.path.join(log_path, f'terminal_mu_{t}.pth'))

        print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

    wandb.finish()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(agent.actor.state_dict(), os.path.join(log_path, 'last_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(log_path, 'last_critic.pth'))
    torch.save(agent.feature_mu.state_dict(), os.path.join(log_path, 'last_feature_mu.pth'))
