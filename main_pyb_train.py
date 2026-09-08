"""Simulation training for the paper methods.

Paper: Predictor-Aligned Soft Actor-Critic for Zero-Shot Quadrotor
Sim-to-Real Transfer.

Paper name                  CLI
--------------------------  -----------------------------------------------
Predictor (nHSIC-IB)        --alg sac_predictor_IB --env tunable-reward
Vanilla SAC                 --alg spederv3         --env tunable-reward
Domain randomization (DR)   --alg domain-randomization --env domain-randomization

STEADY is two stages. This script is only stage 1 when --alg spederv3.
Stage 2 is main_online_STEADY.py (load best_*.pth, then train on a radio CSV).

Default hyperparameters match the paper tables (CF2X hover, 240 Hz, 5 s):
    B=256, gamma=0.99, tau=0.005, feature_dim=512, hidden_dim=256,
    actor lr=3e-5, critic/Predictor lr=1e-4, lambda=0.25, beta=1.
    extra_feature_steps=1 runs N_Pred = extra_feature_steps + 1 = 2
    Predictor (or STEADY-mu) updates per SAC step.

Example::

    python main_pyb_train.py --alg sac_predictor_IB --env tunable-reward \\
        --seed 1 --batch_size 256 --feature_dim 512 --lam 0.25 --beta 1.0
"""
import numpy as np
import torch
import argparse
import os
import pickle as pkl
import gc

import wandb

from train.utils import util, buffer
from train.agent.feature_sac import feature_sac_agent
from train.agent.feature_sac import predictor_sac_IB
from train.utils.env import TunableRewardAviary, DomainRandomizationWrapper

device = 'cuda' if torch.cuda.is_available() else 'cpu'







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Predictor, the Vanilla SAC/STEADY stage-1 baseline, or DR."
    )
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument(
        "--alg",
        default="spederv3",
        choices=["sac_predictor_IB", "spederv3", "domain-randomization"],
        help=(
            "sac_predictor_IB = paper Predictor; "
            "spederv3 = Vanilla SAC policy plus STEADY stage-1 mu checkpoint; "
            "domain-randomization = DR baseline (must pair with --env domain-randomization)."
        ),
    )
    parser.add_argument(
        "--env",
        default="tunable-reward",
        choices=["tunable-reward", "domain-randomization"],
        help="tunable-reward is the paper hover task. Use domain-randomization only with --alg domain-randomization.",
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--start_timesteps", default=0, type=float,
                        help="Random-action warmup. Paper runs use 0 (on-policy collection from the Mellinger actor).")
    parser.add_argument("--eval_freq", default=1e4, type=int)
    parser.add_argument("--max_timesteps", default=10e5, type=float,
                        help="Environment steps. Paper trains for 1e6 steps.")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Minibatch size B. Paper default 256; ablations also use 64 and 128.")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--feature_dim", default=512, type=int,
                        help="Latent width of the Predictor / critic feature / STEADY mu.")
    parser.add_argument("--discount", default=0.99,
                        help="SAC discount gamma.")
    parser.add_argument("--tau", default=0.005,
                        help="Polyak coefficient for the target critic.")
    parser.add_argument("--extra_feature_steps", default=1, type=int,
                        help="Predictor/mu is updated (extra_feature_steps + 1) times per SAC step; default N_Pred=2.")

    # Reward tuning settings (only for tunable-reward environment)
    parser.add_argument("--w_pos", default=2.5, type=float, 
                        help="Weight for position error penalty (higher = tighter tracking)")
    parser.add_argument("--w_rpy", default=1.5, type=float, 
                        help="Weight for orientation (roll/pitch) error penalty")
    parser.add_argument("--w_lin_vel", default=0.05, type=float, 
                        help="Weight for linear velocity penalty (higher = slower movement)")
    parser.add_argument("--w_ang_vel", default=0.05, type=float, 
                        help="Weight for angular velocity penalty")
    parser.add_argument("--w_action", default=0.1, type=float, 
                        help="Weight for action magnitude penalty (reduces overall thrust usage)")

    parser.add_argument('-b', '--beta', default=1.0, type=float,
                        help='Weight of next-state dependence in the nHSIC bottleneck')
    parser.add_argument('-l', '--lam', default=0.25, type=float,
                        help='Predictor-to-critic nHSIC alignment weight')
    parser.add_argument('-ei', '--exp_idx', default=[0], nargs='+', type=int,
                        help='Experiment index used in the output directory name')

    parser.add_argument("--wandb_offline", action="store_true", 
                        help="Run wandb in offline mode (useful for HPC compute nodes without internet)")
    args = parser.parse_args()

    if (args.alg == "domain-randomization") != (args.env == "domain-randomization"):
        parser.error("--alg domain-randomization and --env domain-randomization must be used together")

    exp_idx_str = str(args.exp_idx[0])

    base_exp_name = (f"batch_{args.batch_size}_featureDim_{args.feature_dim}_"
                     f"seed_{args.seed}_lambda_{args.lam}_beta_{args.beta}_"
                     f"extraFeatureStep_{args.extra_feature_steps}_"
                     f"experiment_{exp_idx_str}")
    exp_name = base_exp_name

    log_path = os.path.join("log", args.env, args.alg, exp_name)
    os.makedirs(log_path, exist_ok=True) 

    wandb_mode = "offline" if args.wandb_offline else "online"

    wandb.init(
        project=f"Predictor_quadrotor_{args.env}", 
        name=exp_name,                    
        config=vars(args),               
        dir=log_path,
        mode=wandb_mode
    )

    custom_weights = {
        'w_pos': args.w_pos,
        'w_rpy': args.w_rpy,
        'w_lin_vel': args.w_lin_vel,
        'w_ang_vel': args.w_ang_vel,
        'w_action': args.w_action,
    }
    if args.env == "domain-randomization":
        # Train under DR; evaluate on the unrandomized paper hover task.
        env = DomainRandomizationWrapper(
            TunableRewardAviary(reward_weights=custom_weights, gui=False, record=False)
        )
        eval_env = TunableRewardAviary(
            reward_weights=custom_weights, gui=False, record=False
        )
        mission = "DomainRandomization"
    else:
        env = TunableRewardAviary(
            reward_weights=custom_weights, gui=False, record=False
        )
        eval_env = TunableRewardAviary(
            reward_weights=custom_weights, gui=False, record=False
        )
        mission = "TunableReward"

    kwargs_args = vars(args)
    with open(os.path.join(log_path, 'train_params.pkl'), 'wb') as fp:
        pkl.dump(kwargs_args, fp)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
        "beta":args.beta,
        # sigma=None: nHSIC kernel bandwidth is the paper's median heuristic.
        "sigma": None,
        "lam":args.lam,
        "mission": mission,
        "extra_feature_steps": args.extra_feature_steps,
        "feature_dim": args.feature_dim,
        "device": args.device,
    }

    if args.alg == 'spederv3':
        agent = feature_sac_agent.SPEDERAgentV3Mel(**kwargs)
    elif args.alg == 'sac_predictor_IB':
        agent = predictor_sac_IB.SPEDERAgent_PredictorIB(**kwargs)
    else:
        agent = feature_sac_agent.SPEDERAgentDomainRandomization(**kwargs)

    replay_buffer = buffer.OptimizedReplayBuffer(state_dim, action_dim, device=args.device)

    util.eval_policy(agent, eval_env)

    state, _ = env.reset()
    done = False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    timer = util.Timer()

    best_eval_ret = -1e6

    def save_auxiliary(predictor_name, feature_mu_name):
        # Predictor: Z network. spederv3: STEADY mu (needed for stage 2).
        if args.alg == "sac_predictor_IB":
            torch.save(agent.feature_phi.state_dict(), os.path.join(log_path, predictor_name))
        elif args.alg == "spederv3":
            torch.save(agent.feature_mu.state_dict(), os.path.join(log_path, feature_mu_name))

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            # Paper: start_timesteps=0, so all data comes from the Mellinger SAC policy.
            action = agent.select_action(state, explore=True)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        done_bool = float(done)

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= args.start_timesteps:
            info = agent.train(replay_buffer, batch_size=args.batch_size)

            if (t+1) % 1000 == 0: 
                wandb.log({f"train/{k}": v for k, v in info.items()}, step=t+1)

        if done:
            total_T = t + 1
            ep_num = episode_num + 1
            ep_T = episode_timesteps
            reward = episode_reward

            print(f"Total T: {total_T} Episode Num: {ep_num} Episode T: {ep_T} Reward: {reward:.3f}")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            wandb.log({
                "rollout/Reward_Episode": reward,
                "rollout/Episode_Length": ep_T
            }, step=total_T)

            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0

            episode_num += 1

        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eva_ret = util.eval_policy(agent, eval_env)

            wandb.log({"eval/Average_Return": eva_ret}, step=t+1)

            if eva_ret > best_eval_ret:
                # best_* checkpoints are what STEADY stage 2 and hardware deploy load.
                best_eval_ret = eva_ret
                best_actor = agent.actor.state_dict()
                best_critic = agent.critic.state_dict()
                torch.save(best_actor, os.path.join(log_path, 'best_actor.pth'))
                torch.save(best_critic, os.path.join(log_path, 'best_critic.pth'))
                save_auxiliary("best_predictor.pth", "best_feature_mu.pth")
            
            if t >= int(args.max_timesteps) - 5:
                terminal_actor = agent.actor.state_dict()
                terminal_critic = agent.critic.state_dict()
                torch.save(terminal_actor, os.path.join(log_path, 'terminal_actor_{}.pth'.format(t)))
                torch.save(terminal_critic, os.path.join(log_path, 'terminal_critic_{}.pth'.format(t)))
                save_auxiliary("terminal_predictor.pth", f"terminal_mu_{t}.pth")

            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))
            
        if t % 5000 == 0:   
            gc.collect()
            torch.cuda.empty_cache()

    wandb.finish()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(agent.actor.state_dict(), os.path.join(log_path, 'last_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(log_path, 'last_critic.pth'))
    save_auxiliary("last_predictor.pth", "last_feature_mu.pth")