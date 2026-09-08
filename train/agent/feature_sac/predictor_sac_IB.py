"""Paper Predictor: SAC with an nHSIC information-bottleneck auxiliary.

Paper name: Predictor (nHSIC-IB).
CLI: ``python main_pyb_train.py --alg sac_predictor_IB``.

The nHSIC estimator is adapted from
https://github.com/choasma/HSIC-bottleneck (see ``train/math/hsic.py``).
The Predictor alignment objective around that estimator is this paper.

Symbols
-------
Z = phi(s, a)     Predictor features (MLPFeaturePhi_IB), not used as the actor input.
phi_Q             Critic feature head CriticwithPhi.get_feature(s, a).
beta              Weight of next-state dependence in the bottleneck (paper default 1).
lambda            Alignment weight between phi_Q and Z (paper default 0.25).

Each SAC step:
    1. N_Pred = extra_feature_steps + 1 updates of
           L_IB = nHSIC(Z, [s,a]) - beta * nHSIC(Z, s')
    2. Critic TD loss minus lambda * nHSIC(phi_Q, stopgrad(Z))
    3. Standard SAC actor/alpha on the differentiable Mellinger policy
    4. Polyak update of the target critic
"""
import torch
import torch.nn.functional as F
from train.math.hsic import hsic_normalized_cca_fast

from train.utils import util
from train.agent.sac.sac_agent import SACAgent
from train.agent.sac.critic import CriticwithPhi
from train.networks.features import MLPFeaturePhi_IB
from train.agent.sac.mellinger_pybullet import DifferentiableMellinger_pybullet


class SPEDERAgent_PredictorIB(SACAgent):
    """Predictor-aligned SAC. Replaces the base Gaussian actor with Mellinger."""

    def __init__(
            self,
            state_dim,
            action_dim,
            action_space,
            lr=1e-4,
            discount=0.99,
            target_update_period=2,
            tau=0.005,
            alpha=0.1,
            auto_entropy_tuning=True,
            hidden_dim=256,
            feature_dim=256,
            extra_feature_steps=1,
            device='cpu',
            beta=1,
            sigma=None,
            lam=0,
            mission="HoverAviary",
            **kwargs

    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            action_space=action_space,
            lr=lr,
            tau=tau,
            alpha=alpha,
            discount=discount,
            target_update_period=target_update_period,
            auto_entropy_tuning=auto_entropy_tuning,
            hidden_dim=hidden_dim,
            device=device,
            **kwargs
        )

        self.feature_dim = feature_dim
        self.extra_feature_steps = extra_feature_steps
        self.beta = beta
        # None => median-heuristic bandwidth used in the paper (see hsic.py).
        self.sigma = sigma
        self.lam = lam
        self.env = mission

        self.feature_phi = MLPFeaturePhi_IB(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            device=device,
        )

        self.critic = CriticwithPhi(
            input_dim=state_dim + action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        self.critic_target = CriticwithPhi(
            input_dim=state_dim + action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DifferentiableMellinger_pybullet().to(self.device)

        self.actor.set_device(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        # Paper actor learning rate (PID gains + log-std trunk).
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                        lr=3e-5,
                                        betas=[0.9, 0.999])

        # Predictor Z and critic both use 1e-4 (paper). Actor uses 3e-5.
        self.featurePhi_optimizer = torch.optim.Adam(self.feature_phi.parameters(), lr=1e-4, weight_decay=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=[0.9, 0.999])

    def feature_step(self, batch):
        """Information bottleneck: compress (s,a) while remaining dependent on s'.

        L_IB = I(Z; [s,a]) - beta * I(Z; s'), with I estimated by nHSIC.
        """
        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        next_state = batch.next_state.to(self.device)
        input = torch.cat([state, action], axis=-1)
        feature_phi = self.feature_phi(state, action)

        I_Z_S = hsic_normalized_cca_fast(feature_phi, input, self.sigma)
        I_Z_Y = hsic_normalized_cca_fast(feature_phi, next_state, self.sigma)
        feature_losses = I_Z_S - self.beta * I_Z_Y
        self.featurePhi_optimizer.zero_grad()
        feature_losses.backward()
        self.featurePhi_optimizer.step()

        return {
            'feature_loss': feature_losses.item(),
            }

    def critic_step(self, batch):
        """SAC TD loss minus lambda * nHSIC(phi_Q, stopgrad(Z))."""
        obs, action, next_obs, reward, done = util.unpack_batch(batch)

        obs = obs.to(self.device)
        action = action.to(self.device)
        next_obs = next_obs.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        not_done = 1. - done

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)

        phi_critic = self.critic.get_feature(obs, action)
        # stopgrad(Z): alignment updates Q, not the Predictor, in this step.
        phi_predictor = self.feature_phi(obs, action).detach()
        MI_phi = hsic_normalized_cca_fast(phi_critic, phi_predictor, self.sigma)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) - self.lam * MI_phi
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
			'q_loss': critic_loss.item(),
			'q1': current_Q1.mean().item(),
			'q2': current_Q1.mean().item()
			}

    def train(self, buffer, batch_size):
        """N_Pred = extra_feature_steps + 1 bottleneck updates, then SAC."""
        self.steps += 1
        batch = buffer.sample(batch_size)

        for _ in range(self.extra_feature_steps + 1):
            feature_info = self.feature_step(batch)

        critic_info = self.critic_step(batch)

        actor_info = self.update_actor_and_alpha(batch)

        self.update_target()

        return {

            **feature_info,
            **critic_info,
            **actor_info
        }
