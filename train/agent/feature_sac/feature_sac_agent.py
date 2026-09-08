"""Vanilla SAC (spederv3), STEADY stage 2 (TransferAgent), and the DR agent.

STEADY skill learning and residual transfer are adapted from
https://github.com/mahaitongdae/steady_sim_to_real/tree/published
(Ma et al., arXiv:2404.05051). The Mellinger actor, DR agent, and
how ``spederv3`` is reported as Vanilla SAC are specific to this paper.

Paper mapping
-------------
Vanilla SAC / STEADY stage 1
    SPEDERAgentV3Mel  (--alg spederv3)
    Same Mellinger actor as Predictor. Critic is CriticwithPhi.
    feature_mu is trained with an inner-product / Hilbert-Schmidt loss so
    that STEADY stage 2 can freeze simulator skills. It does **not** enter
    the actor objective; the resulting policy is the paper's Vanilla SAC.

STEADY (complete method)
    Stage 1: this file's SPEDERAgentV3Mel, then
    Stage 2: TransferAgent via main_online_STEADY.py
    Freeze phi^circ, mu^circ, pi^circ. Learn residual phi, mu. Linear
    Q(s,a) = w^T [phi^circ(s,a), phi(s,a)] plus KL(pi || pi^circ).

Domain randomization
    SPEDERAgentDomainRandomization. Mellinger actor, DoubleQCritic from
    SACAgent (not CriticwithPhi). Dynamics randomization is in
    train.utils.env.DomainRandomizationWrapper.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
import os


from train.utils.util import unpack_batch
from train.agent.sac.sac_agent import SACAgent
from train.agent.sac.critic import CriticwithPhi
from train.networks.features import MLPFeatureMu, MLPFeaturePhi
from train.agent.sac.mellinger_pybullet import DifferentiableMellinger_pybullet

class LineaCritic(nn.Module):
    """STEADY linear Q-head: Q1, Q2 = w^T concat(phi^circ, phi)."""

    def __init__(
            self,
            feature_dim,
    ):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(feature_dim, 1)

        # Q2
        self.l2 = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """
		"""

        q1 = self.l1(x)
        q2 = self.l2(x)

        return q1, q2


class SPEDERAgentV3Mel(SACAgent):
    """Vanilla SAC policy used in the paper, and STEADY's simulator stage.

    ``feature_mu`` approximates next-state features for later residual
    discovery. Do not report this run as STEADY unless TransferAgent is
    trained afterwards.
    """

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
            feature_tau=0.001,
            feature_dim=256,  # latent feature dim
            use_feature_target=True,
            extra_feature_steps=1,
            device='cpu',
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
        self.feature_tau = feature_tau
        self.use_feature_target = use_feature_target
        self.extra_feature_steps = extra_feature_steps
        self.env = mission
        self.feature_mu = MLPFeatureMu(state_dim=state_dim, hidden_dim=hidden_dim, feature_dim=feature_dim).to(device)

        # Same object as feature_mu: Polyak on mu is a no-op. The target
        # handle exists so STEADY stage 2 can reuse this class's flags.
        if use_feature_target:
            self.feature_mu_target = self.feature_mu
        self.feature_optimizer = torch.optim.Adam(
            list(self.feature_mu.parameters()),
            lr=lr,
            weight_decay=1e-2
        )

        # Replaces SACAgent's DoubleQCritic. phi_Q = critic.get_feature.
        self.critic = CriticwithPhi(
            input_dim=state_dim + action_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=[0.9, 0.999])

        self.actor = DifferentiableMellinger_pybullet().to(self.device)
        self.actor.set_device(self.device)


        self.actor_optimizer = torch.optim.Adam( self.actor.parameters(), 
                                        lr=3e-5,
                                        betas=[0.9, 0.999])

    def feature_step(self, batch):
        # Hilbert-Schmidt / inner-product skill loss on (phi(s,a), mu(s')).
        # Used only to store mu for STEADY; the actor does not read mu.
        phi = self.critic.get_feature(batch.state, batch.action)
        mu = self.feature_mu(batch.next_state)
        model_learning_loss1 = - 2. * torch.sum(phi * mu, dim=-1)
        model_learning_loss2 = torch.mean(torch.matmul(phi, mu.T) ** 2, dim=1)
        model_learning_loss = model_learning_loss1 + model_learning_loss2
        model_learning_loss = model_learning_loss.mean()

        # loss = model_learning_loss

        self.feature_optimizer.zero_grad()
        model_learning_loss.backward()
        self.feature_optimizer.step()

        return {
            'feature_loss': model_learning_loss.item(),
            'model_learning_loss1': model_learning_loss1.mean().item(),
            'model_learning_loss2': model_learning_loss2.mean().item(),
        }

    def update_feature_target(self):
        for param, target_param in zip(self.feature_mu.parameters(), self.feature_mu_target.parameters()):
            target_param.data.copy_(self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def train(self, buffer, batch_size):
        """One SAC step plus N_mu = extra_feature_steps + 1 skill-mu updates."""
        self.steps += 1
        batch = buffer.sample(batch_size)

        # Acritic step
        critic_info = self.critic_step(batch)

        # Actor and alpha step
        actor_info = self.update_actor_and_alpha(batch)

        # Feature step
        for _ in range(self.extra_feature_steps + 1):
            feature_info = self.feature_step(batch)

            # Update the feature network if needed
            if self.use_feature_target:
                self.update_feature_target()

        # Update the frozen target models
        self.update_target()

        return {
            **feature_info,
            **critic_info,
            **actor_info,
        }

class TransferAgent(SPEDERAgentV3Mel):
    """STEADY online residual-skill stage (main_online_STEADY.py).

    Loads stage-1 ``best_actor.pth``, ``best_critic.pth``, ``best_feature_mu.pth``.
    Simulator phi^circ (critic feature), mu^circ, and pi^circ are frozen.
    Residual phi, mu are trained with the same inner-product loss plus an
    orthogonality penalty E[phi^circ^T phi]. Policy improves with
    Q = w^T [phi^circ, phi] and tau_kl * KL(pi || pi^circ).
    """

    def __init__(self,
                 agent_path,
                 state_dim,
                 action_dim,
                 action_space,
                 lr = 1e-4,
                 aug_feature_dim = 128,
                 **kwargs):
        super(TransferAgent, self).__init__(
            state_dim,
            action_dim,
            action_space,
            lr=lr,
            **kwargs
        )
        self.feature_mu.load_state_dict(
            torch.load(os.path.join(agent_path, 'best_feature_mu.pth'), map_location=self.device))
        self.critic.load_state_dict(
            torch.load(os.path.join(agent_path, 'best_critic.pth'), map_location=self.device))
        self.actor.load_state_dict(
            torch.load(os.path.join(agent_path, 'best_actor.pth'), map_location=self.device))

        # STEADY: freeze simulator skills φ°, μ° and keep a copy of π°
        for net in (self.critic, self.critic_target, self.feature_mu):
            for param in net.parameters():
                param.requires_grad = False
        self.actor_sim = copy.deepcopy(self.actor)
        for param in self.actor_sim.parameters():
            param.requires_grad = False
        # Fine-tune Mellinger PID only. The log_std trunk can collapse
        # (actor_loss ~ 1e14) and does not correspond to firmware gains.
        if hasattr(self.actor, "trunk"):
            for param in self.actor.trunk.parameters():
                param.requires_grad = False
        for name in ("kR_xy", "kw_xy", "kR_z", "kw_z"):
            param = getattr(self.actor, name, None)
            if param is not None:
                param.requires_grad_(True)
        self.tau_kl = kwargs.get('tau_kl', 0.1)
        self.discovery_lambda = kwargs.get('discovery_lambda', 1.0)

        self.aug_feature_dim = aug_feature_dim
        self.augmented_feature_phi = MLPFeaturePhi(
            state_dim, action_dim, hidden_dim=256, feature_dim=aug_feature_dim).to(self.device)
        self.augmented_feature_mu = MLPFeatureMu(
            state_dim, hidden_dim=256, feature_dim=aug_feature_dim).to(self.device)

        if self.use_feature_target:
            self.augmented_feature_phi_target = copy.deepcopy(self.augmented_feature_phi)
            self.augmented_feature_mu_target = copy.deepcopy(self.augmented_feature_mu)

        self.feature_optimizer = torch.optim.Adam(list(self.augmented_feature_phi.parameters())
                                                  + list(self.augmented_feature_mu.parameters()),
            lr=lr,
            weight_decay=1e-2)

        self.augemented_critic = LineaCritic(feature_dim=aug_feature_dim + self.feature_dim).to(self.device)
        # Initialize w1 from simulator Q head (paper: w1 ← w^{π°})
        with torch.no_grad():
            self.augemented_critic.l1.weight.data[:, :self.feature_dim] = self.critic.final_l1.weight.data
            self.augemented_critic.l1.bias.data.copy_(self.critic.final_l1.bias.data)
            self.augemented_critic.l2.weight.data[:, :self.feature_dim] = self.critic.final_l2.weight.data
            self.augemented_critic.l2.bias.data.copy_(self.critic.final_l2.bias.data)
        self.augemented_critic_target = copy.deepcopy(self.augemented_critic)

        self.aug_critic_optimizer = torch.optim.Adam(self.augemented_critic.parameters(), lr=lr, betas=[0.9, 0.999])

    def get_skill_q(self, state, action, use_target=False, detach_features=False):
        """STEADY eq. (13): Q = w^T [φ°, φ]. Simulator φ° is frozen."""
        phi_sim = self.critic.get_feature(state, action)
        if use_target and self.use_feature_target:
            phi_new = self.augmented_feature_phi_target(state, action)
            linear_critic = self.augemented_critic_target
        else:
            phi_new = self.augmented_feature_phi(state, action)
            linear_critic = self.augemented_critic
        if detach_features:
            phi_sim = phi_sim.detach()
            phi_new = phi_new.detach()
        combined_phi = torch.hstack([phi_sim, phi_new])
        return linear_critic(combined_phi)

    def critic_step(self, batch):
        """Update the linear critic over frozen and residual skill features."""
        state, action, next_state, reward, done = unpack_batch(batch)

        with torch.no_grad():
            dist = self.actor(next_state)
            next_action = dist.rsample()
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)

            next_q1, next_q2 = self.get_skill_q(next_state, next_action, use_target=True)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
            target_q = reward + (1. - done) * self.discount * next_q

        q1, q2 = self.get_skill_q(state, action, detach_features=True)
        q1_loss = F.mse_loss(target_q, q1)
        q2_loss = F.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss

        self.aug_critic_optimizer.zero_grad()
        q_loss.backward()
        self.aug_critic_optimizer.step()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1': q1.mean().item(),
            'q2': q2.mean().item()
        }

    def feature_step(self, batch):
        # STEADY skill discovery (eq. 11-12): fit residual skills φ, μ with φ°, μ° frozen
        # and orthogonal to simulator skills.
        with torch.no_grad():
            phi_sim = self.critic.get_feature(batch.state, batch.action)
            mu_sim = self.feature_mu(batch.next_state)
        phi_new = self.augmented_feature_phi(batch.state, batch.action)
        mu_new = self.augmented_feature_mu(batch.next_state)
        phi = torch.cat([phi_sim, phi_new], dim=-1)
        mu = torch.cat([mu_sim, mu_new], dim=-1)

        model_learning_loss1 = -2. * torch.sum(phi * mu, dim=-1)
        model_learning_loss2 = torch.mean(torch.matmul(phi, mu.T) ** 2, dim=1)
        model_learning_loss = (model_learning_loss1 + model_learning_loss2).mean()

        # ⟨φ°_i, φ_j⟩ = E[φ°_i(s,a) φ_j(s,a)]
        cross = torch.matmul(phi_sim.T, phi_new) / phi_sim.shape[0]
        penalty_loss = self.discovery_lambda * cross.abs().mean()
        loss = model_learning_loss + penalty_loss

        self.feature_optimizer.zero_grad()
        loss.backward()
        self.feature_optimizer.step()

        return {
            'feature_loss': loss.item(),
            'model_learning_loss1': model_learning_loss1.mean().item(),
            'model_learning_loss2': model_learning_loss2.mean().item(),
            'model_learning_loss': model_learning_loss.item(),
            'penalty_loss': penalty_loss.item(),
        }

    def update_feature_target(self):
        if not self.use_feature_target:
            return
        for param, target_param in zip(
                self.augmented_feature_phi.parameters(), self.augmented_feature_phi_target.parameters()):
            target_param.data.copy_(
                self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)
        for param, target_param in zip(
                self.augmented_feature_mu.parameters(), self.augmented_feature_mu_target.parameters()):
            target_param.data.copy_(
                self.feature_tau * param.data + (1 - self.feature_tau) * target_param.data)

    def update_target(self):
        super().update_target()
        for param, target_param in zip(
                self.augemented_critic.parameters(), self.augemented_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor_and_alpha(self, batch):
        """Policy improvement with Q = w^T [φ°, φ] and KL(π || π°) (eq. 13-14)."""
        obs = batch.state.to(self.device)

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.get_skill_q(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        with torch.no_grad():
            dist_sim = self.actor_sim(obs)
            log_prob_sim = dist_sim.log_prob(action).sum(-1, keepdim=True)
        kl_pi = (log_prob - log_prob_sim).mean()

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean() + self.tau_kl * kl_pi

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        pwm = dist.loc.detach()
        min_pwm = getattr(self.actor, "MIN_PWM", 20000.0) / getattr(self.actor, "MAX_PWM", 65535.0)
        pwm_clip_frac = ((pwm <= min_pwm + 1e-4) | (pwm >= 1.0 - 1e-4)).float().mean()
        pos_e = (self.actor.goal.to(obs.device) - obs[:, 0:3]).detach()
        gain_names = ("kp_xy", "kd_xy", "kp_z", "ki_z", "kd_z", "kR_xy", "kw_xy")
        gain_values_before = {}
        gain_info = {
            'pwm_clip_frac': pwm_clip_frac.item(),
            'pos_e_z_mean': pos_e[:, 2].mean().item(),
            'pos_xyz_mean_z': obs[:, 2].mean().item(),
        }
        for name in gain_names:
            param = getattr(self.actor, name, None)
            if param is None:
                continue
            gain_values_before[name] = param.detach().clone()
            gain_info[f'{name}_value_before'] = param.detach().item()
            raw_param = self.actor.gain_raw_parameter(name) if hasattr(self.actor, "gain_raw_parameter") else param
            gain_info[f'{name}_raw_grad'] = (
                0.0 if raw_param is None or raw_param.grad is None else raw_param.grad.detach().item()
            )
        self.actor_optimizer.step()
        if hasattr(self.actor, "projection_on_gains"):
            self.actor.projection_on_gains()
        for name, value_before in gain_values_before.items():
            param = getattr(self.actor, name)
            gain_info[f'{name}_value_after'] = param.detach().item()
            gain_info[f'{name}_delta'] = (param.detach() - value_before).item()
        if hasattr(self.actor, "i_range_z") and hasattr(self.actor, "ki_z"):
            gain_info['ki_z_at_upper_bound'] = float(
                self.actor.ki_z.detach().item() >= self.actor.i_range_z
            )

        info = {
            'actor_loss': actor_loss.item(),
            'actor_q': actor_Q.mean().item(),
            'kl_pi': kl_pi.item(),
            **gain_info
        }

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            info['alpha_loss'] = alpha_loss.item()
            info['alpha'] = self.alpha.item()

        return info


class SPEDERAgentDomainRandomization(SACAgent):
    """DR baseline: Mellinger SAC trained under DomainRandomizationWrapper.

    Keeps SACAgent's DoubleQCritic (not CriticwithPhi). Pair with
    ``--env domain-randomization``. Evaluation uses the unrandomized hover task.
    """
    
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
        device='cpu',
        mission="HoverAviaryDR",
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
        
        self.env = mission

        # Keep SACAgent DoubleQCritic; only the actor is Mellinger (paper DR).
        self.actor = DifferentiableMellinger_pybullet().to(self.device)
        self.actor.set_device(self.device)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=3e-5,
            betas=[0.9, 0.999]
        )
    
    def train(self, buffer, batch_size):
        """Run one standard SAC update."""
        self.steps += 1
        batch = buffer.sample(batch_size)
        
        critic_info = self.critic_step(batch)
        
        actor_info = self.update_actor_and_alpha(batch)
        
        self.update_target()
        
        return {
            **critic_info,
            **actor_info
        }

