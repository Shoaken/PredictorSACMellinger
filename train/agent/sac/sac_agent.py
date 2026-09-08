"""Haarnoja SAC (pytorch_sac) used as the shared update engine.

This class constructs DoubleQCritic + DiagGaussianActor. Paper methods
replace those modules in their own __init__:
    Predictor / Vanilla SAC  -> CriticwithPhi + DifferentiableMellinger_pybullet
    DR                       -> keep DoubleQCritic, swap in Mellinger

``--alg sac`` is intentionally not exposed. Keep this file: Mellinger
actors still call update_actor_and_alpha / critic_step / select_action.

Actor lr 3e-5, critic lr from ``lr`` (paper 1e-4), tau=0.005, gamma=0.99,
target entropy -action_dim, automatic alpha.
Adapted from https://github.com/denisyarats/pytorch_sac
"""
import numpy as np
import torch
import torch.nn.functional as F

from train.utils import util

from train.agent.sac.critic import DoubleQCritic
from train.agent.sac.actor import DiagGaussianActor

S2R_HIDDEN_DIM = 64


class SACAgent(object):
	"""Soft Actor-Critic skeleton. Paper agents subclass and swap networks."""
	def __init__(
			self, 
			state_dim, 
			action_dim, 
			action_space, 
			lr=3e-4,
			discount=0.99, 
			target_update_period=2,
			tau=0.005,
			alpha=0.1,
			auto_entropy_tuning=True,
			hidden_dim=1024,
			#device='cuda',
			device='cpu',
			**kwargs
			):

		self.steps = 0

		self.device = device 
		self.action_range = [
			float(action_space.low.min()),
			float(action_space.high.max())
		]
		self.discount = discount 
		self.tau = tau 
		self.target_update_period = target_update_period
		self.learnable_temperature = auto_entropy_tuning

		# Defaults swapped out by Predictor / Vanilla SAC / DR subclasses.
		self.critic = DoubleQCritic(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=2,
			device=self.device
		)
		self.critic_target = DoubleQCritic(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=hidden_dim,
			hidden_depth=2,
			device=self.device
		)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.actor = DiagGaussianActor(
			obs_dim=state_dim, 
			action_dim=action_dim,
			hidden_dim=S2R_HIDDEN_DIM,
			hidden_depth=2,
			log_std_bounds=[-20., 1.],
		).to(self.device)
		self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -action_dim
		
		 # optimizers
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
																						lr=3e-5,
																						betas=[0.9, 0.999])

		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
																							lr=lr,
																							betas=[0.9, 0.999])

		self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
																								lr=lr,
																								betas=[0.9, 0.999])


	@property
	def alpha(self):
		return self.log_alpha.exp()


	def select_action(self, state, explore=False):
		state = torch.FloatTensor(state).to(self.device)
		state = state.unsqueeze(0)
		dist = self.actor(state)
		action = dist.sample() if explore else dist.mean
		action = action.clamp(*self.action_range)
		assert action.ndim == 2 and action.shape[0] == 1
		return util.to_np(action[0])


	def update_target(self):
		if self.steps % self.target_update_period == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def critic_step(self, batch):
		"""Twin-Q TD update. Predictor overrides this to add nHSIC alignment."""
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
		target_V = torch.min(target_Q1,
													target_Q2) - self.alpha.detach() * log_prob
		target_Q = reward + (not_done * self.discount * target_V)
		target_Q = target_Q.detach()

		# get current Q estimates
		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
				current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		return {
			'q_loss': critic_loss.item(), 
			'q1': current_Q1.mean().item(),
			'q2': current_Q1.mean().item()
			}


	def update_actor_and_alpha(self, batch):
		"""SAC actor: E[alpha log pi - min Q]. Then project Mellinger PID gains."""
		obs = batch.state.to(self.device) 

		dist = self.actor(obs)
		action = dist.rsample()
		log_prob = dist.log_prob(action).sum(-1, keepdim=True)
		actor_Q1, actor_Q2 = self.critic(obs, action)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

		# optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		gain_names = ("kp_z", "ki_z", "kd_z")
		gain_values_before = {}
		gain_info = {}
		for name in gain_names:
			param = getattr(self.actor, name, None)
			if param is None:
				continue
			gain_values_before[name] = param.detach().clone()
			gain_info[f'{name}_value_before'] = param.detach().item()
			raw_param = self.actor.gain_raw_parameter(name) if hasattr(self.actor, "gain_raw_parameter") else param
			if raw_param is not None and raw_param.grad is not None:
				gain_info[f'{name}_raw_grad'] = raw_param.grad.detach().item()
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


	def train(self, buffer, batch_size):
		"""
		One train step
		"""
		self.steps += 1

		batch = buffer.sample(batch_size)
		# Acritic step
		critic_info = self.critic_step(batch)

		# Actor and alpha step
		actor_info = self.update_actor_and_alpha(batch)

		# Update the frozen target models
		self.update_target()

		return {
			**critic_info, 
			**actor_info,
		}
	

