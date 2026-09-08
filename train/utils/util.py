"""Shared training helpers (eval, MLP builder, numpy conversion).

eval_policy runs 10 deterministic hover episodes (explore=False). Paper
training logs this every ``eval_freq`` (default 1e4) environment steps.
"""
import time
import numpy as np 

from torch import nn


def unpack_batch(batch):
  return batch.state, batch.action, batch.next_state, batch.reward, batch.done


class Timer:

	def __init__(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def reset(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def set_step(self, step):
		self._step = step
		self._step_time = time.time()

	def time_cost(self):
		return time.time() - self._start_time

	def steps_per_sec(self, step):
		sps = (step - self._step) / (time.time() - self._step_time)
		self._step = step
		self._step_time = time.time()
		return sps


def eval_policy(policy, eval_env, eval_episodes=10):
	"""Mean undiscounted return over ``eval_episodes`` (default 10, paper)."""
	avg_reward = 0.
	for _ in range(eval_episodes):
		state, _ = eval_env.reset()
		done = False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, terminated, truncated, _ = eval_env.step(action)
			done=terminated or truncated
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward



def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, hidden_activation=nn.ELU(inplace=True), output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), hidden_activation] # inplace=True
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), hidden_activation] # inplace=True
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk

def to_np(t):
	if t is None:
		return None
	elif t.nelement() == 0:
		return np.array([])
	else:
		return t.cpu().detach().numpy()









