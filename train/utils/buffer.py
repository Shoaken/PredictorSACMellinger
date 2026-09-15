"""Replay buffers for simulation SAC and STEADY radio CSV.

OptimizedReplayBuffer
    On-policy collection during ``main_pyb_train.py``. Capacity 1e6, 28-D
    state, 4-D PWM action.

RealDataBuffer
    STEADY stage 2. Loads Crazyflie radio / ground-station CSV (not SD-card
    USD logs). Kinematic/motor columns are CSV_REQUIRED_FIELDS. Motors are
    scaled motor_m*req / 65535 to match the Mellinger PWM action.

    Reconstructs the 28-D HoverAviary observation: world xyz (not position
    error), quaternion from Euler with pitch sign flip, integral/derivative
    PID terms. Stock firmware does not log ctrlMel.pos_error_* or
    ctrlMel.i_err_m*; if those CSV columns are missing or NaN,
    ``_reconstruct_ctrlmel_errors`` fills the rotation-integral state.

    Reward uses the same +2 offset and (2.5, 0.05, 0.05, 0.1) weights as
    simulation **except** w_rpy = 0.1 (HoverAviary default). Simulation
    TunableRewardAviary uses w_rpy = 1.5. Leave this unless you re-run STEADY.
"""
import collections
import numpy as np
import torch

import pandas as pd
import os
from scipy.spatial.transform import Rotation


Batch = collections.namedtuple(
	'Batch',
	['state', 'action', 'reward', 'next_state', 'done']
	)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cpu'):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = torch.device(device)

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
	
	def sample(self, batch_size): 
		ind = np.random.randint(0, self.size, size=batch_size)
		
		batch = Batch(
        	state=self.state[ind],
        	action=self.action[ind],
        	next_state=self.next_state[ind],
        	reward=self.reward[ind],
        	done=self.done[ind],
    	)
		
		return Batch(
        		state=torch.from_numpy(batch.state).float().to(self.device, non_blocking=True),
        		action=torch.from_numpy(batch.action).float().to(self.device, non_blocking=True),
        		next_state=torch.from_numpy(batch.next_state).float().to(self.device, non_blocking=True),
        		reward=torch.from_numpy(batch.reward).float().to(self.device, non_blocking=True),
        		done=torch.from_numpy(batch.done).float().to(self.device, non_blocking=True),
    		)
	

class OptimizedReplayBuffer(object):
    """Simulation FIFO replay used by Predictor, Vanilla SAC, and DR."""
    def __init__(self, state_dim, action_dim, max_size=1000000, device='cpu'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.bool_)

        self.device = torch.device(device)
        
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return Batch(
            state=torch.from_numpy(self.state[indices]).float().to(self.device),
            action=torch.from_numpy(self.action[indices]).float().to(self.device),
            next_state=torch.from_numpy(self.next_state[indices]).float().to(self.device),
            reward=torch.from_numpy(self.reward[indices]).float().to(self.device),
            done=torch.from_numpy(self.done[indices]).float().to(self.device),
        )
    
    def clear(self):
        """Clear all stored transitions."""
        self.ptr = 0
        self.size = 0
        self.state.fill(0)
        self.action.fill(0)
        self.next_state.fill(0)
        self.reward.fill(0)
        self.done.fill(False)

		

class RealDataBuffer(ReplayBuffer):
	"""Offline STEADY buffer from one CSV file or a directory of CSVs."""
	CSV_REQUIRED_FIELDS = {
		"stateX", "stateY", "stateZ",
		"stabilizer_roll_deg", "stabilizer_pitch_deg", "stabilizer_yaw_deg",
		"stateEstimate_vx", "stateEstimate_vy", "stateEstimate_vz",
		"stateEstimateZ_rateRoll_mrad_s",
		"stateEstimateZ_ratePitch_mrad_s",
		"stateEstimateZ_rateYaw_mrad_s",
		"motor_m1req", "motor_m2req", "motor_m3req", "motor_m4req",
	}

	# Optional: stock Mellinger does not expose these as log variables.
	# Missing or NaN values are filled by _reconstruct_ctrlmel_errors.
	CSV_CTRLMEL_FIELDS = (
		"ctrlMel_pos_error_x", "ctrlMel_pos_error_y", "ctrlMel_pos_error_z",
		"ctrlMel_i_err_mx", "ctrlMel_i_err_my", "ctrlMel_i_err_mz",
	)

	def __init__(self, max_size=int(1e6), device='cpu'):
		super(RealDataBuffer, self).__init__(state_dim = 28, action_dim = 4, max_size=max_size,
											 device=device)

	def compute_reward(self, pos_error, rpy, vxyz, rpy_rate, control):
		# Matches sim +2 offset. w_rpy=0.1 here; paper sim hover uses 1.5.
		rew_pos = - 2.5 * np.linalg.norm(pos_error, axis=1)
		rew_rpy = - 0.1 * np.linalg.norm(rpy, axis=1)
		rew_lin_vel = - 0.05 * np.linalg.norm(vxyz, axis=1)
		rew_ang_vel = - 0.05 * np.linalg.norm(rpy_rate, axis=1)
		rew_action = - 0.1 * np.linalg.norm(control, axis=1)
		return 2 + (rew_pos +
					rew_rpy +
					rew_lin_vel +
					rew_ang_vel +
					rew_action)

	def _clip_integral_pos(self, pos_e, dt):
		integral_pos = np.zeros_like(pos_e)
		acc = np.zeros(3)
		for i in range(pos_e.shape[0]):
			acc = np.clip(acc + pos_e[i] * float(dt[i]), -2.0, 2.0)
			acc[2] = np.clip(acc[2], -0.15, 0.15)
			integral_pos[i] = acc
		return integral_pos

	def _transitions_from_world(self, world_pos, rpy, vxyz, rpy_rate, integral_rpy_error, ctrl, goal, dt):
		# Match HoverAviary / Mellinger: obs[0:3] is world position, not position error.
		pos_e = goal[None, :] - world_pos
		integral_pos_error = self._clip_integral_pos(pos_e, dt)
		quat = np.zeros([rpy.shape[0], 4])
		for i, angle in enumerate(rpy):
			quat[i] = Rotation.from_euler('XYZ', np.multiply([1, -1, 1], angle)).as_quat()
		state = np.hstack([world_pos, quat, rpy, vxyz, rpy_rate,
						   integral_pos_error,
						   -1 * vxyz,
						   integral_rpy_error,
						   -1 * rpy_rate,
						   ])
		st = state[:-1]
		at = ctrl[:-1]
		stp1 = state[1:]
		reward = self.compute_reward(pos_e, rpy, vxyz, rpy_rate, ctrl)[:-1]
		return st, at, reward, stp1

	def _world_position(self, pos, goal):
		# Radio CSV currently logs xyz near the hover origin (z ~ 0), not world z = 1.
		if np.median(np.abs(pos[:, 2] - goal[2])) < np.median(np.abs(pos[:, 2])):
			return pos
		return pos + goal

	def _reconstruct_ctrlmel_errors(self, pos, rpy, vxyz, dt, goal):
		"""Firmware-style Mellinger errors when ctrlMel_* was not logged.

		ctrlMel.pos_error  = setpoint - position
		ctrlMel.i_err_m    = integral of rotation error (DSLPID / Mellinger)
		"""
		n = pos.shape[0]
		ctrl_pos_error = goal[None, :] - pos
		kp = np.array([0.4, 0.4, 1.25])
		ki = np.array([0.05, 0.05, 0.05])
		kd = np.array([0.2, 0.2, 0.6])
		gravity = np.array([0.0, 0.0, 9.81 * 0.027])
		target_x_c = np.array([1.0, 0.0, 0.0])

		i_pos = np.zeros(3)
		i_m = np.zeros(3)
		i_err_m = np.zeros((n, 3))
		for i in range(n):
			dti = float(dt[i])
			pos_e = ctrl_pos_error[i]
			i_pos = np.clip(i_pos + pos_e * dti, -2.0, 2.0)
			i_pos[2] = np.clip(i_pos[2], -0.15, 0.15)

			target_thrust = kp * pos_e + ki * i_pos + kd * (-vxyz[i]) + gravity
			norm_t = np.linalg.norm(target_thrust)
			target_z = target_thrust / norm_t if norm_t > 1e-8 else np.array([0.0, 0.0, 1.0])
			target_y = np.cross(target_z, target_x_c)
			norm_y = np.linalg.norm(target_y)
			target_y = target_y / norm_y if norm_y > 1e-8 else np.array([0.0, 1.0, 0.0])
			target_x = np.cross(target_y, target_z)
			r_des = np.stack([target_x, target_y, target_z], axis=1)

			r_cur = Rotation.from_euler(
				'XYZ', np.multiply([1, -1, 1], rpy[i])).as_matrix()
			rot_matrix_e = r_des.T @ r_cur - r_cur.T @ r_des
			rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
			i_m = i_m - rot_e * dti
			i_m = np.clip(i_m, -1500.0, 1500.0)
			i_m[0:2] = np.clip(i_m[0:2], -1.0, 1.0)
			i_err_m[i] = i_m
		return ctrl_pos_error, i_err_m

	def load_csv_data(self, filename, goal=np.array([0.0, 0.0, 1.0])):
		# Radio / ground-station CSV with the same physical channels as the USD log.
		df = pd.read_csv(filename)
		missing = sorted(self.CSV_REQUIRED_FIELDS.difference(df.columns))
		if missing:
			raise ValueError(
				f"{filename} is missing required radio-log columns: {', '.join(missing)}"
			)
		index = df['motor_m1req'].to_numpy() > 0
		if not np.any(index):
			raise ValueError(f"{filename} contains no samples with motor_m1req > 0")
		pos = np.vstack([
			df['stateX'].to_numpy()[index],
			df['stateY'].to_numpy()[index],
			df['stateZ'].to_numpy()[index],
		]).T
		pos = self._world_position(pos, goal)
		rpy = np.vstack([
			df['stabilizer_roll_deg'].to_numpy()[index],
			df['stabilizer_pitch_deg'].to_numpy()[index],
			df['stabilizer_yaw_deg'].to_numpy()[index],
		]).T / 180. * np.pi
		vxyz = np.vstack([
			df['stateEstimate_vx'].to_numpy()[index],
			df['stateEstimate_vy'].to_numpy()[index],
			df['stateEstimate_vz'].to_numpy()[index],
		]).T
		rpy_rate = np.vstack([
			df['stateEstimateZ_rateRoll_mrad_s'].to_numpy()[index],
			df['stateEstimateZ_ratePitch_mrad_s'].to_numpy()[index],
			df['stateEstimateZ_rateYaw_mrad_s'].to_numpy()[index],
		]).T / 1000.
		ctrl = np.vstack([
			df['motor_m1req'].to_numpy()[index] / 65535.,
			df['motor_m2req'].to_numpy()[index] / 65535.,
			df['motor_m3req'].to_numpy()[index] / 65535.,
			df['motor_m4req'].to_numpy()[index] / 65535.,
		]).T

		if 'time' in df.columns:
			t = df['time'].to_numpy()[index]
			dt = np.diff(t, prepend=t[0])
			median_dt = np.median(dt[dt > 1e-6]) if np.any(dt > 1e-6) else 0.02
			dt = np.where(dt > 1e-6, dt, median_dt)
		else:
			freq = float(df['fixedFrequency'].to_numpy()[index][0]) if 'fixedFrequency' in df.columns else 50.0
			dt = np.full(pos.shape[0], 1.0 / max(freq, 1.0))

		use_logged_ctrlmel = all(name in df.columns for name in self.CSV_CTRLMEL_FIELDS)
		if use_logged_ctrlmel:
			logged_pos_error = np.vstack([
				df['ctrlMel_pos_error_x'].to_numpy()[index],
				df['ctrlMel_pos_error_y'].to_numpy()[index],
				df['ctrlMel_pos_error_z'].to_numpy()[index],
			]).T
			logged_i_err = np.vstack([
				df['ctrlMel_i_err_mx'].to_numpy()[index],
				df['ctrlMel_i_err_my'].to_numpy()[index],
				df['ctrlMel_i_err_mz'].to_numpy()[index],
			]).T
			use_logged_ctrlmel = (
				np.isfinite(logged_pos_error).all()
				and np.isfinite(logged_i_err).all()
			)
		if use_logged_ctrlmel:
			integral_rpy_error = logged_i_err
		else:
			_, integral_rpy_error = self._reconstruct_ctrlmel_errors(
				pos, rpy, vxyz, dt, goal)

		return self._transitions_from_world(
			pos, rpy, vxyz, rpy_rate, integral_rpy_error, ctrl, goal, dt)

	def load_all_data(self, log_path):
		sts, ats, rewards, stp1s = [], [], [], []
		if os.path.isfile(log_path):
			files = [log_path]
		else:
			if not os.path.isdir(log_path):
				raise FileNotFoundError(f"CSV log path does not exist: {log_path}")
			files = [
				os.path.join(log_path, name) for name in os.listdir(log_path)
				if os.path.isfile(os.path.join(log_path, name))
				and name.lower().endswith(".csv")
			]
		if not files:
			raise ValueError(f"No CSV flight logs found at: {log_path}")
		for log_file in files:
			if not log_file.lower().endswith('.csv'):
				raise ValueError(f"Only radio/ground-station CSV logs are supported: {log_file}")
			st, at, reward, stp1 = self.load_csv_data(log_file)
			sts.append(st)
			ats.append(at)
			rewards.append(reward)
			stp1s.append(stp1)
		self.state = np.vstack(sts)
		self.action = np.vstack(ats)
		self.reward = np.hstack(rewards)[:, np.newaxis]
		self.next_state = np.vstack(stp1s)
		self.done = np.zeros_like(self.reward)
		self.size = self.state.shape[0]


