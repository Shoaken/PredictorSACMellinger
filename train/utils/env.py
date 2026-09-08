"""Paper hover task and the DR dynamics wrapper.

Simulator: Crazyflie CF2X in gym-pybullet-drones HoverAviary
(submodule ``env/`` at commit 7856f34). Control rate 240 Hz, episode 5 s.
Observation is 28-D (16 kinematic + 12 PID error terms). Action is 4 PWM
commands in [-1, 1], which the Mellinger actor produces as pwm / MAX_PWM.

Paper reward (TunableRewardAviary; override HoverAviary defaults)::

    r = 2
        - 2.5  ||p - p*||          position, goal p* = (0, 0, 1)
        - 1.5  ||(roll, pitch)||   yaw is not penalized (state[7:9])
        - 0.05 ||v||
        - 0.05 ||omega||
        - 0.1  ||(u - u_hover) / u_max||

STEADY's RealDataBuffer currently uses w_rpy = 0.1 (HoverAviary default)
instead of 1.5. Do not silently "fix" that unless reproducing a new run.

DR randomizes mass, wind, linear drag, and motor noise at every reset.
``gravity_range`` is (9.81, 9.81): gravity is *not* randomized in the
paper experiments even though the field exists.

Do not edit files under ``env/gym_pybullet_drones/``; wrap here instead.
``train/utils/env.py`` (this module) is unrelated to the submodule folder.
"""

import sys
from pathlib import Path

import gymnasium
import numpy as np
import pybullet as p

# ./env is the mahaitongdae/gym-pybullet-drones submodule.
# Our code imports env.gym_pybullet_drones; the library itself still
# uses `import gym_pybullet_drones`, so the submodule root stays on path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBMODULE_ROOT = _REPO_ROOT / "env"
for _path in (_REPO_ROOT, _SUBMODULE_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from env.gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary


class DomainRandomizationWrapper(gymnasium.Wrapper):
    """Paper DR baseline: resample dynamics on ``reset``, noise/wind on ``step``.

    Pair with ``--alg domain-randomization --env domain-randomization``.
    Evaluation in ``main_pyb_train.py`` uses an *unwrapped* TunableRewardAviary
    so reported returns are comparable to Predictor / Vanilla SAC.
    """

    def __init__(self, env, dr_params=None):
        super().__init__(env)
        if hasattr(self.env, "unwrapped") and hasattr(
            self.env.unwrapped, "EPISODE_LEN_SEC"
        ):
            self.env.unwrapped.EPISODE_LEN_SEC = 5.0
        elif hasattr(self.env, "EPISODE_LEN_SEC"):
            self.env.EPISODE_LEN_SEC = 5.0

        # Paper DR ranges. Gravity bounds are identical on purpose.
        self.dr_params = dr_params or {
            "mass_range": (0.024, 0.030),
            "gravity_range": (9.81, 9.81),
            "wind_range": (-0.05, 0.05),
            "drag_range": (0.0001, 0.0010),
            "motor_noise_range": (0.0, 0.05),
        }
        self.current_mass = None
        self.current_gravity = None
        self.current_wind = np.zeros(3)
        self.current_drag = None
        self.current_motor_noise = None

    def _get_env_attr(self, attr_name):
        if hasattr(self.env, "unwrapped") and hasattr(
            self.env.unwrapped, attr_name
        ):
            return getattr(self.env.unwrapped, attr_name)
        if hasattr(self.env, attr_name):
            return getattr(self.env, attr_name)
        return None

    def _apply_domain_randomization(self):
        self.current_mass = np.random.uniform(*self.dr_params["mass_range"])
        self.current_gravity = np.random.uniform(*self.dr_params["gravity_range"])
        self.current_wind = np.random.uniform(
            *self.dr_params["wind_range"], size=3
        )
        self.current_drag = np.random.uniform(*self.dr_params["drag_range"])
        self.current_motor_noise = np.random.uniform(
            *self.dr_params["motor_noise_range"]
        )

        client = self._get_env_attr("CLIENT")
        if client is None:
            return
        p.setGravity(0, 0, -self.current_gravity, physicsClientId=client)
        drone_ids = self._get_env_attr("DRONE_IDS")
        if drone_ids is not None and len(drone_ids) > 0:
            p.changeDynamics(
                int(drone_ids[0]),
                -1,
                mass=self.current_mass,
                linearDamping=self.current_drag,
                angularDamping=self.current_drag * 0.1,
                physicsClientId=client,
            )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._apply_domain_randomization()
        info["dr_params"] = {
            "mass": self.current_mass,
            "gravity": self.current_gravity,
            "wind": self.current_wind,
            "drag": self.current_drag,
            "motor_noise": self.current_motor_noise,
        }
        return obs, info

    def step(self, action):
        # Paper DR: isotropic Gaussian on PWM, then clip to the gym Box.
        if self.current_motor_noise > 0:
            action = action + np.random.normal(
                0, self.current_motor_noise, size=action.shape
            )
            action = np.clip(action, -1.0, 1.0)

        # Additive wind in world frame; motor noise on the PWM action.
        if self.current_wind is not None and np.any(self.current_wind != 0):
            client = self._get_env_attr("CLIENT")
            drone_ids = self._get_env_attr("DRONE_IDS")
            if client is not None and drone_ids is not None and len(drone_ids) > 0:
                drone_id = int(drone_ids[0])
                pos, _ = p.getBasePositionAndOrientation(
                    drone_id, physicsClientId=client
                )
                p.applyExternalForce(
                    drone_id,
                    -1,
                    self.current_wind.tolist(),
                    pos,
                    p.WORLD_FRAME,
                    physicsClientId=client,
                )
        return self.env.step(action)


class TunableRewardAviary(HoverAviary):
    """CF2X hover with the paper reward weights (w_rpy=1.5, not the library 0.1).

    HoverAviary already supplies KIN observations with add_pd=True (28-D)
    and PWM actions. This subclass only changes the scalar reward and
    forces EPISODE_LEN_SEC = 5.
    """

    def __init__(
        self,
        reward_weights=None,
        integral_pos_z_limit=0.15,
        **kwargs,
    ):
        self.integral_pos_z_limit = integral_pos_z_limit
        self.reward_weights = {
            "w_pos": 2.5,
            "w_rpy": 1.5,
            "w_lin_vel": 0.05,
            "w_ang_vel": 0.05,
            "w_action": 0.1,
        }
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)
        super().__init__(**kwargs)
        self.EPISODE_LEN_SEC = 5.0

    def get_pos_error(self):
        pos_error_d = (self.pos[0] - self.last_pos) * self.CTRL_FREQ
        pos_error_i = np.clip(self.intergral_pos_e, -2, 2)
        pos_error_i[2] = np.clip(
            pos_error_i[2], -self.integral_pos_z_limit, self.integral_pos_z_limit
        )
        return pos_error_i, pos_error_d

    def _computeReward(self, verbose=False):
        # state[7:9] = (roll, pitch); yaw is excluded from the attitude penalty.
        state = self._getDroneStateVector(0)
        rew_pos = -self.reward_weights["w_pos"] * np.linalg.norm(
            self.goal - state[0:3]
        )
        rew_rpy = -self.reward_weights["w_rpy"] * np.linalg.norm(state[7:9])
        rew_lin_vel = -self.reward_weights["w_lin_vel"] * np.linalg.norm(
            state[10:13]
        )
        rew_ang_vel = -self.reward_weights["w_ang_vel"] * np.linalg.norm(
            state[13:16]
        )
        rew_action = -self.reward_weights["w_action"] * np.linalg.norm(
            (self.last_clipped_action[0] - self.HOVER_RPM) / self.MAX_RPM
        )
        self.rew_info = {
            "rew_pos": rew_pos,
            "rew_rpy": rew_rpy,
            "rew_lin_vel": rew_lin_vel,
            "rew_ang_vel": rew_ang_vel,
            "rew_action": rew_action,
        }
        return 2.0 + sum(self.rew_info.values())
