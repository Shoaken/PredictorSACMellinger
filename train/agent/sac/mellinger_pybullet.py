"""Differentiable Crazyflie Mellinger / DSLPID actor used by all paper methods.

The policy mean is the firmware-style cascaded controller (position PID ->
desired attitude -> attitude PID -> mixer -> PWM). SAC entropy comes from a
separate MLP ``trunk`` that outputs log_std only; those weights are *not*
firmware gains and are frozen in STEADY stage 2.

Observation (HoverAviary add_pd=True, 28-D), matching mellinger_control::

    [0:3]   world xyz              goal is (0, 0, 1)
    [3:7]   quaternion (xyzw)
    [7:10]  roll, pitch, yaw
    [10:13] linear velocity
    [13:16] angular velocity
    [16:19] integral position error (z clipped to +/- 0.15)
    [19:22] derivative position error
    [22:25] integral rotation error (ctrlMel.i_err_m)
    [25:28] derivative attitude error

Action: 4 motor commands in [MIN_PWM, MAX_PWM] / MAX_PWM, i.e. about
[0.305, 1]. The gym Box is [-1, 1]; select_action clamps to that box.

Learned PID (paper projection after each actor step):
    kp_z >= 0.9
    ki_z in [0, i_range_z=0.25]
    kd_z >= 2 * zeta_z * sqrt(m * kp_z)   zeta_z = 1.361
    ki_m_xy, ki_m_z clipped to attitude I ranges
Attitude P/D (kR, kw) start frozen in sim training; STEADY may unfreeze kR/kw.

PWM/RPM constants (PWM2RPM_SCALE, massThrust, MIXER_MATRIX, KF, ...) are
kept as firmware reference. Do not delete them when reproducing hardware.
"""
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from train.utils import util
from train.agent.sac.actor import SquashedNormal

class DifferentiableMellinger_pybullet(nn.Module):
    # Crazyflie 2.x firmware constants (see Bitcraze Mellinger / DSLPID).
    cf_mass = 0.027
    massThrust = 132000
    INT16_MAX = 65536
    PWM2RPM_SCALE = 0.2685
    PWM2RPM_CONST = 4070.3
    MIN_PWM = 20000
    MAX_PWM = 65535
    GRAVITY = 9.81 * cf_mass
    KF = 3.16e-10

    def __init__(self, max_rpm=21600, ctrl_freq : int = 240, output = "pwm"):
        """

        Args:
            max_rpm: firmware RPM cap; unused when output="pwm" (paper default)
            ctrl_freq: 240 Hz, same as HoverAviary CTRL_FREQ
            output: "pwm" for the paper (normalized by MAX_PWM); "rpm" for debug
        """
        super().__init__()
        self.CTRL_FREQ = ctrl_freq
        self.MAX_RPM = max_rpm
        self.integral_error = torch.zeros([3, ])

        self.kp_xy = torch.nn.Parameter(torch.tensor(0.4), requires_grad=True)
        self.ki_xy = torch.nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.kd_xy = torch.nn.Parameter(torch.tensor(0.2), requires_grad=True)
        # Z position
        self.kp_z = torch.nn.Parameter(torch.tensor(1.25), requires_grad=True)
        self.ki_z = torch.nn.Parameter(torch.tensor(0.05), requires_grad=True)
        self.kd_z = torch.nn.Parameter(torch.tensor(0.6), requires_grad=True)
        # Attitude (firmware roles: kR=P, kw=D, ki_m=I)
        self.kR_xy = torch.nn.Parameter(torch.tensor(70000.), requires_grad=False)
        self.kw_xy = torch.nn.Parameter(torch.tensor(20000.), requires_grad=False)
        self.ki_m_xy = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.kR_z = torch.nn.Parameter(torch.tensor(60000.), requires_grad=False)
        self.kw_z = torch.nn.Parameter(torch.tensor(12000.), requires_grad=False)
        self.ki_m_z = torch.nn.Parameter(torch.tensor(500.), requires_grad=True)

        self.i_range_xy = 0.1
        self.i_range_z = 0.25
        self.d_range_z = 0.7
        self.zeta_z = 1.361
        self.att_i_range_xy = 5.0
        self.att_i_range_z = 600.

        self.register_buffer("goal", torch.tensor([0., 0., 1.]))
        # CF2X mixer: rows = motors m1..m4, columns = (roll, pitch, yaw) torque.
        self.register_buffer("MIXER_MATRIX", torch.tensor([
            [-.5, -.5, -1], [-.5, .5, 1], [.5, .5, -1], [.5, -.5, 1]
        ]).float())
        self.register_buffer("target_x_c", torch.tensor([1., 0., 0.]))
        self.register_buffer("gravity", torch.tensor([0, 0, self.GRAVITY]))
        self.register_buffer("target_rpy_rates", torch.zeros(3))        


        self.output_type = output
        self.reset()

        # SAC log_std network. Not a PID gain; freeze it in STEADY stage 2.
        self.trunk = util.mlp(28, 256, 4,
                              2, hidden_activation=nn.ELU(inplace=True))

        self.log_std_bounds=[-20., 1.]

        def transpose(x):
            return x.T
        self.vec_transpose = torch.vmap(transpose)

    def projection_on_gains(self):
        """Paper feasible-gain set: kp_z floor, ki_z box, kd_z damping ratio."""
        with torch.no_grad():
            # self.ki_xy.clamp_(min=0., max=self.i_range_xy)
            self.kp_z.clamp_(min=0.9,)
            # self.kp_z.clamp_(min=0.2,)
            self.ki_z.clamp_(min=0., max=self.i_range_z)
            kd_z_min = 2.0 * self.zeta_z * torch.sqrt(self.cf_mass * self.kp_z)
            self.kd_z.clamp_(min=kd_z_min)
            # self.kd_z.clamp_(min=self.d_range_z,)
            self.ki_m_xy.clamp_(min=-self.att_i_range_xy, max=self.att_i_range_xy)
            self.ki_m_z.clamp_(min=0., max=self.att_i_range_z)

    def get_controller_parameters_dict(self):

        return self.state_dict()


    def set_device(self, device):
        self.MIXER_MATRIX = self.MIXER_MATRIX.to(device)
        self.goal = self.goal.to(device)
        self.target_x_c = self.target_x_c.to(device)
        self.gravity = self.gravity.to(device)
        self.target_rpy_rates = self.target_rpy_rates.to(device)
        self.last_rpy = self.last_rpy.to(device)
        self.last_pos_e = self.last_pos_e.to(device)
        self.integral_pos_e = self.integral_pos_e.to(device)
        self.last_rpy_e = self.last_rpy_e.to(device)
        self.integral_rpy_e = self.integral_rpy_e.to(device)


    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part last,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        i, j, k, r = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def _index_from_letter(self, letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")

    def _angle_from_tan(self,
            axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
    ) -> torch.Tensor:
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.

        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.

        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def matrix_to_euler_angles(self, matrix: torch.Tensor, convention = "XYZ") -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to Euler angles in radians.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.

        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = self._index_from_letter(convention[0])
        i2 = self._index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            self._angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            self._angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)


    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """

        current_device = self.goal.device
        self.last_rpy = torch.zeros(3).to(current_device)
        self.last_pos_e = torch.zeros(1, 3).to(current_device)
        self.integral_pos_e = torch.zeros(1, 3).to(current_device)
        self.last_rpy_e = torch.zeros(3).to(current_device)
        self.integral_rpy_e = torch.zeros(3).to(current_device)

    def mellinger_control(self, obs):
        """Deterministic Mellinger mean (PWM or RPM). SAC samples around this.

        Integral/derivative errors are *read from the 28-D observation*, not
        from the internal caches last_* / integral_* (those stay for firmware
        reference / reset). Mixing uses MIXER_MATRIX then clips to PWM limits.
        """

        P_COEFF_FOR = torch.stack([self.kp_xy, self.kp_xy, self.kp_z])
        I_COEFF_FOR = torch.stack([self.ki_xy, self.ki_xy, self.ki_z])
        D_COEFF_FOR = torch.stack([self.kd_xy, self.kd_xy, self.kd_z])
        P_COEFF_TOR = torch.stack([self.kR_xy, self.kR_xy, self.kR_z])
        # Attitude: firmware roles kR=P, kw=D, ki_m=I
        I_COEFF_TOR = torch.stack([self.ki_m_xy, self.ki_m_xy, self.ki_m_z])
        D_COEFF_TOR = torch.stack([self.kw_xy, self.kw_xy, self.kw_z])

        # position control
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        pos_e = self.goal - obs[:, 0:3]
        cur_quat = obs[:, 3:7]
        cur_rpy = obs[:, 7:10]
        cur_vel = obs[:, 10:13]
        integral_pos_error = obs[:, 16:19]
        diff_pos_error = obs[:, 19:22]
        integral_rpy_error = obs[:, 22:25]
        # Observation stores attitude-rate error; firmware D term uses -omega.
        diff_rpy_error = -1 * obs[:, 25:28]
        cur_rotation = self.quaternion_to_matrix(cur_quat)
        vel_e = - cur_vel
        # self.integral_pos_e = self.integral_pos_e + pos_e * 1 / 240
        # self.integral_pos_e = torch.clip(self.integral_pos_e, -2., 2.)
        # self.integral_pos_e[:, 2] = torch.clip(self.integral_pos_e[:, 2], -0.15, .15)
        #### PID target thrust #####################################
        target_thrust = torch.multiply(P_COEFF_FOR, pos_e) \
                        + torch.multiply(I_COEFF_FOR, integral_pos_error) \
                        + torch.multiply(D_COEFF_FOR, vel_e) + self.gravity  # , device=self.de
        scalar_thrust = torch.clamp(torch.vmap(torch.inner)(target_thrust, cur_rotation[:, :, 2]), 0, torch.inf)
        # thrust_pwm = (torch.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        thrust_pwm = self.massThrust * scalar_thrust
        target_z_ax = F.normalize(target_thrust, dim=1)
        # target_x_c = torch.tensor([1., 0., 0.]) # assume target rpy always 0
        target_y_ax = F.normalize(torch.vmap(torch.cross, in_dims=(0, None))(target_z_ax, self.target_x_c),
                                  dim=1)  # / torch.norm(torch.cross(target_z_ax, target_x_c))
        target_x_ax = torch.vmap(torch.cross)(target_y_ax, target_z_ax)
        target_rotation_transposed = torch.stack([target_x_ax, target_y_ax, target_z_ax], dim=1)
        target_rotation = torch.permute(target_rotation_transposed, [0, 2, 1])
        #### Target rotation #######################################
        target_euler = self.matrix_to_euler_angles(target_rotation)

        # Altitude control
        # cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        # target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat()
        # w, x, y, z = target_quat
        # target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()

        rot_matrix_e = (torch.matmul(self.vec_transpose(target_rotation), cur_rotation)
                        - torch.matmul(self.vec_transpose(cur_rotation), target_rotation))
        rot_e = torch.stack([rot_matrix_e[:, 2, 1], rot_matrix_e[:, 0, 2], rot_matrix_e[:, 1, 0]], dim=1)
        #print(rot_e)
        #self.rot_e = rot_e
        #rpy_rates_e = self.target_rpy_rates - (cur_rpy - self.last_rpy) * 240
        #self.last_rpy = cur_rpy
        #integral_rpy_e = self.integral_rpy_e
        #integral_rpy_e = integral_rpy_e - rot_e / 240
        #integral_rpy_e = torch.clip(integral_rpy_e, -1500., 1500.)
        #integral_rpy_e[0:2] = torch.clip(integral_rpy_e[0:2], -1., 1.)
        #self.integral_rpy_e = integral_rpy_e

        #self.integral_rpy_e = self.integral_rpy_e - rot_e / 240
        #self.integral_rpy_e = torch.clip(self.integral_rpy_e, -1500., 1500.)
        #self.integral_rpy_e[0:2] = torch.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - torch.multiply(P_COEFF_TOR, rot_e) \
                         + torch.multiply(D_COEFF_TOR, diff_rpy_error) \
                         + torch.multiply(I_COEFF_TOR, integral_rpy_error)
        target_torques = torch.clip(target_torques, -32000, 32000)
        #print(target_torques)
        pwm = thrust_pwm.unsqueeze(1) + torch.matmul(self.MIXER_MATRIX, target_torques.unsqueeze(2)).squeeze()
        pwm = torch.clip(pwm, self.MIN_PWM, self.MAX_PWM)  # .squeeze(dim=-1)
        # Paper uses PWM. rpm = PWM2RPM_SCALE * pwm + PWM2RPM_CONST is firmware.
        if self.output_type == "pwm":
            return pwm / self.MAX_PWM
        elif self.output_type == "rpm":
            return (self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST) / self.MAX_RPM
        else:
            raise ValueError(f"Invalid output type {self.output_type}.")


    def forward(self, obs):
        """SquashedNormal(mean=Mellinger PWM, std=exp(trunk(obs)))."""
        obs = obs.to(self.goal.device)
        control = self.mellinger_control(obs)
        log_std = self.trunk(obs)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()
        dist = SquashedNormal(control, std)
        return dist



