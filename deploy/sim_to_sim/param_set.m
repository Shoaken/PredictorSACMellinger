%% param_set.m
% Set Mellinger PID gains and a single initial/goal pose, then run
% quadrotor_mellinger.slx to inspect that controller interactively.
% Uncomment (or paste) one gain block below. Gains can be copied from
% scripts/pth_reader.py. This script does not sweep the 8 evaluation
% tasks; use main2.m for batch success/convergence reporting.
clear; clc;

%% --- 1. Simulation setting --- (not really used)
Ts = 1/240;              % Sample time (matches ctrl_freq=240)
Tf = 30;                 % Simulation length (s)
Fs = 1/Ts;
%% --- 2. Physical parameters (Crazyflie 2.x) ---
ori_mass = 0.027;         % kg
g = 9.81;

ixx=1.4e-5;
ixy=0.0;
ixz=0.0;
iyy=1.4e-5;
iyz=0.0;
izz=2.17e-5;

J_base = [
    ixx, ixy, ixz;
    ixy, iyy, iyz;
    ixz, iyz, izz
    ];

% 2. Payload parameters
mass_payload = 0.000; % payload mass (kg); e.g. 0.015 = 15 g
% Assume payload mounted 10 mm above CoG and offset 5 mm toward nose (X)
pos_payload = [0.005, 0.0, 0.010];
cf_mass = ori_mass + mass_payload;

% ===== Core inertia calculation =====
[J_new, CoG_new] = calculate_loaded_inertia(J_base, ori_mass, mass_payload, pos_payload);

% Print results
fprintf('=== Dynamics after payload mount ===\n');
fprintf('Total mass: %.3f kg\n', cf_mass);
fprintf('New CoG (m): [%.5f, %.5f, %.5f]\n', CoG_new(1), CoG_new(2), CoG_new(3));
disp('New inertia matrix J_new (kg*m^2):');
disp(J_new);

inertia = J_new;

GRAVITY_FORCE = g * cf_mass;
MassThrust = 132000;     % Python: massThrust
MaxPWM = 65535;
MinPWM = 20000;
thrust2weight=2.25;
PWM2RPM_SCALE = 0.2685;
PWM2RPM_CONST = 4070.3;
UINT16_MAX = 65535;

kf = 3.16e-10;
km = 7.94e-12;
l = 0.028; % x or y arm length from URDF
MaxRPM=sqrt(thrust2weight*GRAVITY_FORCE/4/kf);

Matrix_rpm_Ftau = [
    % M1(RR)   M2(FR)   M3(FL)   M4(RL)
     kf,        kf,      kf,      kf; % Thrust
    -kf*l,     -kf*l,    kf*l,    kf*l; % Roll (Right-, Left+)
    -kf*l,      kf*l,    kf*l,   -kf*l; % Pitch (Back-, Front+)
    -km,        km,     -km,      km % Yaw
];

%% --- 3. Initialization ---
init_pos = [0, 0, 0.05];    % Initial position (m)
% init_pos = [0.05, -0.05, 0.98];
init_vel = [0, 0, 0];    % Initial velocity (m/s)
init_ang = [0, 0, 0];    % Initial Euler angles (rad) [Roll, Pitch, Yaw]
% goal_pos = [0, 0, 0.4];    % Goal position
goal_pos = [0, 0.01, 1];
% goal_pos = [0.1, 0.1, 1];
% init_pos = goal_pos + 0.02 * randn(1, 3);

%% --- 4. Observation normalization & controller settings ---
MAX_XY = 1.0;
MAX_Z = 1.0;
MAX_PITCH_ROLL = pi;
MAX_LIN_VEL_XY = 3;
MAX_LIN_VEL_Z = 1;
MAX_Omega = 10;

% --- Mellinger PID ---
% Firmware roles: Kr_rot = P (kR), Kw_rot = D (kw), Ki_rot = I (ki_m)
target_x_c = [1; 0; 0];

% Kp_lin = [0.4; 0.4; 1.25];
% Ki_lin = [0.05; 0.05; 0.05];
% Kd_lin = [0.2; 0.2; 0.4];
% Kr_rot = [70000.0; 70000.0; 60000.0];    % P (kR)
% Kw_rot = [20000.0; 20000.0; 12000.0];    % D (kw)
% Ki_rot = [0; 0; 500];                    % I (ki_m)

% Kp_lin = [0.1000; 0.1000; 0.3000];
% Ki_lin = [0.0000; 0.0000; 0.0000];
% Kd_lin = [0.0500; 0.0500; 0.1000];
% Kr_rot = [30000.0; 30000.0; 30000.0];    % P (kR)
% Kw_rot = [8000.0; 8000.0; 5000.0];       % D (kw)
% Ki_rot = [0.0000; 0.0000; 0.0000];       % I (ki_m)

%------------------------------------------
% no-Normalized Actor (bound_perfect, frozen attitude PD)
% Predictor

% seed_1
% Kp_lin = [0.1785; 0.1785; 1.9391];
% Ki_lin = [0.0007; 0.0007; 0.2291];
% Kd_lin = [0.1914; 0.1914; 0.6228];
% Kr_rot = [70000.00; 70000.00; 60000.00];    % P term
% Kw_rot = [20000.00; 20000.00; 12000.00];    % D term (Note: based on your keys)
% Ki_rot = [-4.4154; -4.4154; 506.9823];    % I term

% seed_42 
% Kp_lin = [0.1688; 0.1688; 1.4604];
% Ki_lin = [0.0075; 0.0075; 0.2500];
% Kd_lin = [0.2826; 0.2826; 0.5405];
% Kr_rot = [70000.00; 70000.00; 60000.00];    % P term
% Kw_rot = [20000.00; 20000.00; 12000.00];    % D term (Note: based on your keys)
% Ki_rot = [-5.0000; -5.0000; 511.1072];    % I term

% seed_123 
% Kp_lin = [0.2020; 0.2020; 2.3473];
% Ki_lin = [0.0084; 0.0084; 0.2500];
% Kd_lin = [0.2034; 0.2034; 0.6993];
% Kr_rot = [70000.00; 70000.00; 60000.00];    % P term
% Kw_rot = [20000.00; 20000.00; 12000.00];    % D term (Note: based on your keys)
% Ki_rot = [-3.3311; -3.3311; 504.7868];    % I term

% seed_456
% Kp_lin = [0.2479; 0.2479; 2.2161];
% Ki_lin = [0.1638; 0.1638; 0.2458];
% Kd_lin = [0.3546; 0.3546; 0.7350];
% Kr_rot = [70000.00; 70000.00; 60000.00];    % P term
% Kw_rot = [20000.00; 20000.00; 12000.00];    % D term (Note: based on your keys)
% Ki_rot = [-0.3578; -0.3578; 500.9348];    % I term
% seed_789
% Kp_lin = [0.2662; 0.2662; 3.1925];
% Ki_lin = [0.2661; 0.2661; 0.2500];
% Kd_lin = [0.2430; 0.2430; 0.7992];
% Kr_rot = [70000.00; 70000.00; 60000.00];    % P term
% Kw_rot = [20000.00; 20000.00; 12000.00];    % D term (Note: based on your keys)
% Ki_rot = [-4.4010; -4.4010; 498.7159];    % I term


% seed1
% % Predictor_Reward_wP2.5_wR1.5_wLV0.05_wAV0.05_wA0.1_wSm0.0_wOS0.0(KIbound-0.25)
Kp_lin = [0.4319; 0.4319; 2.5857];
Ki_lin = [0.0027; 0.0027; 0.2500];
Kd_lin = [0.3114; 0.3114; 1.1489];
Kr_rot = [70000.0; 70000.0; 60000.0];    % P (kR)
Kw_rot = [20000.0; 20000.0; 12000.0];    % D (kw)
Ki_rot = [-1.2303; -1.2303; 494.9237];   % I (ki_m)





%***
MIXER_MATRIX = [ ...
    -0.5, -0.5, -1.0; ...
    -0.5,  0.5,  1.0; ...
     0.5,  0.5, -1.0; ...
     0.5, -0.5,  1.0 ];

T_flip = [1,  0,  0; ...
          0,  1,  0; ...
          0,  0,  1];

%% --- 5. Sensor & environment config ---
measure_switch = 0; % set =1 to enable, =0 to disable
complex_pulse = 0;  % set =1 to enable, =0 to disable
% K_gyro = 131;
%K_acc = ;
bias_gyro = (-1+2*rand)*1;
bias_acc = (-1+2*rand)*20/1000;

% bias_gyro = 0;
% bias_acc = 0;

delay_coef_motor = 50;   % tau_motor = 20 ms
delay_coef_gyro  = 230;  % tau_gyro  = 4.35 ms
delay_coef_acc   = 100;  % tau_acc   = 10 ms

% delay_coef_motor = 50*2*pi;
% delay_coef_gyro = 80*2*pi;
% delay_coef_acc = 80*2*pi;

% % Firmware-equivalent sensor filters
% [b_gyro, a_gyro] = butter(2, 80/(Fs/2));
% [b_acc,  a_acc]  = butter(2, 30/(Fs/2));
% % Identified Crazyflie motor lag
% tau_motor = 0.072;
% alpha_motor = exp(-Ts/tau_motor);
% b_motor = 1-alpha_motor;
% a_motor = [1, -alpha_motor];

WorkTemperature = 25;
Wind_x = 0;
Wind_y = 0;
Wind_z = 0;
Wind_phi = 0;
Wind_theta = 0;
Wind_psi = 0;

Speed_Sound = 340;
pressure = 101.3e3;
air_density=1.229;

function [J_total, CoG_new] = calculate_loaded_inertia(J_base, m_base, m_payload, pos_payload)
    % Assume the original body CoG is at the origin [0, 0, 0]
    CoG_base = [0, 0, 0];

    % 1. Compute the new system center of gravity
    m_total = m_base + m_payload;
    CoG_new = (m_base * CoG_base + m_payload * pos_payload) / m_total;

    % 2. Shift the body inertia matrix to the new CoG
    d_base = CoG_base - CoG_new;
    J_base_shifted = shift_inertia_matrix(J_base, m_base, d_base);

    % 3. Payload treated as a point mass about the new CoG
    d_payload = pos_payload - CoG_new;
    % Point-mass intrinsic inertia is zero; only the parallel-axis term remains
    J_payload_shifted = shift_inertia_matrix(zeros(3,3), m_payload, d_payload);

    % 4. Superpose to obtain the total inertia about the new CoG
    J_total = J_base_shifted + J_payload_shifted;
end

% ==========================================
% Math helper: parallel-axis theorem
% ==========================================
function J_shifted = shift_inertia_matrix(J_orig, mass, d)
    % J_orig: original inertia matrix
    % mass: mass
    % d: vector from the new CoG to the body's original CoG [dx, dy, dz]

    dx = d(1);
    dy = d(2);
    dz = d(3);

    % Parallel-axis contribution
    J_shift = zeros(3, 3);

    % Diagonal terms (moments of inertia increase)
    J_shift(1, 1) = mass * (dy^2 + dz^2);
    J_shift(2, 2) = mass * (dx^2 + dz^2);
    J_shift(3, 3) = mass * (dx^2 + dy^2);

    % Off-diagonal terms (products of inertia; note the minus sign)
    J_shift(1, 2) = -mass * (dx * dy);
    J_shift(2, 1) = J_shift(1, 2);

    J_shift(1, 3) = -mass * (dx * dz);
    J_shift(3, 1) = J_shift(1, 3);

    J_shift(2, 3) = -mass * (dy * dz);
    J_shift(3, 2) = J_shift(2, 3);

    % Inertia after the shift
    J_shifted = J_orig + J_shift;
end