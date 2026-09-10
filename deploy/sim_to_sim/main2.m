%% setup_params.m - Automated Simulation and Testing Framework
clear;
clc;

%% 0. Test settings and thresholds

model_name = 'quadrotor_mellinger';

% Select the controller parameter file.
% param_file = 'Ablation_Seed_789_lambda_0.25_beta_0.0';
param_file = 'Ablation2_Seed_456_lambda_0.0_beta_1.0';
% param_file = 'Steady_Seed_456';
% param_file = 'SAC_Seed_1';
% param_file = 'DR_Seed_789';
% param_file = 'Predictor_Seed_789_batch_256_lambda_0.25_beta_4.0';

param_path = fullfile('parameter', [param_file, '.mat']);

% Performance criteria for constrained-gain experiments.
perf_rise_time = 5.0;
perf_overshoot = 5.0;
perf_settle_time = 10.0;

% Full-episode convergence thresholds.
rmse_threshold_xy = 0.40;
rmse_threshold_z = 0.15;

% Survival thresholds.
survive_z_min = -0.05;
survive_angle_max = deg2rad(80);

% Offline smoothing used only for noisy transient metrics.
noise_smoothing_time = 0.10;

% Fixed random seed for reproducible controller comparisons.
evaluation_random_seed = 2026;
rng(evaluation_random_seed, 'twister');

%% 1. Load controller parameters

if exist(param_path, 'file')
    load(param_path);
    fprintf('Loaded controller parameters from: %s\n', param_path);
else
    warning('Parameter file not found. Using hardcoded default gains.');

    Kp_lin = [0.4319; 0.4319; 2.5857];
    Ki_lin = [0.0027; 0.0027; 0.2500];
    Kd_lin = [0.3114; 0.3114; 1.1489];

    Kr_rot = [70000.0; 70000.0; 60000.0];
    Kw_rot = [20000.0; 20000.0; 12000.0];
    Ki_rot = [-1.2303; -1.2303; 494.9237];
end

%% 2. Simulation and physical settings

Ts = 1 / 240;
Tf = 30;

ori_mass = 0.027;
g = 9.81;

% Nominal inertia tensor [kg m^2].
ixx = 1.4e-5;
ixy = 0.0;
ixz = 0.0;
iyy = 1.4e-5;
iyz = 0.0;
izz = 2.17e-5;

J_base = [
    ixx, ixy, ixz;
    ixy, iyy, iyz;
    ixz, iyz, izz
];

% Payload is disabled in the reported evaluation.
mass_payload = 0.000;

% Hypothetical payload position relative to the nominal centre of mass.
% This has no effect when mass_payload is zero.
pos_payload = [0.005, 0.0, 0.010];

cf_mass = ori_mass + mass_payload;

[J_new, CoG_new] = calculate_loaded_inertia( ...
    J_base, ori_mass, mass_payload, pos_payload);

fprintf('=== Vehicle dynamic parameters ===\n');
fprintf('Total mass: %.3f kg\n', cf_mass);
fprintf( ...
    'Centre of mass [m]: [%.5f, %.5f, %.5f]\n', ...
    CoG_new(1), CoG_new(2), CoG_new(3));

disp('Inertia matrix [kg m^2]:');
disp(J_new);

inertia = J_new;

GRAVITY_FORCE = g * cf_mass;

% Motor and PWM parameters matched to the Python controller interface.
MassThrust = 132000;
MaxPWM = 65535;
MinPWM = 20000;
thrust2weight = 2.25;
PWM2RPM_SCALE = 0.2685;
PWM2RPM_CONST = 4070.3;
UINT16_MAX = 65535;

% Rotor force and moment coefficients.
kf = 3.16e-10;
km = 7.94e-12;

% Simulink mixer moment arm [m].
l = 0.028;

MaxRPM = sqrt(thrust2weight * GRAVITY_FORCE / (4 * kf));

Matrix_rpm_Ftau = [
     kf,        kf,       kf,       kf;
    -kf*l,     -kf*l,     kf*l,      kf*l;
    -kf*l,      kf*l,     kf*l,     -kf*l;
    -km,         km,      -km,         km
];

init_vel = [0, 0, 0];
init_ang = [0, 0, 0];

MAX_XY = 1.0;
MAX_Z = 1.0;
MAX_PITCH_ROLL = pi;
MAX_LIN_VEL_XY = 3;
MAX_LIN_VEL_Z = 1;
MAX_Omega = 10;

MIXER_MATRIX = [
    -0.5, -0.5, -1.0;
    -0.5,  0.5,  1.0;
     0.5,  0.5, -1.0;
     0.5, -0.5,  1.0
];

target_x_c = [1; 0; 0];
T_flip = eye(3);

%% 3. Sensor and environment configuration

WorkTemperature = 25;

% Wind perturbations are disabled.
Wind_x = 0;
Wind_y = 0;
Wind_z = 0;
Wind_phi = 0;
Wind_theta = 0;
Wind_psi = 0;

Speed_Sound = 340;
pressure = 101.3e3;
air_density = 1.229;

% Unity-gain first-order lag coefficients:
% H(s) = a / (s + a), with time constant tau = 1/a.
%
% These are pole rates in s^-1, not datasheet bandwidths in Hz.
delay_coef_motor = 50;
delay_coef_gyro = 230;
delay_coef_acc = 100;

%% 4. Pre-generate reproducible task conditions

num_tasks = 8;

% Tasks 1 and 2 use exactly the same initial position.
hover_pair_init = [0, 0, 1] + 0.02 * randn(1, 3);

task_init_pos = zeros(num_tasks, 3);
task_goal_pos = zeros(num_tasks, 3);
task_bias_gyro = zeros(num_tasks, 1);
task_bias_acc = zeros(num_tasks, 1);

for task_id = 1:num_tasks
    switch task_id
        case {1, 2}
            task_goal_pos(task_id, :) = [0, 0, 1];
            task_init_pos(task_id, :) = hover_pair_init;

        case {3, 4}
            task_goal_pos(task_id, :) = [0, 0, 1];
            task_init_pos(task_id, :) = [0, 0, 0.05];

        case {5, 6}
            task_goal_pos(task_id, :) = [0.1, 0.1, 1];
            task_init_pos(task_id, :) = [0, 0, 0.05];

        case {7, 8}
            task_goal_pos(task_id, :) = [0, 0, 1];
            task_init_pos(task_id, :) = [0, 0, 0.05];
    end

    % Even-numbered tasks include constant sensor biases.
    if mod(task_id, 2) == 0
        task_bias_gyro(task_id) = -1 + 2 * rand;
        task_bias_acc(task_id) = (-1 + 2 * rand) * 20 / 1000;
    end
end

fprintf('\n=== Fixed evaluation conditions ===\n');
fprintf('Random seed: %d\n', evaluation_random_seed);

fprintf( ...
    'Shared Task 1/2 initial position: [%.5f, %.5f, %.5f]\n', ...
    hover_pair_init(1), hover_pair_init(2), hover_pair_init(3));

for task_id = 2:2:num_tasks
    fprintf( ...
        'Task %d biases: gyro = %.6f, accelerometer = %.6f\n', ...
        task_id, task_bias_gyro(task_id), task_bias_acc(task_id));
end

%% 5. Automated task evaluation

perf_results = cell(1, num_tasks);
conv_results = cell(1, num_tasks);

clean_perf_passed = false(1, num_tasks);
clean_conv_passed = false(1, num_tasks);

fprintf('\nStarting %d task evaluations...\n', num_tasks);
fprintf('--------------------------------------------------\n');

for task_id = 1:num_tasks

    %% 5.1 Configure the current task

    measure_switch = mod(task_id, 2) == 0;
    complex_pulse = ismember(task_id, [7, 8]);

    init_pos = task_init_pos(task_id, :);
    goal_pos = task_goal_pos(task_id, :);
    bias_gyro = task_bias_gyro(task_id);
    bias_acc = task_bias_acc(task_id);

    % Explicit axis classification prevents hover tasks from being
    % misclassified as step responses.
    switch task_id
        case {1, 2}
            axis_mode = {'hover', 'hover', 'hover'};

        case {3, 4}
            axis_mode = {'hover', 'hover', 'step'};

        case {5, 6}
            axis_mode = {'step', 'step', 'step'};

        case {7, 8}
            axis_mode = {'pulse', 'pulse', 'pulse'};
    end

    % Export task variables to the base workspace for Simulink.
    assignin('base', 'init_pos', init_pos);
    assignin('base', 'goal_pos', goal_pos);
    assignin('base', 'measure_switch', measure_switch);
    assignin('base', 'complex_pulse', complex_pulse);
    assignin('base', 'bias_gyro', bias_gyro);
    assignin('base', 'bias_acc', bias_acc);

    fprintf('Running Task %d...\n', task_id);

    try
        %% 5.2 Run the Simulink model

        out = sim(model_name, 'ReturnWorkspaceOutputs', 'on');

        %% 5.3 Extract logged signals

        time = as_column(out.Z_to_T.time);

        ref_x = as_column(out.X_to_T.signals(1).values);
        act_x = as_column(out.X_to_T.signals(2).values);

        ref_y = as_column(out.Y_to_T.signals(1).values);
        act_y = as_column(out.Y_to_T.signals(2).values);

        ref_z = as_column(out.Z_to_T.signals(1).values);
        act_z = as_column(out.Z_to_T.signals(2).values);

        % Roll and pitch are required for survival evaluation.
        try
            act_roll = as_column(out.Phi_to_T.signals.values);
            act_pitch = as_column(out.Theta_to_T.signals.values);
        catch attitude_error
            error( ...
                'Required roll/pitch logs are unavailable: %s', ...
                attitude_error.message);
        end

        %% 5.4 Validate signal lengths and numerical values

        expected_length = length(time);

        signal_lengths = [
            length(ref_x), length(act_x), ...
            length(ref_y), length(act_y), ...
            length(ref_z), length(act_z), ...
            length(act_roll), length(act_pitch)
        ];

        if any(signal_lengths ~= expected_length)
            error('Logged signals do not have matching lengths.');
        end

        all_logged_values = [
            time;
            ref_x;
            act_x;
            ref_y;
            act_y;
            ref_z;
            act_z;
            act_roll;
            act_pitch
        ];

        if any(~isfinite(all_logged_values))
            error('One or more logged signals contain NaN or Inf.');
        end

        %% 5.5 Calculate raw trajectory RMSE

        rmse_x = sqrt(mean((act_x - ref_x).^2));
        rmse_y = sqrt(mean((act_y - ref_y).^2));
        rmse_z = sqrt(mean((act_z - ref_z).^2));

        %% 5.6 Evaluate survival

        if complex_pulse
            z_drop_tolerance = 0.5;
            survive_cond_z = all( ...
                act_z >= (ref_z - z_drop_tolerance));
        else
            survive_cond_z = all(act_z >= survive_z_min);
        end

        survive_cond_attitude = ...
            all(abs(act_roll) < survive_angle_max) && ...
            all(abs(act_pitch) < survive_angle_max);

        is_survived = survive_cond_z && survive_cond_attitude;

        %% 5.7 Evaluate convergence

        conv_xy = ...
            is_survived && ...
            (rmse_x < rmse_threshold_xy) && ...
            (rmse_y < rmse_threshold_xy);

        conv_z = ...
            is_survived && ...
            (rmse_z < rmse_threshold_z);

        %% 5.8 Select clean/noisy tolerances

        if measure_switch
            err_band = 0.075;
            abs_tol = 0.075;
        else
            err_band = 0.05;
            abs_tol = 0.05;
        end

        %% 5.9 Evaluate control performance

        if ~complex_pulse
            [pass_x, rt_x, os_x, st_x] = eval_perf( ...
                act_x, ref_x, time, axis_mode{1}, measure_switch, ...
                err_band, abs_tol, perf_rise_time, ...
                perf_overshoot, perf_settle_time, ...
                noise_smoothing_time);

            [pass_y, rt_y, os_y, st_y] = eval_perf( ...
                act_y, ref_y, time, axis_mode{2}, measure_switch, ...
                err_band, abs_tol, perf_rise_time, ...
                perf_overshoot, perf_settle_time, ...
                noise_smoothing_time);

            [pass_z, rt_z, os_z, st_z] = eval_perf( ...
                act_z, ref_z, time, axis_mode{3}, measure_switch, ...
                err_band, abs_tol, perf_rise_time, ...
                perf_overshoot, perf_settle_time, ...
                noise_smoothing_time);

            fprintf( ...
                ['  [X] Rise: %7.3f s | Overshoot: %7.2f%% | ' ...
                 'Settling: %7.3f s | Pass: %d\n'], ...
                rt_x, os_x, st_x, pass_x);

            fprintf( ...
                ['  [Y] Rise: %7.3f s | Overshoot: %7.2f%% | ' ...
                 'Settling: %7.3f s | Pass: %d\n'], ...
                rt_y, os_y, st_y, pass_y);

            fprintf( ...
                ['  [Z] Rise: %7.3f s | Overshoot: %7.2f%% | ' ...
                 'Settling: %7.3f s | Pass: %d\n'], ...
                rt_z, os_z, st_z, pass_z);

            % Performance requires both convergence and transient quality.
            perf_pass_xy = conv_xy && pass_x && pass_y;
            perf_pass_z = conv_z && pass_z;
        else
            % Pulse tasks use survival and RMSE criteria only.
            perf_pass_xy = conv_xy;
            perf_pass_z = conv_z;
        end

        %% 5.10 Enforce clean/noisy pair dependency

        if measure_switch
            base_task_id = task_id - 1;

            if ~clean_perf_passed(base_task_id)
                perf_pass_z = false;
                perf_pass_xy = false;

                fprintf( ...
                    ['  Noisy performance rejected because clean ' ...
                     'Task %d failed.\n'], ...
                    base_task_id);
            end

            if ~clean_conv_passed(base_task_id)
                conv_z = false;
                conv_xy = false;

                fprintf( ...
                    ['  Noisy convergence rejected because clean ' ...
                     'Task %d failed convergence.\n'], ...
                    base_task_id);
            end
        else
            clean_perf_passed(task_id) = ...
                perf_pass_z && perf_pass_xy;

            clean_conv_passed(task_id) = ...
                conv_z && conv_xy;
        end

        %% 5.11 Format performance result

        if perf_pass_z
            if perf_pass_xy
                perf_str = '+';
            elseif measure_switch
                perf_str = '-(xy)+(z)';
            else
                perf_str = '-';
            end
        else
            perf_str = '-';
        end

        perf_results{task_id} = ...
            sprintf('task%d%s', task_id, perf_str);

        %% 5.12 Format convergence result

        if conv_z
            if conv_xy
                conv_str = 'y';
            elseif measure_switch
                conv_str = 'n(xy)y(z)';
            else
                conv_str = 'n';
            end
        else
            conv_str = 'n';
        end

        conv_results{task_id} = ...
            sprintf('task%d%s', task_id, conv_str);

        %% 5.13 Print task summary

        fprintf( ...
            ['Task %d completed. Survival: %d | ' ...
             'RMSE X: %.3f m, Y: %.3f m, Z: %.3f m | ' ...
             'Performance: %s | Convergence: %s\n'], ...
            task_id, is_survived, ...
            rmse_x, rmse_y, rmse_z, ...
            perf_results{task_id}, conv_results{task_id});

    catch ME
        %% 5.14 Handle simulation or evaluation failure

        fprintf('\nSimulink/evaluation error: %s\n', ME.message);
        fprintf('Error identifier: %s\n', ME.identifier);

        for stack_id = 1:numel(ME.stack)
            fprintf( ...
                '  File: %s | Function: %s | Line: %d\n', ...
                ME.stack(stack_id).file, ...
                ME.stack(stack_id).name, ...
                ME.stack(stack_id).line);
        end

        fprintf( ...
            'Task %d aborted and recorded as failed.\n', ...
            task_id);

        perf_results{task_id} = sprintf('task%d-', task_id);
        conv_results{task_id} = sprintf('task%dn', task_id);

        fprintf( ...
            'Task %d completed with error. RMSE recorded as Inf.\n', ...
            task_id);
    end
end

%% 6. Print final report

fprintf('\n==================================================\n');
fprintf('Controller configuration: %s\n', param_file);
fprintf('Performance results:\n%s\n', strjoin(perf_results, ' '));
fprintf('Convergence results:\n%s\n', strjoin(conv_results, ' '));
fprintf('==================================================\n');

%% Local helper functions

function [pass, rt, os, st] = eval_perf( ...
    act, ref, time, axis_mode, is_noisy, ...
    err_band, abs_tol, perf_rt_limit, ...
    perf_os_limit, perf_st_limit, ...
    noise_smoothing_time)

    % Evaluate one trajectory axis.
    %
    % Noisy transient metrics use an offline centred moving average.
    % Raw trajectories remain in use for RMSE and survival.

    act = as_column(act);
    ref = as_column(ref);
    time = as_column(time);

    if is_noisy
        dt = median(diff(time));

        if ~isfinite(dt) || dt <= 0
            error('Invalid time vector supplied to eval_perf.');
        end

        window_samples = max( ...
            1, round(noise_smoothing_time / dt));

        act_eval = movmean( ...
            act, window_samples, ...
            'Endpoints', 'shrink');
    else
        act_eval = act;
    end

    switch lower(axis_mode)

        case 'hover'
            % Rise time and percentage overshoot are not meaningful
            % for near-hover regulation.
            rt = 0;
            os = 0;

            error_abs = abs(act_eval - ref);
            outside_indices = find(error_abs > abs_tol);

            if isempty(outside_indices)
                st = 0;
            elseif outside_indices(end) >= length(time)
                st = inf;
            else
                st = time(outside_indices(end) + 1);
            end

            pass = st <= perf_st_limit;

        case 'step'
            % stepinfo uses the raw clean trajectory or the offline-smoothed
            % noisy trajectory.
            si = stepinfo( ...
                act_eval, time, ref(end), ...
                'SettlingTimeThreshold', err_band);

            rt = si.RiseTime;
            os = si.Overshoot;
            st = si.SettlingTime;

            if ~isfinite(rt)
                rt = inf;
            end

            if ~isfinite(os)
                os = inf;
            end

            if ~isfinite(st)
                st = inf;
            end

            pass = ...
                (rt <= perf_rt_limit) && ...
                (os <= perf_os_limit) && ...
                (st <= perf_st_limit);

        otherwise
            error( ...
                'Unsupported axis evaluation mode: %s', ...
                axis_mode);
    end
end

function output = as_column(input)

    % Convert a logged signal to a one-dimensional column vector.

    output = squeeze(input);
    output = output(:);
end

function [J_total, CoG_new] = calculate_loaded_inertia( ...
    J_base, m_base, mass_payload, pos_payload)

    % Calculate the combined centre of mass and inertia tensor.

    CoG_base = [0, 0, 0];
    total_mass = m_base + mass_payload;

    if total_mass <= 0
        error('Total mass must be positive.');
    end

    CoG_new = ...
        (m_base * CoG_base + mass_payload * pos_payload) ...
        / total_mass;

    d_base = CoG_base - CoG_new;

    J_base_shifted = shift_inertia_matrix( ...
        J_base, m_base, d_base);

    d_payload = pos_payload - CoG_new;

    J_payload_shifted = shift_inertia_matrix( ...
        zeros(3, 3), mass_payload, d_payload);

    J_total = J_base_shifted + J_payload_shifted;
end

function J_shifted = shift_inertia_matrix(J_orig, mass, d)

    % Apply the parallel-axis theorem.

    dx = d(1);
    dy = d(2);
    dz = d(3);

    J_shift = zeros(3, 3);

    J_shift(1, 1) = mass * (dy^2 + dz^2);
    J_shift(2, 2) = mass * (dx^2 + dz^2);
    J_shift(3, 3) = mass * (dx^2 + dy^2);

    J_shift(1, 2) = -mass * dx * dy;
    J_shift(2, 1) = J_shift(1, 2);

    J_shift(1, 3) = -mass * dx * dz;
    J_shift(3, 1) = J_shift(1, 3);

    J_shift(2, 3) = -mass * dy * dz;
    J_shift(3, 2) = J_shift(2, 3);

    J_shifted = J_orig + J_shift;
end