%% FlyReferenceMellinger.m
% Hardware flight helper for Crazyflie 2.1 + Loco Positioning Deck.
% Run this file from deploy/sim_to_real/.
%
% 1. Edit the reference (xref, yref, zref, yaw) and fixedFrequency.
% 2. Paste Mellinger gains from scripts/pth_reader.py (or keep defaults).
% 3. Set crazyflieUri to the same radio URI as FlightLocoMellinger.py.
% 4. Set pythonExecutable to a Python that has cflib installed.
% 5. Run this script: it writes trajectory.csv, applies gains over the
%    radio, launches FlightLocoMellinger.py, then plots state_log.csv.
%
% Firmware target: Crazyflie 2026.04. This script does not persist gains
% in flash; they are runtime parameters for the next flight.

format long;
% clear; close all; clc;

%% Reference sampled at fixedFrequency
fixedFrequency = 50;                 % Hz
Ts = 1/fixedFrequency;
tsampled = (0:Ts:13)';

% Rising helix reference
xref = 0*tsampled;%0.5 - 0.5*cos(1.0*tsampled);
yref = 0*tsampled;%0.5*sin(1.0*tsampled);
zref = 0*tsampled;%0.05*tsampled;
yawref_deg = zeros(size(tsampled));

%% Write the reference for Python
trajectoryTable = table(tsampled, xref, yref, zref, yawref_deg, ...
    'VariableNames', {'timestamp','x','y','z','yaw'});
writetable(trajectoryTable, 'trajectory.csv');

%% Mellinger gains for this flight

Kp_lin = [0.4; 0.4; 1.25];        %%% DEFAULT VALUES
Ki_lin = [0.05; 0.05; 0.05];
Kd_lin = [0.2; 0.2; 0.4*2];

Kr_rot = [70000.00; 70000.00; 60000.00];
Kw_rot = [20000.00; 20000.00; 12000.00];
Ki_rot = [0.0; 0.0; 500.0];

% Kp_lin = [0.1785; 0.1785; 1.9391];
% Ki_lin = [0.0007; 0.0007; 0.2291];
% Kd_lin = [0.1914; 0.1914; 0.6228];
% 
% Kr_rot = [70000.00; 70000.00; 60000.00];
% Kw_rot = [20000.00; 20000.00; 12000.00];
% Ki_rot = [-4.4154; -4.4154; 506.9823];

% Kp_lin = [0.4319; 0.4319; 2.5857];
% Ki_lin = [0.0027; 0.0027; 0.2500];
% Kd_lin = [0.3114; 0.3114; 1.1489];
% Kr_rot = [70000.0; 70000.0; 60000.0];    % P (kR)
% Kw_rot = [20000.0; 20000.0; 12000.0];    % D (kw)
% Ki_rot = [-1.2303; -1.2303; 494.9237];   % I (ki_m)

% Crazyflie firmware: 2026.04
% Must match the URI used by FlightLocoMellinger.py. Replace with yours.
crazyflieUri = 'radio://0/80/2M/E7E7E7E7E7';

%% Run one flight
pythonExecutable = 'python';  % Replace with a local python that has cflib.
pythonScript = 'FlightLocoMellinger.py';

fprintf('\nPrepared one %.2f s reference flight with %d samples at %.1f Hz.\n', ...
    tsampled(end), numel(tsampled), 1/Ts);
fprintf('Verify the Loco system, clear the flight area, and place the CF2.1 at the launch point.\n');
input('Press Enter to start the Python flight script, or Ctrl+C to cancel: ', 's');

% Apply Mellinger gains as runtime parameters before the flight

gainSetterScript = [tempname, '.py'];
fid = fopen(gainSetterScript, 'w');
if fid < 0
    error('Could not create temporary Python gain-setter script.');
end

fprintf(fid, 'import time\n');
fprintf(fid, 'import cflib.crtp\n');
fprintf(fid, 'from cflib.crazyflie import Crazyflie\n');
fprintf(fid, 'from cflib.crazyflie.syncCrazyflie import SyncCrazyflie\n');
fprintf(fid, 'uri = r''%s''\n', crazyflieUri);
fprintf(fid, 'gains = {\n');
fprintf(fid, '    ''ctrlMel.kp_xy'': %.10g,\n', Kp_lin(1));
fprintf(fid, '    ''ctrlMel.ki_xy'': %.10g,\n', Ki_lin(1));
fprintf(fid, '    ''ctrlMel.kd_xy'': %.10g,\n', Kd_lin(1));
fprintf(fid, '    ''ctrlMel.kp_z'': %.10g,\n', Kp_lin(3));
fprintf(fid, '    ''ctrlMel.ki_z'': %.10g,\n', Ki_lin(3));
fprintf(fid, '    ''ctrlMel.kd_z'': %.10g,\n', Kd_lin(3));
fprintf(fid, '    ''ctrlMel.kR_xy'': %.10g,\n', Kr_rot(1));
fprintf(fid, '    ''ctrlMel.kR_z'': %.10g,\n', Kr_rot(3));
fprintf(fid, '    ''ctrlMel.kw_xy'': %.10g,\n', Kw_rot(1));
fprintf(fid, '    ''ctrlMel.kw_z'': %.10g,\n', Kw_rot(3));
fprintf(fid, '    ''ctrlMel.ki_m_xy'': %.10g,\n', Ki_rot(1));
fprintf(fid, '    ''ctrlMel.ki_m_z'': %.10g,\n', Ki_rot(3));
fprintf(fid, '}\n');
fprintf(fid, 'cflib.crtp.init_drivers(enable_debug_driver=False)\n');
fprintf(fid, 'with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:\n');
fprintf(fid, '    cf = scf.cf\n');
fprintf(fid, '    t0 = time.monotonic()\n');
fprintf(fid, '    while not cf.param.is_updated:\n');
fprintf(fid, '        if time.monotonic() - t0 > 10.0:\n');
fprintf(fid, '            raise RuntimeError("Timed out waiting for parameter TOC.")\n');
fprintf(fid, '        time.sleep(0.05)\n');
fprintf(fid, '    for name, value in gains.items():\n');
fprintf(fid, '        group, param = name.split(".", 1)\n');
fprintf(fid, '        if group not in cf.param.toc.toc or param not in cf.param.toc.toc[group]:\n');
fprintf(fid, '            raise RuntimeError(f"{name} is not present in the Crazyflie parameter TOC.")\n');
fprintf(fid, '        cf.param.set_value(name, str(value))\n');
fprintf(fid, '        time.sleep(0.03)\n');
fprintf(fid, '        print(f"{name} = {cf.param.get_value(name)}")\n');
fclose(fid);

gainCommand = sprintf('"%s" "%s"', pythonExecutable, gainSetterScript);
[gainStatus, gainOutput] = system(gainCommand);
fprintf('%s', gainOutput);
delete(gainSetterScript);

if gainStatus ~= 0
    error(['Failed to apply the requested Mellinger gains. ', ...
           'The flight was not started.']);
end

command = sprintf('"%s" "%s"', pythonExecutable, pythonScript);
[status, commandOutput] = system(command);
fprintf('%s', commandOutput);

if status ~= 0
    error(['The Python flight script returned status %d. ', ...
           'The flight was aborted or did not complete normally.'], status);
end

%% Read the recorded response
if ~isfile('state_log.csv')
    error('state_log.csv was not created by the Python script.');
end

T = readtable('state_log.csv');
if isempty(T)
    error('state_log.csv contains no measured samples.');
end

t_meas = T.time;
x_meas = T.stateX;
y_meas = T.stateY;
z_meas = T.stateZ;
roll_meas = T.roll_rad;
pitch_meas = T.pitch_rad;
yaw_meas = T.yaw_rad;
u_meas = T.u;
v_meas = T.v;
w_meas = T.w;
p_meas = T.p_rad;
q_meas = T.q_rad;
r_meas = T.r_rad;

% Additional firmware-2026.04 logs
motor_m1req = T.motor_m1req;
motor_m2req = T.motor_m2req;
motor_m3req = T.motor_m3req;
motor_m4req = T.motor_m4req;

vx_global = T.stateEstimate_vx;
vy_global = T.stateEstimate_vy;
vz_global = T.stateEstimate_vz;

rateRoll_mrad_s  = T.stateEstimateZ_rateRoll_mrad_s;
ratePitch_mrad_s = T.stateEstimateZ_ratePitch_mrad_s;
rateYaw_mrad_s   = T.stateEstimateZ_rateYaw_mrad_s;

% Mellinger position-error integral states
ctrlMel_i_err_x = T.ctrlMel_i_err_x;
ctrlMel_i_err_y = T.ctrlMel_i_err_y;
ctrlMel_i_err_z = T.ctrlMel_i_err_z;

%% Compare the logged position with the reference at the actual log times
xref_meas = interp1(tsampled, xref, t_meas, 'linear', 'extrap');
yref_meas = interp1(tsampled, yref, t_meas, 'linear', 'extrap');
zref_meas = interp1(tsampled, zref, t_meas, 'linear', 'extrap');
yawref_meas_rad = deg2rad(interp1( ...
    tsampled, yawref_deg, t_meas, 'linear', 'extrap'));

% Stock Mellinger does not expose ctrlMel.pos_error_* as log variables.
% Compute the position error directly from reference minus measured position.
position_error = [xref_meas - x_meas, ...
                  yref_meas - y_meas, ...
                  zref_meas - z_meas];

fprintf('\nLogged samples: %d\n', height(T));
fprintf('Position RMSE [x y z] = [%.5f %.5f %.5f] m\n', ...
    sqrt(mean(position_error.^2, 1)));
fprintf('Position-error 2-norm over the recorded trajectory = %.5f\n', ...
    norm(position_error, 'fro'));

%% Plot the single-flight result
figure('Name','Mellinger reference flight using Loco positioning');

subplot(3,2,1)
plot(t_meas, x_meas, 'b', tsampled, xref, 'y--', 'LineWidth', 1.1)
xlabel('time [s]'); ylabel('x [m]');% legend('measured','reference');

subplot(3,2,3)
plot(t_meas, y_meas, 'b', tsampled, yref, 'y--', 'LineWidth', 1.1)
xlabel('time [s]'); ylabel('y [m]');% legend('measured','reference');

subplot(3,2,5)
plot(t_meas, z_meas, 'b', tsampled, zref, 'y--', 'LineWidth', 1.1)
xlabel('time [s]'); ylabel('z [m]');% legend('measured','reference');

subplot(3,2,2)
plot(t_meas, roll_meas, 'b', 'LineWidth', 1.1)
xlabel('time [s]'); ylabel('\phi [rad]');

subplot(3,2,4)
plot(t_meas, pitch_meas, 'b', 'LineWidth', 1.1)
xlabel('time [s]'); ylabel('\theta [rad]');

subplot(3,2,6)
plot(t_meas, yaw_meas, 'b', tsampled, deg2rad(yawref_deg), 'y--', 'LineWidth', 1.1)
xlabel('time [s]'); ylabel('\psi [rad]');% legend('measured','reference');