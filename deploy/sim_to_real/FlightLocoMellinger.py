"""Crazyflie 2.1 hardware flight: Loco deck, onboard EKF, onboard Mellinger.

Run from deploy/sim_to_real/. Typical path is FlyReferenceMellinger.m, which
writes trajectory.csv, applies Mellinger gains over the radio, then launches
this script. You can also run it directly after providing trajectory.csv.

Edit uri to match your radio. Firmware target: Crazyflie 2026.04.
Requires cflib. Input: trajectory.csv (timestamp [s], x/y/z [m], yaw [deg]).
Output: state_log.csv (reference interval only; landing is not logged).
"""

import csv
import math
import sched
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger


# USER SETTINGS

uri = 'radio://0/80/2M/E7E7E7E7E7'  # Replace with your Crazyflie radio URI.

# Loco mode: 0=auto, 1=TWR, 2=TDoA2, 3=TDoA3.
# Keep 0 unless you deliberately want to force the mode configured in anchors.
LOCO_MODE = 0

# Crazyflie firmware target: 2026.04
# Fixed reference/logging frequency [Hz].
# Change this single constant if the experiment frequency is changed.
fixedFrequency = 50
Ts = 1.0 / fixedFrequency
LOG_PERIOD_MS = int(round(1000.0 / fixedFrequency))

TAKEOFF_HEIGHT = 0.40   # metres above the measured launch position
TAKEOFF_TIME = 2.0      # seconds
LANDING_TIME = 2.0      # seconds

# Set True to apply the gain values below at the start of every flight.
# Set False to use the active firmware values. Changes made here are runtime
# changes only; this script does not store them persistently.
USE_CUSTOM_MELLINGER_GAINS = False

# Edit these values when tuning. The numbers shown are the current stock
# Mellinger values. With USE_CUSTOM_MELLINGER_GAINS=False, this dictionary is
# not written to the Crazyflie.
MELLINGER_GAINS = {
    # Horizontal position gains
    'ctrlMel.kp_xy': 0.4,
    'ctrlMel.kd_xy': 0.2,
    'ctrlMel.ki_xy': 0.05,
    'ctrlMel.i_range_xy': 2.0,

    # Vertical position gains
    'ctrlMel.kp_z': 1.25,
    'ctrlMel.kd_z': 0.4,
    'ctrlMel.ki_z': 0.05,
    'ctrlMel.i_range_z': 0.4,

    # Roll/pitch attitude gains
    'ctrlMel.kR_xy': 70000.0,
    'ctrlMel.kw_xy': 20000.0,
    'ctrlMel.ki_m_xy': 0.0,
    'ctrlMel.i_range_m_xy': 1.0,

    # Yaw attitude gains
    'ctrlMel.kR_z': 60000.0,
    'ctrlMel.kw_z': 12000.0,
    'ctrlMel.ki_m_z': 500.0,
    'ctrlMel.i_range_m_z': 1500.0,

    # Additional roll/pitch angular-rate derivative gain
    'ctrlMel.kd_omega_rp': 200.0,
}

# Mellinger also uses the complete flight-ready mass. Leave None to retain the firmware value
MELLINGER_MASS_KG = None


# READ THE MATLAB REFERENCE

trajectory = []
with open('trajectory.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        trajectory.append({
            'timestamp': float(row['timestamp']),
            'x': float(row['x']),
            'y': float(row['y']),
            'z': float(row['z']),
            'yaw': float(row['yaw']),
        })

if not trajectory:
    raise RuntimeError('trajectory.csv contains no samples.')

if abs(trajectory[0]['timestamp']) > 1e-9:
    raise RuntimeError('The first trajectory timestamp must be 0.0 s.')

for i in range(1, len(trajectory)):
    if trajectory[i]['timestamp'] <= trajectory[i - 1]['timestamp']:
        raise RuntimeError('Trajectory timestamps must be strictly increasing.')


# STATE AND LOG STORAGE

log_data = []
start_x = start_y = start_z = start_yaw = 0.0
received = {'log1': False, 'log2': False, 'log3': False, 'log4': False}

latest_data = {
    'stateX': 0.0, 'stateY': 0.0, 'stateZ': 0.0,
    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    'vX': 0.0, 'vY': 0.0, 'vZ': 0.0,
    'p': 0.0, 'q': 0.0, 'r': 0.0,

    # Additional requested logs available in firmware 2026.04
    'm1req': 0, 'm2req': 0, 'm3req': 0, 'm4req': 0,
    'vx_world': 0.0, 'vy_world': 0.0, 'vz_world': 0.0,
    'rateRoll_mrad_s': 0, 'ratePitch_mrad_s': 0, 'rateYaw_mrad_s': 0,

    # Mellinger position-error integral states exposed by stock firmware.
    'i_err_x': 0.0, 'i_err_y': 0.0, 'i_err_z': 0.0,
}


def wait_for_position_estimator(scf, timeout_s=30.0):
    """Reset/wait logic used by cflib for a converged Kalman position."""
    print('Waiting for the Kalman position estimate to converge...')

    log_config = LogConfig(name='KalmanVariance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_x_history = [1000.0] * 10
    var_y_history = [1000.0] * 10
    var_z_history = [1000.0] * 10
    threshold = 0.001
    start_time = time.monotonic()

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            dx = max(var_x_history) - min(var_x_history)
            dy = max(var_y_history) - min(var_y_history)
            dz = max(var_z_history) - min(var_z_history)

            if dx < threshold and dy < threshold and dz < threshold:
                print(
                    f'Kalman estimate converged: '
                    f'dVar=[{dx:.3e}, {dy:.3e}, {dz:.3e}]'
                )
                return

            if time.monotonic() - start_time > timeout_s:
                raise RuntimeError(
                    'Kalman estimate did not converge. Check the Loco anchors, '
                    'anchor geometry/mode, and deck reception before flying.'
                )


def request_arming(cf, arm):
    """Use the current supervisor API, with an older-cflib fallback."""
    if hasattr(cf, 'supervisor'):
        cf.supervisor.send_arming_request(arm)
    else:
        # Compatibility only: the platform arming API is deprecated in
        # current cflib, but it exists in older installations.
        cf.platform.send_arming_request(arm)


def make_log_event(t0):
    """Snapshot the latest Crazyflie log values into the output CSV."""
    def _log_event():
        elapsed = time.monotonic() - t0

        roll = math.radians(latest_data['roll'])
        pitch = math.radians(latest_data['pitch'])
        yaw_rel = math.radians(latest_data['yaw'] - start_yaw)

        # kalman.statePX/PY/PZ are already body-frame velocities in m/s,
        # so they are logged directly as u, v, and w. Do not rotate them again.
        u = latest_data['vX']
        v = latest_data['vY']
        w = latest_data['vZ']

        p = math.radians(latest_data['p'])
        q = math.radians(latest_data['q'])
        r = math.radians(latest_data['r'])

        log_data.append([
            elapsed,
            latest_data['stateX'] - start_x,
            latest_data['stateY'] - start_y,
            latest_data['stateZ'] - start_z,
            roll, pitch, yaw_rel,
            u, v, w,
            p, q, r,

            # Raw stabilizer attitude values [deg]
            latest_data['roll'],
            latest_data['pitch'],
            latest_data['yaw'],

            # Global-frame estimator velocity [m/s]
            latest_data['vx_world'],
            latest_data['vy_world'],
            latest_data['vz_world'],

            # Compressed estimator angular rates [mrad/s]
            latest_data['rateRoll_mrad_s'],
            latest_data['ratePitch_mrad_s'],
            latest_data['rateYaw_mrad_s'],

            # Requested motor power, including battery compensation
            latest_data['m1req'],
            latest_data['m2req'],
            latest_data['m3req'],
            latest_data['m4req'],

            # Stock Mellinger position-error integral states
            latest_data['i_err_x'],
            latest_data['i_err_y'],
            latest_data['i_err_z'],

            # Experiment configuration constant
            fixedFrequency,
        ])

    return _log_event


# CONNECT, CONFIGURE, FLY, AND LOG

cflib.crtp.init_drivers(enable_debug_driver=False)
exit_code = 0

try:
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        log1 = None
        log2 = None
        log3 = None
        log4 = None
        logs_started = False
        armed = False

        try:
            # Wait until all parameter values are available.
            parameter_wait_start = time.monotonic()
            while not cf.param.is_updated:
                if time.monotonic() - parameter_wait_start > 10.0:
                    raise RuntimeError('Timed out waiting for parameters.')
                time.sleep(0.1)

            # Firmware 2026.04 current Loco deck parameter.
            if int(float(cf.param.get_value('deck.bcLoco'))) == 0:
                raise RuntimeError('The Loco Positioning Deck was not detected.')
            print('Loco deck detected (deck.bcLoco=1).')

            # Firmware 2026.04 exposes both of these Loco parameters.
            cf.param.set_value('loco.mode', str(LOCO_MODE))
            cf.param.set_value('loco.fwdToEstimator', '1')

            # Select the extended Kalman estimator explicitly.
            cf.param.set_value('stabilizer.estimator', '2')
            time.sleep(0.2)

            # Reset the EKF and block until its position variance settles.
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            wait_for_position_estimator(scf)

            # Select the onboard Mellinger controller.
            cf.param.set_value('stabilizer.controller', '2')
            time.sleep(0.2)

            # Optional runtime Mellinger gain overrides.
            if USE_CUSTOM_MELLINGER_GAINS:
                print('Applying custom Mellinger gains:')
                for name, value in MELLINGER_GAINS.items():
                    cf.param.set_value(name, str(value))
                    time.sleep(0.03)
                    print(f'  {name} = {value}')
            else:
                print('Using active firmware Mellinger gains (no override).')

            # Optional runtime mass correction for the complete flying vehicle.
            if MELLINGER_MASS_KG is not None:
                if MELLINGER_MASS_KG <= 0.0:
                    raise RuntimeError('MELLINGER_MASS_KG must be positive.')
                cf.param.set_value('ctrlMel.mass', str(MELLINGER_MASS_KG))
                time.sleep(0.1)
                print(f'Applied ctrlMel.mass = {MELLINGER_MASS_KG} kg.')

            controller_id = int(float(
                cf.param.get_value('stabilizer.controller')
            ))
            if controller_id != 2:
                raise RuntimeError(
                    f'Mellinger selection failed: controller={controller_id}'
                )
            print('Controller selected: Mellinger (controller=2).')

            # Position/orientation log block: six floats = 24 bytes.
            log1 = LogConfig(name='States1', period_in_ms=LOG_PERIOD_MS)
            log1.add_variable('kalman.stateX', 'float')
            log1.add_variable('kalman.stateY', 'float')
            log1.add_variable('kalman.stateZ', 'float')
            log1.add_variable('stabilizer.roll', 'float')
            log1.add_variable('stabilizer.pitch', 'float')
            log1.add_variable('stabilizer.yaw', 'float')

            # Existing body-velocity/gyro block: six floats = 24 bytes.
            log2 = LogConfig(name='States2', period_in_ms=LOG_PERIOD_MS)
            log2.add_variable('kalman.statePX', 'float')
            log2.add_variable('kalman.statePY', 'float')
            log2.add_variable('kalman.statePZ', 'float')
            log2.add_variable('gyro.x', 'float')
            log2.add_variable('gyro.y', 'float')
            log2.add_variable('gyro.z', 'float')

            # Firmware 2026.04 motor/rate block:
            # 4 x LOG_INT32 motor requests + 3 x LOG_INT16 rates = 22 bytes.
            log3 = LogConfig(name='MotorsRates', period_in_ms=LOG_PERIOD_MS)
            log3.add_variable('motor.m1req')
            log3.add_variable('motor.m2req')
            log3.add_variable('motor.m3req')
            log3.add_variable('motor.m4req')
            log3.add_variable('stateEstimateZ.rateRoll')
            log3.add_variable('stateEstimateZ.ratePitch')
            log3.add_variable('stateEstimateZ.rateYaw')

            # Six floats = 24 bytes:
            # global-frame velocity + stock Mellinger position-integral states.
            log4 = LogConfig(name='VelMelI', period_in_ms=LOG_PERIOD_MS)
            log4.add_variable('stateEstimate.vx', 'float')
            log4.add_variable('stateEstimate.vy', 'float')
            log4.add_variable('stateEstimate.vz', 'float')
            log4.add_variable('ctrlMel.i_err_x', 'float')
            log4.add_variable('ctrlMel.i_err_y', 'float')
            log4.add_variable('ctrlMel.i_err_z', 'float')

            def cb1(timestamp, data, logconf):
                latest_data['stateX'] = data['kalman.stateX']
                latest_data['stateY'] = data['kalman.stateY']
                latest_data['stateZ'] = data['kalman.stateZ']
                latest_data['roll'] = data['stabilizer.roll']
                latest_data['pitch'] = data['stabilizer.pitch']
                latest_data['yaw'] = data['stabilizer.yaw']
                received['log1'] = True

            def cb2(timestamp, data, logconf):
                latest_data['vX'] = data['kalman.statePX']
                latest_data['vY'] = data['kalman.statePY']
                latest_data['vZ'] = data['kalman.statePZ']
                latest_data['p'] = data['gyro.x']
                latest_data['q'] = data['gyro.y']
                latest_data['r'] = data['gyro.z']
                received['log2'] = True

            def cb3(timestamp, data, logconf):
                latest_data['m1req'] = data['motor.m1req']
                latest_data['m2req'] = data['motor.m2req']
                latest_data['m3req'] = data['motor.m3req']
                latest_data['m4req'] = data['motor.m4req']
                latest_data['rateRoll_mrad_s'] = data['stateEstimateZ.rateRoll']
                latest_data['ratePitch_mrad_s'] = data['stateEstimateZ.ratePitch']
                latest_data['rateYaw_mrad_s'] = data['stateEstimateZ.rateYaw']
                received['log3'] = True

            def cb4(timestamp, data, logconf):
                latest_data['vx_world'] = data['stateEstimate.vx']
                latest_data['vy_world'] = data['stateEstimate.vy']
                latest_data['vz_world'] = data['stateEstimate.vz']
                latest_data['i_err_x'] = data['ctrlMel.i_err_x']
                latest_data['i_err_y'] = data['ctrlMel.i_err_y']
                latest_data['i_err_z'] = data['ctrlMel.i_err_z']
                received['log4'] = True

            cf.log.add_config(log1)
            cf.log.add_config(log2)
            cf.log.add_config(log3)
            cf.log.add_config(log4)

            log1.data_received_cb.add_callback(cb1)
            log2.data_received_cb.add_callback(cb2)
            log3.data_received_cb.add_callback(cb3)
            log4.data_received_cb.add_callback(cb4)

            log1.start()
            log2.start()
            log3.start()
            log4.start()

            logs_started = True
            time.sleep(0.5)

            if not all(received[name] for name in ('log1', 'log2', 'log3', 'log4')):
                raise RuntimeError('Required state/control logs were not received.')

            # Measured launch position in the Loco coordinate frame.
            ground_x = latest_data['stateX']
            ground_y = latest_data['stateY']
            ground_z = latest_data['stateZ']
            ground_yaw = latest_data['yaw']

            # Brushed CF2.1 normally auto-arms, but it is better to use request_arming() with the
            # current supervisor API and an older-cflib fallback.
            request_arming(cf, True)
            armed = True
            time.sleep(1.0)

            # Smooth takeoff to TAKEOFF_HEIGHT above the measured launch height.
            takeoff_steps = max(1, round(TAKEOFF_TIME / Ts))
            for i in range(takeoff_steps + 1):
                alpha = i / takeoff_steps
                cf.commander.send_position_setpoint(
                    ground_x,
                    ground_y,
                    ground_z + alpha * TAKEOFF_HEIGHT,
                    ground_yaw,
                )
                time.sleep(Ts)
            time.sleep(0.5)

            # Preserve the original interface: MATLAB's trajectory is relative to the measured post-takeoff hover point
            start_x = latest_data['stateX']
            start_y = latest_data['stateY']
            start_z = latest_data['stateZ']
            start_yaw = latest_data['yaw']

            # Schedule one output snapshot and one position setpoint at every timestamp supplied by MATLAB
            t0 = time.monotonic()
            scheduler = sched.scheduler(
                timefunc=time.monotonic,
                delayfunc=time.sleep,
            )
            log_event = make_log_event(t0)

            for point in trajectory:
                abs_t = t0 + point['timestamp']

                # Lower priority number executes first: sample y(t), then apply reference position associated with that sample instant
                scheduler.enterabs(
                    abs_t,
                    priority=1,
                    action=log_event,
                )
                scheduler.enterabs(
                    abs_t,
                    priority=2,
                    action=cf.commander.send_position_setpoint,
                    argument=(
                        start_x + point['x'],
                        start_y + point['y'],
                        start_z + point['z'],
                        start_yaw + point['yaw'],
                    ),
                )

            print(f'Flying the reference and logging at {fixedFrequency} Hz...')
            scheduler.run()

            # The output CSV contains the reference interval only, not landing.
            log1.stop()
            log2.stop()
            log3.stop()
            log4.stop()
            logs_started = False

            # Smoothly return to the measured launch position and height.
            land_x0 = latest_data['stateX']
            land_y0 = latest_data['stateY']
            land_z0 = latest_data['stateZ']
            land_yaw0 = latest_data['yaw']
            landing_steps = max(1, round(LANDING_TIME / Ts))

            for i in range(landing_steps + 1):
                alpha = i / landing_steps
                cf.commander.send_position_setpoint(
                    land_x0 + alpha * (ground_x - land_x0),
                    land_y0 + alpha * (ground_y - land_y0),
                    land_z0 + alpha * (ground_z - land_z0),
                    land_yaw0 + alpha * (ground_yaw - land_yaw0),
                )
                time.sleep(Ts)

            cf.commander.send_stop_setpoint()
            time.sleep(0.1)
            request_arming(cf, False)
            armed = False
            time.sleep(0.2)

        finally:
            # Ensure a failure cannot leave the setpoint stream active.
            if logs_started:
                try:
                    for logconf in (log1, log2, log3, log4):
                        if logconf is not None:
                            logconf.stop()
                except Exception:
                    pass
            try:
                cf.commander.send_stop_setpoint()
                time.sleep(0.1)
            except Exception:
                pass
            if armed:
                try:
                    request_arming(cf, False)
                    time.sleep(0.2)
                except Exception:
                    pass

except Exception as error:
    print(f'Flight aborted: {error}', file=sys.stderr)
    exit_code = 1


# SAVE THE REFERENCE-INTERVAL LOG

with open('state_log.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'time', 'stateX', 'stateY', 'stateZ',
        'roll_rad', 'pitch_rad', 'yaw_rad',
        'u', 'v', 'w', 'p_rad', 'q_rad', 'r_rad',

        'stabilizer_roll_deg', 'stabilizer_pitch_deg', 'stabilizer_yaw_deg',
        'stateEstimate_vx', 'stateEstimate_vy', 'stateEstimate_vz',
        'stateEstimateZ_rateRoll_mrad_s',
        'stateEstimateZ_ratePitch_mrad_s',
        'stateEstimateZ_rateYaw_mrad_s',
        'motor_m1req', 'motor_m2req', 'motor_m3req', 'motor_m4req',
        'ctrlMel_i_err_x', 'ctrlMel_i_err_y', 'ctrlMel_i_err_z',
        'fixedFrequency',
    ])
    writer.writerows(log_data)

print(f'Saved {len(log_data)} samples to state_log.csv.')
sys.exit(exit_code)
