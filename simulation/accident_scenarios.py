"""
accident_scenarios.py
----------------------
IMPORTANT ACADEMIC NOTE:
This module is a RULE-BASED SIMULATOR. It is used ONLY to generate realistic
synthetic sensor sequences representing different driving events. It does NOT
perform any accident detection itself. Detection and severity classification
are performed later by the trained Deep Learning models
(see models/train_accident_model.py and models/train_severity_model.py).

Think of this file as a "virtual test track" that produces believable sensor
data for seven driving scenarios:
    1. NORMAL
    2. HARD_BRAKING
    3. POTHOLE
    4. SHARP_TURN
    5. MINOR_ACCIDENT
    6. MAJOR_ACCIDENT
    7. ROLLOVER
"""

import numpy as np

EVENT_CLASSES = [
    "NORMAL",
    "HARD_BRAKING",
    "POTHOLE",
    "SHARP_TURN",
    "MINOR_ACCIDENT",
    "MAJOR_ACCIDENT",
    "ROLLOVER",
]

# accident flag (0/1) per event type
ACCIDENT_FLAG = {
    "NORMAL": 0,
    "HARD_BRAKING": 0,
    "POTHOLE": 0,
    "SHARP_TURN": 0,
    "MINOR_ACCIDENT": 1,
    "MAJOR_ACCIDENT": 1,
    "ROLLOVER": 1,
}

# severity label per event type: 0=None,1=Low,2=Medium,3=High,4=Critical
SEVERITY_LABEL = {
    "NORMAL": 0,
    "HARD_BRAKING": 0,
    "POTHOLE": 0,
    "SHARP_TURN": 0,
    "MINOR_ACCIDENT": 1,
    "MAJOR_ACCIDENT": 3,
    "ROLLOVER": 4,
}


def _noise(n, scale):
    return np.random.normal(0, scale, n)


def generate_normal(n):
    """Normal driving: stable speed, low vibration, minimal gyro activity."""
    speed = np.clip(55 + np.cumsum(_noise(n, 0.4)), 45, 65)
    acc_x = _noise(n, 0.05)
    acc_y = _noise(n, 0.05)
    acc_z = 9.8 + _noise(n, 0.03)
    gyro_x = _noise(n, 0.5)
    gyro_y = _noise(n, 0.5)
    gyro_z = _noise(n, 0.5)
    vibration = np.abs(_noise(n, 0.2)) + 0.3
    temperature = 30 + _noise(n, 0.5)
    tilt = np.abs(_noise(n, 0.3))
    return speed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, vibration, temperature, tilt


def generate_hard_braking(n):
    """Sudden braking: speed drops sharply, large negative longitudinal accel."""
    speed = np.linspace(65, 20, n) + _noise(n, 1.0)
    speed = np.clip(speed, 15, 70)
    acc_x = np.concatenate([_noise(n // 3, 0.1), np.linspace(-0.2, -0.7, n - n // 3)])
    acc_y = _noise(n, 0.08)
    acc_z = 9.8 + _noise(n, 0.05)
    gyro_x = _noise(n, 1.0)
    gyro_y = _noise(n, 1.0)
    gyro_z = _noise(n, 1.5)
    vibration = np.abs(_noise(n, 0.6)) + 1.0
    temperature = 30 + _noise(n, 0.5)
    tilt = np.abs(_noise(n, 0.8)) + 0.5
    return speed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, vibration, temperature, tilt


def generate_pothole(n):
    """Pothole: short sharp vertical vibration spike, speed mostly unaffected."""
    speed = np.clip(50 + np.cumsum(_noise(n, 0.4)), 40, 60)
    acc_x = _noise(n, 0.08)
    acc_y = _noise(n, 0.08)
    spike_idx = n // 2
    acc_z = 9.8 + _noise(n, 0.05)
    acc_z[spike_idx:spike_idx + 3] += np.random.uniform(3, 5)
    gyro_x = _noise(n, 1.0)
    gyro_y = _noise(n, 1.0)
    gyro_z = _noise(n, 0.5)
    vibration = np.abs(_noise(n, 0.3)) + 0.5
    vibration[spike_idx:spike_idx + 3] += np.random.uniform(4, 6)
    temperature = 30 + _noise(n, 0.5)
    tilt = np.abs(_noise(n, 0.4))
    tilt[spike_idx:spike_idx + 3] += np.random.uniform(3, 6)
    return speed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, vibration, temperature, tilt


def generate_sharp_turn(n):
    """Sharp turn: large lateral acceleration and yaw (gyro_z) change."""
    speed = np.clip(45 - np.cumsum(np.abs(_noise(n, 0.2))), 25, 50)
    acc_x = _noise(n, 0.1)
    acc_y = np.linspace(0, np.random.uniform(2.5, 4.0), n) + _noise(n, 0.2)
    acc_z = 9.8 + _noise(n, 0.05)
    gyro_x = _noise(n, 1.0)
    gyro_y = _noise(n, 1.0)
    gyro_z = np.linspace(0, np.random.uniform(30, 60), n) + _noise(n, 2.0)
    vibration = np.abs(_noise(n, 0.4)) + 0.8
    temperature = 30 + _noise(n, 0.5)
    tilt = np.abs(np.linspace(0, np.random.uniform(8, 15), n)) + _noise(n, 1.0)
    return speed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, vibration, temperature, tilt


def _collision_profile(n, peak_g, peak_tilt, final_speed_ratio, peak_vibration):
    """Shared shape for accident-type events: normal -> spike -> settle/stop."""
    pre = int(n * 0.4)
    impact = int(n * 0.15)
    post = n - pre - impact

    # Speed: normal, then rapid drop at impact, then near-zero after
    speed_pre = np.clip(60 + np.cumsum(_noise(pre, 0.4)), 45, 75)
    start_speed = speed_pre[-1] if pre > 0 else 60
    speed_impact = np.linspace(start_speed, start_speed * final_speed_ratio, impact)
    speed_post = np.clip(speed_impact[-1] + _noise(post, 0.3), 0, None) * np.linspace(1, 0.05, post)
    speed = np.concatenate([speed_pre, speed_impact, speed_post])

    # Acceleration: spike at impact
    acc_x = np.concatenate([_noise(pre, 0.08),
                             np.linspace(0, -peak_g, impact) + _noise(impact, peak_g * 0.15),
                             _noise(post, 0.3)])
    acc_y = np.concatenate([_noise(pre, 0.08),
                             _noise(impact, peak_g * 0.4),
                             _noise(post, 0.2)])
    acc_z = np.concatenate([9.8 + _noise(pre, 0.05),
                             9.8 + np.linspace(0, peak_g * 0.6, impact) + _noise(impact, peak_g * 0.2),
                             9.8 + _noise(post, 0.3)])

    # Gyroscope: large spike at impact then settle
    gyro_x = np.concatenate([_noise(pre, 1.0), _noise(impact, peak_g * 20), _noise(post, 1.0)])
    gyro_y = np.concatenate([_noise(pre, 1.0), _noise(impact, peak_g * 20), _noise(post, 1.0)])
    gyro_z = np.concatenate([_noise(pre, 1.0), _noise(impact, peak_g * 25), _noise(post, 1.0)])

    # Vibration: spike then decay
    vib_pre = np.abs(_noise(pre, 0.3)) + 0.4
    vib_impact = np.linspace(0.5, peak_vibration, impact) + np.abs(_noise(impact, peak_vibration * 0.2))
    vib_post = np.clip(peak_vibration * np.exp(-np.linspace(0, 3, post)), 0.3, None) + np.abs(_noise(post, 0.3))
    vibration = np.concatenate([vib_pre, vib_impact, vib_post])

    # Temperature: mostly stable, tiny bump possible
    temperature = 30 + _noise(n, 0.5)

    # Tilt: rises during/after impact and stays elevated (esp. rollover)
    tilt_pre = np.abs(_noise(pre, 0.3))
    tilt_impact = np.linspace(0, peak_tilt, impact) + np.abs(_noise(impact, peak_tilt * 0.1))
    tilt_post = peak_tilt * (0.85 + 0.15 * np.sin(np.linspace(0, 2, post))) + np.abs(_noise(post, 1.0))
    tilt = np.concatenate([tilt_pre, tilt_impact, tilt_post])

    return speed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, vibration, temperature, tilt


def generate_minor_accident(n):
    return _collision_profile(n, peak_g=np.random.uniform(1.5, 2.5), peak_tilt=np.random.uniform(5, 12),
                               final_speed_ratio=0.4, peak_vibration=np.random.uniform(4, 6))


def generate_major_accident(n):
    return _collision_profile(n, peak_g=np.random.uniform(4.0, 6.5), peak_tilt=np.random.uniform(20, 40),
                               final_speed_ratio=0.05, peak_vibration=np.random.uniform(7, 9))


def generate_rollover(n):
    return _collision_profile(n, peak_g=np.random.uniform(5.5, 8.0), peak_tilt=np.random.uniform(55, 90),
                               final_speed_ratio=0.02, peak_vibration=np.random.uniform(8, 10))


GENERATORS = {
    "NORMAL": generate_normal,
    "HARD_BRAKING": generate_hard_braking,
    "POTHOLE": generate_pothole,
    "SHARP_TURN": generate_sharp_turn,
    "MINOR_ACCIDENT": generate_minor_accident,
    "MAJOR_ACCIDENT": generate_major_accident,
    "ROLLOVER": generate_rollover,
}


def generate_event_sequence(event_type, n_samples):
    """Return a dict of sensor arrays of length n_samples for the given event type."""
    if event_type not in GENERATORS:
        raise ValueError(f"Unknown event type: {event_type}")
    (speed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
     vibration, temperature, tilt) = GENERATORS[event_type](n_samples)
    return {
        "speed": speed,
        "acceleration_x": acc_x,
        "acceleration_y": acc_y,
        "acceleration_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "vibration": vibration,
        "temperature": temperature,
        "tilt_angle": tilt,
        "event_type": event_type,
        "accident": ACCIDENT_FLAG[event_type],
        "severity": SEVERITY_LABEL[event_type],
    }
