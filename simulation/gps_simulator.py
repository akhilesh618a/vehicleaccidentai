"""
gps_simulator.py
-----------------
Simulates GPS latitude/longitude for a vehicle driving around a base location
(default: New Delhi area). No real GPS hardware is used - this is purely for
academic simulation of vehicle location tracking.
"""

import numpy as np

# Base location: New Delhi (used only as a realistic demo center point)
BASE_LAT = 28.6139
BASE_LON = 77.2090


class GPSSimulator:
    def __init__(self, base_lat=BASE_LAT, base_lon=BASE_LON, seed=None):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.lat = base_lat + np.random.uniform(-0.01, 0.01)
        self.lon = base_lon + np.random.uniform(-0.01, 0.01)
        self._frozen = False
        if seed is not None:
            np.random.seed(seed)

    def step(self, moving=True):
        """Advance the simulated vehicle position by a small random amount."""
        if self._frozen:
            return self.lat, self.lon
        if moving:
            self.lat += np.random.uniform(-0.0006, 0.0006)
            self.lon += np.random.uniform(-0.0006, 0.0006)
        return self.lat, self.lon

    def freeze(self):
        """Freeze location - called the moment an accident is detected,
        so the frozen coordinates become the official accident location."""
        self._frozen = True
        return self.lat, self.lon

    def reset(self):
        self.lat = self.base_lat + np.random.uniform(-0.01, 0.01)
        self.lon = self.base_lon + np.random.uniform(-0.01, 0.01)
        self._frozen = False


def generate_track(n_points, base_lat=BASE_LAT, base_lon=BASE_LON):
    """Generate an array of (lat, lon) points for dataset generation purposes."""
    gps = GPSSimulator(base_lat, base_lon)
    lats, lons = [], []
    for _ in range(n_points):
        lat, lon = gps.step()
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)
