"""
distance_calculator.py
------------------------
Implements the Haversine formula to compute great-circle distance (in km)
between two latitude/longitude points. Used to find the nearest emergency
and service facilities to a simulated accident location, without requiring
any real mapping API.
"""

import math


def haversine_distance(lat1, lon1, lat2, lon2):
    """Return the distance in kilometers between two GPS points."""
    R = 6371.0  # Earth's radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


if __name__ == "__main__":
    # City Care Hospital vs a sample accident point
    d = haversine_distance(28.6139, 77.2090, 28.6250, 77.2150)
    print(f"Distance: {d:.2f} km")
