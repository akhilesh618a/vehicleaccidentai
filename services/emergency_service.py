"""
emergency_service.py
----------------------
Given an accident location, finds the nearest facility of each type from the
simulated service_centers.json database, and builds a simulated emergency
alert payload. No real emergency calls or SMS/GSM services are used -
everything here is a SOFTWARE SIMULATION for academic demonstration.
"""

import os
import json
from datetime import datetime

from services.distance_calculator import haversine_distance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_CENTERS_PATH = os.path.join(BASE_DIR, "service_centers.json")


def load_service_centers(path=SERVICE_CENTERS_PATH):
    with open(path, "r") as f:
        return json.load(f)


def find_nearest_by_type(lat, lon, facility_type, centers=None):
    """Return the nearest facility of a given type (e.g. 'Hospital') and its distance in km."""
    if centers is None:
        centers = load_service_centers()

    candidates = [c for c in centers if c["type"] == facility_type]
    if not candidates:
        return None

    best, best_dist = None, float("inf")
    for c in candidates:
        d = haversine_distance(lat, lon, c["latitude"], c["longitude"])
        if d < best_dist:
            best, best_dist = c, d

    result = dict(best)
    result["distance_km"] = round(best_dist, 2)
    return result


def find_all_nearest(lat, lon, centers=None):
    """Return the nearest facility of every category, keyed by facility type."""
    if centers is None:
        centers = load_service_centers()

    types = sorted(set(c["type"] for c in centers))
    return {t: find_nearest_by_type(lat, lon, t, centers) for t in types}


def build_emergency_alert(vehicle_id, accident_prob, severity_label, severity_confidence,
                           speed_before_impact, peak_acceleration_g, tilt_angle,
                           lat, lon, vehicle_status="STOPPED"):
    """Build a structured, simulated emergency alert payload (not a real dispatch)."""
    nearest = find_all_nearest(lat, lon)
    alert = {
        "alert_type": "ACCIDENT ALERT (SIMULATED)",
        "vehicle_id": vehicle_id,
        "timestamp": datetime.now().strftime("%I:%M:%S %p"),
        "accident_detected": True,
        "accident_confidence_pct": round(accident_prob * 100, 1),
        "severity": severity_label,
        "severity_confidence_pct": round(severity_confidence * 100, 1),
        "speed_before_impact_kmph": round(speed_before_impact, 1),
        "peak_acceleration_g": round(peak_acceleration_g, 2),
        "tilt_deg": round(tilt_angle, 1),
        "gps": {"latitude": round(lat, 6), "longitude": round(lon, 6)},
        "vehicle_status": vehicle_status,
        "nearest_facilities": nearest,
        "notified": [
            "Emergency Contact",
            "Nearest Hospital",
            "Nearest Police Station",
            "Vehicle Service Center",
        ],
        "disclaimer": "This is a simulated notification for academic demonstration only. "
                       "No real emergency services were contacted.",
    }
    return alert


if __name__ == "__main__":
    alert = build_emergency_alert(
        vehicle_id="VH-001", accident_prob=0.968, severity_label="HIGH",
        severity_confidence=0.953, speed_before_impact=78, peak_acceleration_g=6.4,
        tilt_angle=43, lat=28.6139, lon=77.2090,
    )
    print(json.dumps(alert, indent=2))
