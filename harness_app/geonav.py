"""Geo-navigation state for the main app: GPS + compass + Kalman filter in one place.

- Listens to every GPS epoch (Gps.listeners, called from the GPS reader thread) and feeds the
  Kalman filter (kalman.GpsKalman).
- Reads the compass (compass.Compass, its own 15 Hz thread) and smooths the heading
  (kalman.HeadingFilter) every time status() is called.
- status() returns a flat dict for the CSV log and the web view (nav_* keys). It never raises:
  the navigation loop must not stop because of the GPS or the compass.

Later stages (waypoints, turn cues) read position/heading from here.
"""
import math
import threading
import time

import config as C
import kalman as KF

NAV_FIELDS = ["nav_lat", "nav_lon", "nav_sigma_m", "nav_speed_kmh", "nav_course", "nav_still",
              "nav_rejected", "heading_deg", "compass_b_ut"]


def make_compass():
    """Compass from config (defaults match the 2026-11-23 calibration); None if unavailable."""
    if not getattr(C, "COMPASS_ENABLED", True):
        return None
    from compass import Compass
    return Compass(getattr(C, "COMPASS_BUS", 7),
                   getattr(C, "COMPASS_CAL_FILE", C.BASE / "scripts" / "compass_cal.json"),
                   getattr(C, "COMPASS_FLIP", True), getattr(C, "COMPASS_MOUNT_DEG", 90.0),
                   getattr(C, "COMPASS_DECLINATION_DEG", 0.0))


class GeoNav:
    def __init__(self, gps=None, compass=None):
        self.gps, self.compass = gps, compass
        self.kf = KF.GpsKalman()
        self.hf = KF.HeadingFilter()
        self.lock = threading.Lock()
        self.state = None            # last Kalman state dict
        self.t_state = None
        self.heading = None
        self.b_ut = None
        self.errors = 0
        if gps is not None:
            gps.listeners.append(self._on_gps)

    # ---- GPS thread -------------------------------------------------------------
    def _on_gps(self, ev):
        lat = ev.get("lat")              # present only when the GGA has a fix
        try:
            with self.lock:
                st = self.kf.step(ev["t"], lat, ev.get("lon"), ev.get("hdop"),
                                  ev.get("cn0_top4"), ev.get("speed_kmh"), ev.get("course_deg"))
                if st is not None:
                    self.state, self.t_state = st, ev["t"]
        except Exception as e:          # never break the GPS reader
            self.errors += 1
            if self.errors <= 3:
                print(f"GeoNav filter error: {e}")

    # ---- any thread -------------------------------------------------------------
    def _update_heading(self):
        if self.compass is None:
            return
        try:
            c = self.compass.read()
        except Exception:
            return
        with self.lock:
            self.heading = self.hf.step(time.monotonic(), c.get("compass_deg"))
            self.b_ut = c.get("compass_b_ut")

    def position(self):
        """(lat, lon, sigma_m) of the filtered position, or None if there is no recent fix."""
        with self.lock:
            st, t = self.state, self.t_state
        if st is None or t is None or time.monotonic() - t > 3.0:
            return None
        return st["lat"], st["lon"], st["pos_sigma_m"]

    def heading_deg(self):
        self._update_heading()
        with self.lock:
            return self.heading

    def status(self):
        self._update_heading()
        with self.lock:
            st, t, h, b = self.state, self.t_state, self.heading, self.b_ut
        out = {k: None for k in NAV_FIELDS}
        out["heading_deg"] = None if h is None else round(h, 1)
        out["compass_b_ut"] = b
        if st is not None and t is not None and time.monotonic() - t <= 3.0:
            out.update({"nav_lat": round(st["lat"], 7), "nav_lon": round(st["lon"], 7),
                        "nav_sigma_m": round(st["pos_sigma_m"], 1),
                        "nav_speed_kmh": round(st["speed_kmh"], 2),
                        "nav_course": None if st["course_deg"] is None else round(st["course_deg"], 1),
                        "nav_still": int(bool(st["still"])), "nav_rejected": self.kf.n_rejected})
        return out

    def close(self):
        if self.gps is not None and self._on_gps in self.gps.listeners:
            self.gps.listeners.remove(self._on_gps)
        if self.compass is not None:
            self.compass.close()


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 (0 = north, 90 = east)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def distance_m(lat1, lon1, lat2, lon2):
    k = math.pi / 180.0 * KF.EARTH_R
    return math.hypot((lon2 - lon1) * k * math.cos(math.radians((lat1 + lat2) / 2)), (lat2 - lat1) * k)
