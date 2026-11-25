"""CSV logger: one row per processed frame."""
import csv
import glob
import time
from datetime import datetime
from pathlib import Path

import config as C

FIELDS = ["t_s", "fps", "lat_ms", "seg_ms", "det_ms", "plan_ms", "command", "reason", "error",
          "sidewalk_frac", "n_boxes", "n_obst", "angle_deg", "path_len", "obst_dist", "v_ina", "i_ina", "p_ina", "v_bat_adc",
          "v_jet", "pico_mode", "tj_c", "cam_fps", "exp_ms", "iso", "rec", "web_clients",
          "gps_fix", "gps_sats", "gps_sats_view", "gps_cn0_top4", "gps_hdop", "gps_lat", "gps_lon", "gps_speed_kmh", "gps_course",
          "nav_lat", "nav_lon", "nav_sigma_m", "nav_speed_kmh", "nav_course", "nav_still", "nav_rejected",
          "heading_deg", "compass_b_ut", "route", "wp_i", "wp_type", "wp_dist_m", "wp_rel_deg", "cue", "chest"]


def _tj_path():
    """Find the SoC thermal zone (tj-thermal on Orin)."""
    for zone in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
        try:
            if Path(zone, "type").read_text().strip().lower().startswith("tj"):
                return Path(zone, "temp")
        except OSError:
            continue
    return None


class CsvLogger:
    def __init__(self):
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = C.LOG_DIR / f"nav_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._w.writeheader()
        self._t0 = time.monotonic()
        self._t_flush = self._t0
        self._tj_path = _tj_path()
        self._tj = None
        self._t_tj = 0.0

    def temperature(self):
        now = time.monotonic()
        if self._tj_path and now - self._t_tj > 1.0:
            try:
                self._tj = int(self._tj_path.read_text()) / 1000.0
            except (OSError, ValueError):
                self._tj = None
            self._t_tj = now
        return self._tj

    def write(self, row):
        row = dict(row)
        row["t_s"] = round(time.monotonic() - self._t0, 3)
        row["tj_c"] = self.temperature()
        self._w.writerow({k: row.get(k) for k in FIELDS})
        if time.monotonic() - self._t_flush > 1.0:
            self._f.flush()
            self._t_flush = time.monotonic()

    def close(self):
        self._f.close()
