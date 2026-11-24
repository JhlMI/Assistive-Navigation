"""GPS (u-blox M8N on the Jetson UART, pins 8/10 -> /dev/ttyTHS1) in a background thread.

- At start: finds the baud rate and configures the module for walking (UBX): GGA + RMC every fix,
  GSV (satellites in view and their signal) once per second, pedestrian dynamic model, 5 Hz, 115200 baud. Harmless if it is already configured.
- Keeps the latest fix (lat, lon, satellites, HDOP, speed, course) for the main loop and the web.
- Field test: start_test() writes every fix to logs/gps/gpstest_<time>.csv; mark() tags the next
  fix with a label (corner A, corner B...), so the stops can be analysed later.
- track(): the test points in meters (east, north) from the first point, for the live map.

Wiring (M8N 4-pin connector): red 5 V -> pin 2, black GND -> pin 6,
green RX -> pin 8 (Jetson TX), yellow TX -> pin 10 (Jetson RX).
"""
import csv
import math
import struct
import threading
import time
from datetime import datetime

import serial

import config as C

BAUDS = (115200, 9600, 38400, 57600)
EARTH_R = 6371000.0


# ---- UBX configuration --------------------------------------------------------
def _ubx(cls, mid, payload=b""):
    body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
    a = b = 0
    for x in body:
        a = (a + x) & 0xFF
        b = (b + a) & 0xFF
    return b"\xb5\x62" + body + bytes([a, b])


def _find_baud(port):
    for b in BAUDS:
        with serial.Serial(port, b, timeout=0.5) as s:
            t0 = time.time()
            while time.time() - t0 < 2.0:
                if s.readline(256).startswith(b"$G"):     # readline(256): never hangs on a wrong baud
                    return b
    return None


def configure(port):
    """Applies the walking setup. Returns the baud rate in use afterwards (None if no GPS)."""
    baud = _find_baud(port)
    if baud is None:
        return None
    with serial.Serial(port, baud, timeout=0.2) as s:
        gsv_rate = max(1, 1000 // C.GPS_RATE_MS)           # GSV (satellites in view) once per second
        for msg_id, rate in ((0x01, 0), (0x02, 0), (0x03, gsv_rate), (0x05, 0)):   # GLL, GSA, GSV, VTG
            s.write(_ubx(0x06, 0x01, bytes([0xF0, msg_id, rate])))
            time.sleep(0.05)
        s.write(_ubx(0x06, 0x24, struct.pack("<HB", 0x0001, C.GPS_DYN_MODEL) + bytes(33)))   # CFG-NAV5
        time.sleep(0.05)
        s.write(_ubx(0x06, 0x08, struct.pack("<HHH", C.GPS_RATE_MS, 1, 1)))                  # CFG-RATE
        time.sleep(0.05)
        if baud != C.GPS_BAUD:                                                               # CFG-PRT
            s.write(_ubx(0x06, 0x00, struct.pack("<BBHIIHHHH", 1, 0, 0, 0x08D0, C.GPS_BAUD, 3, 3, 0, 0)))
            s.flush()
            time.sleep(0.3)
    if baud != C.GPS_BAUD:
        with serial.Serial(port, C.GPS_BAUD, timeout=0.2) as s:
            time.sleep(0.3)
            s.write(_ubx(0x06, 0x09, struct.pack("<IIIB", 0, 0xFFFF, 0, 0x17)))               # CFG-CFG save
            time.sleep(0.3)
    return C.GPS_BAUD


# ---- NMEA parsing -------------------------------------------------------------
def _checksum_ok(line):
    try:
        body, cs = line[1:].split("*")
        x = 0
        for ch in body:
            x ^= ord(ch)
        return int(cs[:2], 16) == x
    except ValueError:
        return False


def _to_deg(value, hemi):
    if not value:
        return None
    dot = value.index(".")
    deg = float(value[:dot - 2]) + float(value[dot - 2:]) / 60.0
    return -deg if hemi in ("S", "W") else deg


def to_local(lat, lon, lat0, lon0):
    """Equirectangular projection: meters east/north of (lat0, lon0). Error < 1 cm over a few km."""
    k = math.pi / 180.0 * EARTH_R
    return (lon - lon0) * k * math.cos(math.radians(lat0)), (lat - lat0) * k


class Gps:
    FIELDS = ["t_s", "utc", "lat", "lon", "fix", "sats", "hdop", "alt_m", "speed_kmh", "course_deg",
              "east_m", "north_m", "sats_view", "cn0_top4", "mark"]
    CONSTELLATION = {"GP": "GPS", "GL": "GLONASS", "GA": "Galileo", "GB": "BeiDou", "BD": "BeiDou"}

    def __init__(self, port=None, configure_module=True):
        self.port = port or C.GPS_PORT
        self.baud = configure(self.port) if configure_module else _find_baud(self.port)
        if self.baud is None:
            raise RuntimeError(f"no NMEA on {self.port}")
        self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
        self.lock = threading.Lock()
        self.listeners = []                 # callables f(fix_dict) called on every GGA (fix or not)
        self.fix = {}                       # latest GGA/RMC data
        self.t_fix = None                   # monotonic time of the latest valid fix
        self.n_sentences = 0
        self._speed = self._course = None
        self.sky = {}                       # talker -> [(prn, elevation, snr)], from GSV, once per second
        self._sky_tmp = {}
        self.t_sky = None
        # field test
        self.test_path = None
        self._test_f = self._test_w = None
        self._t0 = None
        self._origin = None
        self._pending_mark = None
        self.marks = []                     # (label, east, north)
        self.points = []                    # (east, north)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---- reading ----
    def _loop(self):
        while self._running:
            try:
                line = self._ser.readline(256).decode("ascii", "ignore").strip()
            except serial.SerialException:
                time.sleep(0.5)
                continue
            if not line.startswith("$") or not _checksum_ok(line):
                continue
            self.n_sentences += 1
            p = line.split("*")[0].split(",")
            kind = p[0][3:]
            if kind == "GSV" and len(p) > 3:
                self._on_gsv(p)
            elif kind == "RMC" and len(p) > 8:
                self._speed = float(p[7]) * 1.852 if p[7] else None
                self._course = float(p[8]) if p[8] else None
            elif kind == "GGA" and len(p) > 9:
                self._on_gga(p)

    def _on_gsv(self, p):
        """GSV comes in parts (up to 4 satellites each) per constellation; commit when the last part arrives."""
        talker = p[0][1:3]
        try:
            n_msg, i_msg = int(p[1]), int(p[2])
        except ValueError:
            return
        if i_msg == 1:
            self._sky_tmp[talker] = []
        sats = self._sky_tmp.setdefault(talker, [])
        for k in range(4, len(p) - 3, 4):
            prn, elev, snr = p[k], p[k + 1], p[k + 3]
            if prn:
                sats.append((prn, int(elev) if elev else None, int(snr) if snr else None))
        if i_msg == n_msg:
            with self.lock:
                self.sky[talker] = self._sky_tmp.pop(talker)
                self.t_sky = time.monotonic()

    def sky_quality(self):
        """Satellites in view, how many have signal, and the mean C/N0 of the 4 strongest (dB-Hz).
        u-blox rule of thumb: top-4 C/N0 >= 40 is a clear sky; < 35 means blocked sky or interference."""
        with self.lock:
            fresh = self.t_sky is not None and time.monotonic() - self.t_sky < 5.0
            sats = [s for lst in self.sky.values() for s in lst] if fresh else []
        snr = sorted((s[2] for s in sats if s[2]), reverse=True)
        return {"sats_view": len(sats) if fresh else None, "sats_signal": len(snr) if fresh else None,
                "cn0_top4": round(sum(snr[:4]) / len(snr[:4]), 1) if snr else None}

    def sky_list(self):
        with self.lock:
            return [{"sys": self.CONSTELLATION.get(t, t), "prn": prn, "elev": el, "snr": snr}
                    for t, lst in sorted(self.sky.items()) for prn, el, snr in lst]

    def _on_gga(self, p):
        fix_q, sats = int(p[6] or 0), int(p[7] or 0)
        d = {"utc": p[1][:6], "fix": fix_q, "sats": sats, "hdop": float(p[8]) if p[8] else None}
        if fix_q > 0:
            d.update(lat=_to_deg(p[2], p[3]), lon=_to_deg(p[4], p[5]),
                     alt_m=float(p[9]) if p[9] else None,
                     speed_kmh=None if self._speed is None else round(self._speed, 2),
                     course_deg=self._course)
        with self.lock:
            self.fix = d
            if fix_q > 0:
                self.t_fix = time.monotonic()
                if self._test_w is not None:
                    self._write_test(d)
        if self.listeners:
            q = self.sky_quality()
            ev = {**d, "t": time.monotonic(), "cn0_top4": q["cn0_top4"], "sats_view": q["sats_view"]}
            for f in self.listeners:
                try:
                    f(ev)
                except Exception as e:          # a listener must never stop the reader
                    print(f"GPS listener error: {e}")

    def _write_test(self, d):
        if self._origin is None:
            self._origin = (d["lat"], d["lon"])
        e, n = to_local(d["lat"], d["lon"], *self._origin)
        mark, self._pending_mark = self._pending_mark, None
        if mark:
            self.marks.append((mark, e, n))
        self.points.append((e, n))
        q = self._sky_unlocked()
        self._test_w.writerow({**{k: d.get(k) for k in self.FIELDS}, **q, "t_s": round(time.monotonic() - self._t0, 2),
                               "lat": round(d["lat"], 7), "lon": round(d["lon"], 7),
                               "east_m": round(e, 2), "north_m": round(n, 2), "mark": mark})
        self._test_f.flush()

    def _sky_unlocked(self):
        """sky_quality() for use while self.lock is already held."""
        sats = [s for lst in self.sky.values() for s in lst]
        snr = sorted((s[2] for s in sats if s[2]), reverse=True)
        return {"sats_view": len(sats), "cn0_top4": round(sum(snr[:4]) / len(snr[:4]), 1) if snr else None}

    # ---- state for the main loop / web ----
    def status(self):
        with self.lock:
            d = dict(self.fix)
            age = None if self.t_fix is None else time.monotonic() - self.t_fix
        valid = age is not None and age < 2.0 and d.get("lat") is not None
        return {"gps_fix": ("yes" if valid else "NO") if d else "no data",
                "gps_sats": d.get("sats"), "gps_hdop": d.get("hdop"),
                "gps_lat": round(d["lat"], 6) if valid else None,
                "gps_lon": round(d["lon"], 6) if valid else None,
                "gps_speed_kmh": d.get("speed_kmh") if valid else None,
                "gps_course": d.get("course_deg") if valid else None,
                "gps_test": self.test_path.name if self.test_path else None,
                "gps_spread_m": self.spread(),
                **{"gps_" + k: v for k, v in self.sky_quality().items()}}

    def spread(self, n=None):
        """RMS distance to the mean of the last n test points (5 s at 5 Hz): dispersion while standing still."""
        n = n or max(5, int(5000 / C.GPS_RATE_MS))
        with self.lock:
            pts = self.points[-n:]
        if len(pts) < 5:
            return None
        me = sum(p[0] for p in pts) / len(pts)
        mn = sum(p[1] for p in pts) / len(pts)
        return round(math.sqrt(sum((e - me) ** 2 + (n_ - mn) ** 2 for e, n_ in pts) / len(pts)), 2)

    def track(self, max_points=3000):
        with self.lock:
            pts = self.points
            step = max(1, len(pts) // max_points)
            out = {"points": [(round(e, 2), round(n, 2)) for e, n in pts[::step]],
                   "marks": [(m, round(e, 2), round(n, 2)) for m, e, n in self.marks],
                   "testing": self._test_w is not None,
                   "file": self.test_path.name if self.test_path else None}
        out["spread_m"] = self.spread()
        out.update({k: v for k, v in self.status().items()
                    if k in ("gps_fix", "gps_sats", "gps_sats_view", "gps_cn0_top4", "gps_hdop", "gps_speed_kmh")})
        out["sky"] = self.sky_list()
        return out

    # ---- field test ----
    def start_test(self):
        with self.lock:
            if self._test_w is not None:
                return self.test_path
            out = C.BASE / "logs" / "gps"
            out.mkdir(parents=True, exist_ok=True)
            self.test_path = out / f"gpstest_{datetime.now():%Y%m%d_%H%M%S}.csv"
            self._test_f = open(self.test_path, "w", newline="")
            self._test_w = csv.DictWriter(self._test_f, fieldnames=self.FIELDS)
            self._test_w.writeheader()
            self._t0 = time.monotonic()
            self._origin = None
            self.points, self.marks, self._pending_mark = [], [], None
        print(f"GPS test: recording {self.test_path}")
        return self.test_path

    def stop_test(self):
        with self.lock:
            if self._test_w is None:
                return None
            self._test_f.close()
            self._test_f = self._test_w = None
            path, n = self.test_path, len(self.points)
        print(f"GPS test saved: {path} ({n} fixes, {len(self.marks)} marks)")
        return path

    def mark(self, label=None):
        with self.lock:
            if self._test_w is None:
                return None
            label = label or chr(ord("A") + len(self.marks) % 26)
            self._pending_mark = label
        return label

    def close(self):
        self.stop_test()
        self._running = False
        self._thread.join(timeout=1.0)
        self._ser.close()
