"""Magnetometer of the M8N GPS module (HMC5883L or LIS3MDL at 0x1E on I2C bus 7).

- Compass: background reader (~15 Hz) with 2D hard/soft-iron calibration, mounting offset and
  axis flip. read() returns the latest heading (circular mean of the last samples) and |B|.
- calibration_from(): builds the calibration from raw samples taken while turning around.
- analyze_*(): analysis of the compass tests recorded by gps_lab.py.

The sensor has no accelerometer: the heading is valid only with the module roughly level.
Calibration file (JSON): offset[3], scale[2], radius_uT[2], and optionally flip / mount_deg,
which override config.py when present.
"""
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

ADDR = 0x1E


def _s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


class _Hmc5883l:
    name = "HMC5883L"

    def __init__(self, bus):
        self.bus = bus
        bus.write_byte_data(ADDR, 0x00, 0x70)  # 8-sample average, 15 Hz
        bus.write_byte_data(ADDR, 0x01, 0x20)  # gain +-1.3 Ga (1090 LSB/G)
        bus.write_byte_data(ADDR, 0x02, 0x00)  # continuous
        time.sleep(0.07)

    def read(self):
        d = self.bus.read_i2c_block_data(ADDR, 0x03, 6)  # X, Z, Y big-endian
        x, z, y = _s16(d[0], d[1]), _s16(d[2], d[3]), _s16(d[4], d[5])
        if -4096 in (x, y, z):
            return None
        k = 100.0 / 1090.0
        return x * k, y * k, z * k


class _Lis3mdl:
    name = "LIS3MDL"

    def __init__(self, bus):
        self.bus = bus
        bus.write_byte_data(ADDR, 0x20, 0x70)
        bus.write_byte_data(ADDR, 0x21, 0x00)  # +-4 G (6842 LSB/G)
        bus.write_byte_data(ADDR, 0x23, 0x0C)
        bus.write_byte_data(ADDR, 0x22, 0x00)
        time.sleep(0.12)

    def read(self):
        d = self.bus.read_i2c_block_data(ADDR, 0x28 | 0x80, 6)
        x, y, z = _s16(d[1], d[0]), _s16(d[3], d[2]), _s16(d[5], d[4])
        k = 100.0 / 6842.0
        return x * k, y * k, z * k


def _detect(bus):
    try:
        if bytes(bus.read_i2c_block_data(ADDR, 0x0A, 3)) == b"H43":
            return _Hmc5883l(bus)
    except OSError:
        pass
    try:
        if bus.read_byte_data(ADDR, 0x0F) == 0x3D:
            return _Lis3mdl(bus)
    except OSError:
        pass
    raise RuntimeError(f"no HMC5883L/LIS3MDL at 0x{ADDR:02X}")


# ---- math -----------------------------------------------------------------------------
def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def circ_mean(angles):
    a = [x for x in angles if x is not None]
    if not a:
        return None
    s = sum(math.sin(math.radians(x)) for x in a)
    c = sum(math.cos(math.radians(x)) for x in a)
    return math.degrees(math.atan2(s, c)) % 360.0


def heading_from(raw, cal, flip, mount, declination=0.0):
    x, y = raw[0], raw[1]
    if cal:
        x = (x - cal["offset"][0]) * cal["scale"][0]
        y = (y - cal["offset"][1]) * cal["scale"][1]
    h = math.degrees(math.atan2(-y if flip else y, x))
    return (h - mount + declination) % 360.0


def calibration_from(samples):
    """samples: list of (x, y, z) in uT taken while turning around, level. Returns cal dict or error."""
    if len(samples) < 30:
        return {"error": f"muy pocas muestras ({len(samples)})"}
    lo = [min(s[i] for s in samples) for i in range(3)]
    hi = [max(s[i] for s in samples) for i in range(3)]
    offset = [(hi[i] + lo[i]) / 2 for i in range(3)]
    radius = [(hi[i] - lo[i]) / 2 for i in range(2)]
    if min(radius) < 1.0:
        return {"error": "rango muy pequeño: no se completó la vuelta"}
    # coverage: how much of the circle was swept (needs ~full turn)
    bins = {int(((math.degrees(math.atan2((s[1] - offset[1]) / radius[1], (s[0] - offset[0]) / radius[0]))
                  + 360) % 360) // 30) for s in samples}
    avg = sum(radius) / 2
    return {"offset": [round(v, 3) for v in offset], "scale": [round(avg / radius[0], 4), round(avg / radius[1], 4)],
            "radius_uT": [round(r, 2) for r in radius], "ratio_xy": round(max(radius) / min(radius), 3),
            "coverage_pct": round(100 * len(bins) / 12), "samples": len(samples),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---- reader ---------------------------------------------------------------------------
class Compass:
    def __init__(self, bus=7, cal_file=None, flip=False, mount=0.0, declination=0.0, smooth=5):
        from smbus2 import SMBus
        self._bus = SMBus(bus)
        self.chip = _detect(self._bus)
        self.cal_file = Path(cal_file) if cal_file else None
        self.flip, self.mount, self.declination = flip, mount, declination
        self.cal = None
        if self.cal_file and self.cal_file.exists():
            self.cal = json.loads(self.cal_file.read_text())
            self.flip = self.cal.get("flip", self.flip)
            self.mount = self.cal.get("mount_deg", self.mount)
        self._lock = threading.Lock()
        self._raw = None
        self._hist = deque(maxlen=smooth)
        self.t_read = 0.0
        self.errors = 0
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            try:
                v = self.chip.read()
            except OSError:
                self.errors += 1
                v = None
            if v:
                h = heading_from(v, self.cal, self.flip, self.mount, self.declination)
                with self._lock:
                    self._raw = v
                    self._hist.append(h)
                    self.t_read = time.monotonic()
            time.sleep(1 / 15)

    def read(self):
        with self._lock:
            if self._raw is None or time.monotonic() - self.t_read > 1.0:
                return {"compass_deg": None, "compass_b_ut": None, "mag_x": None, "mag_y": None, "mag_z": None}
            x, y, z = self._raw
            h = circ_mean(self._hist)
        return {"compass_deg": round(h, 1), "compass_b_ut": round(math.sqrt(x * x + y * y + z * z), 2),
                "mag_x": round(x, 2), "mag_y": round(y, 2), "mag_z": round(z, 2)}

    def _save(self):
        if self.cal_file and self.cal:
            self.cal_file.parent.mkdir(parents=True, exist_ok=True)
            if self.cal_file.exists():
                self.cal_file.with_suffix(".prev.json").write_text(self.cal_file.read_text())
            self.cal_file.write_text(json.dumps(self.cal, indent=2))

    def set_calibration(self, cal):
        with self._lock:
            self.cal = {**cal, "flip": self.flip, "mount_deg": self.mount}
            self._hist.clear()
        self._save()

    def set_mount(self, mount, flip=None):
        with self._lock:
            self.mount = round(mount % 360.0, 1)
            if flip is not None:
                self.flip = flip
            if self.cal is not None:
                self.cal["mount_deg"], self.cal["flip"] = self.mount, self.flip
            self._hist.clear()
        self._save()

    def info(self):
        return {"chip": self.chip.name, "calibrated": self.cal is not None,
                "cal_date": (self.cal or {}).get("date"), "flip": self.flip, "mount_deg": self.mount}

    def close(self):
        self._run = False
        time.sleep(0.1)
        self._bus.close()


# ---- analysis of gps_lab compass tests -----------------------------------------------
DIRS = {"N": 0.0, "E": 90.0, "S": 180.0, "O": 270.0}


def _vals(rows, phase, key):
    return [r[key] for r in rows if r.get("phase") == phase and r.get(key) is not None]


def _mean(v):
    return sum(v) / len(v) if v else None


def analyze_check(rows, flip, mount):
    """Compass vs the four cardinal directions given by the phone (hold_N/E/S/O phases)."""
    out = {"test": "compass_check"}
    errs, bs, signs = {}, [], []
    for d, truth in DIRS.items():
        h = _vals(rows, f"hold_{d}", "compass_deg")
        b = _vals(rows, f"hold_{d}", "compass_b_ut")
        if len(h) < 3:
            continue
        m = circ_mean(h)
        e = angle_diff(m, truth)
        errs[d] = e
        bs += b
        out[f"dir_{d}"] = {"heading_deg": round(m, 1), "error_deg": round(e, 1),
                           "noise_deg": round(max(abs(angle_diff(x, m)) for x in h), 1),
                           "b_ut": round(_mean(b), 1)}
    if len(errs) < 2:
        out["error"] = "faltan direcciones"
        return out
    # rotation sense: walking N->E->S->O the measured heading must increase
    order = [d for d in DIRS if d in errs]
    heads = [out[f"dir_{d}"]["heading_deg"] for d in order]
    steps = [angle_diff(heads[i + 1], heads[i]) for i in range(len(heads) - 1)]
    inverted = sum(1 for s in steps if s < 0) > len(steps) / 2
    out["sense_inverted"] = inverted
    if inverted:
        out["note"] = "el sentido de giro está invertido: aplica la corrección (cambia flip)"
        return out
    bias = circ_mean(list(errs.values()))
    bias = angle_diff(bias, 0.0)
    resid = {d: angle_diff(e, bias) for d, e in errs.items()}
    out["bias_deg"] = round(bias, 1)
    out["error_max_raw_deg"] = round(max(abs(e) for e in errs.values()), 1)
    out["error_max_after_bias_deg"] = round(max(abs(r) for r in resid.values()), 1)
    out["b_range_ut"] = f"{min(bs):.1f} – {max(bs):.1f}" if bs else None
    out["suggested_mount_deg"] = round((mount + bias) % 360.0, 1)
    return out


def analyze_motors(rows):
    phases = ["m_off", "m_left", "m_right", "m_both", "m_off_end"]
    base_h = _vals(rows, "m_off", "compass_deg")[3:]
    base_b = _vals(rows, "m_off", "compass_b_ut")[3:]
    out = {"test": "motors"}
    if len(base_h) < 3:
        out["error"] = "sin datos de brújula en la fase apagada"
        return out
    h0, b0 = circ_mean(base_h), _mean(base_b)
    worst = 0.0
    for p in phases:
        h = _vals(rows, p, "compass_deg")[3:]
        b = _vals(rows, p, "compass_b_ut")[3:]
        if len(h) < 3:
            continue
        shift = angle_diff(circ_mean(h), h0)
        if p != "m_off":
            worst = max(worst, abs(shift))
        out[p] = {"shift_deg": round(shift, 1), "max_deg": round(max(abs(angle_diff(x, h0)) for x in h), 1),
                  "b_ut": round(_mean(b), 2), "db_ut": round(_mean(b) - b0, 2)}
    out["worst_shift_deg"] = round(worst, 1)
    out["verdict"] = "sin interferencia apreciable" if worst < 2.5 else "interferencia: leer la brújula entre pulsos"
    return out


def compass_walk(rows, seg, true_brg):
    """Compass while walking a segment: bias/noise vs the true bearing and vs the GPS course."""
    w = [r for r in rows if r.get("phase") == seg and r.get("compass_deg") is not None]
    if len(w) < 5:
        return None
    d_true = [angle_diff(r["compass_deg"], true_brg) for r in w] if true_brg is not None else []
    d_gps = [angle_diff(r["compass_deg"], r["course_deg"]) for r in w
             if r.get("course_deg") is not None and (r.get("speed_kmh") or 0) >= 2.0]

    def stats(d):
        if len(d) < 3:
            return None, None
        m = _mean(d)
        return round(m, 1), round(math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)), 1)
    out = {}
    out["compass_bias_deg"], out["compass_sigma_deg"] = stats(d_true)
    out["compass_vs_gps_bias_deg"], out["compass_vs_gps_sigma_deg"] = stats(d_gps)
    return out
