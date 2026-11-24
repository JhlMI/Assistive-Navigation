#!/usr/bin/env python3
"""Compass probe for the magnetometer inside the M8N GPS module (I2C bus 7, address 0x1E).

Supports HMC5883L (ID 'H43') and LIS3MDL (WHO_AM_I 0x3D), both answer at 0x1E.

Usage:
  python compass_probe.py                 # heading 5 times per second (uses calibration if present)
  python compass_probe.py --raw           # raw field in microtesla
  python compass_probe.py --calibrate 60  # rotate the harness slowly for 60 s, saves compass_cal.json
  python compass_probe.py --log out.csv   # also write a CSV
  python compass_probe.py --flip --mount 94   # sensor axis rotated on the harness

The sensor has no accelerometer: the heading is only valid with the module held level.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

try:
    from smbus2 import SMBus
except ImportError:
    sys.exit("Missing smbus2:  pip install smbus2")

BUS = 7
ADDR = 0x1E
CAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compass_cal.json")


def s16(hi, lo):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


class Hmc5883l:
    name = "HMC5883L"
    LSB_PER_GAUSS = 1090.0  # gain 1.3 Ga

    def __init__(self, bus):
        self.bus = bus
        bus.write_byte_data(ADDR, 0x00, 0x70)  # 8-sample average, 15 Hz
        bus.write_byte_data(ADDR, 0x01, 0x20)  # gain +-1.3 Ga
        bus.write_byte_data(ADDR, 0x02, 0x00)  # continuous mode
        time.sleep(0.07)

    def read(self):
        d = self.bus.read_i2c_block_data(ADDR, 0x03, 6)  # X, Z, Y big-endian
        x, z, y = s16(d[0], d[1]), s16(d[2], d[3]), s16(d[4], d[5])
        if -4096 in (x, y, z):
            return None  # overflow
        k = 100.0 / self.LSB_PER_GAUSS  # gauss -> microtesla
        return x * k, y * k, z * k


class Lis3mdl:
    name = "LIS3MDL"
    LSB_PER_GAUSS = 6842.0  # +-4 gauss

    def __init__(self, bus):
        self.bus = bus
        bus.write_byte_data(ADDR, 0x20, 0x70)  # XY ultra-high performance, 10 Hz
        bus.write_byte_data(ADDR, 0x21, 0x00)  # +-4 gauss
        bus.write_byte_data(ADDR, 0x23, 0x0C)  # Z ultra-high performance
        bus.write_byte_data(ADDR, 0x22, 0x00)  # continuous mode
        time.sleep(0.12)

    def read(self):
        d = self.bus.read_i2c_block_data(ADDR, 0x28 | 0x80, 6)  # X, Y, Z little-endian
        x, y, z = s16(d[1], d[0]), s16(d[3], d[2]), s16(d[5], d[4])
        k = 100.0 / self.LSB_PER_GAUSS
        return x * k, y * k, z * k


def detect(bus):
    try:
        ident = bus.read_i2c_block_data(ADDR, 0x0A, 3)
        if bytes(ident) == b"H43":
            return Hmc5883l(bus)
    except OSError:
        pass
    try:
        if bus.read_byte_data(ADDR, 0x0F) == 0x3D:
            return Lis3mdl(bus)
    except OSError:
        pass
    sys.exit(f"Device at 0x{ADDR:02X} on bus {BUS} is neither HMC5883L nor LIS3MDL")


def load_cal():
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE) as f:
            return json.load(f)
    return None


def apply_cal(v, cal):
    if not cal:
        return v
    x = (v[0] - cal["offset"][0]) * cal["scale"][0]
    y = (v[1] - cal["offset"][1]) * cal["scale"][1]
    return x, y, v[2] - cal["offset"][2]


def heading_deg(x, y, declination, flip, mount=0.0):
    h = math.degrees(math.atan2(y if not flip else -y, x))
    return (h - mount + declination) % 360.0


def calibrate(sensor, seconds):
    print(f"Rotate the harness slowly through a full turn (or two), level, for {seconds} s...")
    lo = [1e9] * 3
    hi = [-1e9] * 3
    n = 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        v = sensor.read()
        if v:
            for i in range(3):
                lo[i] = min(lo[i], v[i])
                hi[i] = max(hi[i], v[i])
            n += 1
        left = t_end - time.time()
        print(f"\r  {left:5.1f} s  samples {n}  X[{lo[0]:6.1f},{hi[0]:6.1f}]  Y[{lo[1]:6.1f},{hi[1]:6.1f}] uT",
              end="", flush=True)
        time.sleep(0.05)
    print()
    offset = [(hi[i] + lo[i]) / 2 for i in range(3)]
    radius = [(hi[i] - lo[i]) / 2 for i in range(2)]
    if min(radius) < 5:
        sys.exit("Range too small: the harness was not rotated enough. Repeat.")
    avg = sum(radius) / 2
    scale = [avg / radius[0], avg / radius[1]]
    cal = {"chip": sensor.name, "offset": offset, "scale": scale, "radius_uT": radius,
           "samples": n, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)
    ratio = max(radius) / min(radius)
    print(f"Saved {CAL_FILE}")
    print(f"  hard-iron offset X {offset[0]:.1f}  Y {offset[1]:.1f}  Z {offset[2]:.1f} uT")
    print(f"  radius X {radius[0]:.1f}  Y {radius[1]:.1f} uT   (X/Y ratio {ratio:.2f}; >1.3 = strong soft-iron)")


def main():
    ap = argparse.ArgumentParser(description="M8N compass probe")
    ap.add_argument("--raw", action="store_true", help="print raw field only")
    ap.add_argument("--calibrate", type=float, metavar="S", help="hard/soft-iron calibration for S seconds")
    ap.add_argument("--declination", type=float, default=0.0, help="magnetic declination in degrees (east +)")
    ap.add_argument("--flip", action="store_true", help="invert Y if the heading turns the wrong way")
    ap.add_argument("--mount", type=float, default=0.0,
                    help="mounting offset in degrees: heading shown when facing magnetic north (after --flip)")
    ap.add_argument("--rate", type=float, default=5.0, help="lines per second")
    ap.add_argument("--log", metavar="CSV", help="write samples to CSV")
    a = ap.parse_args()

    with SMBus(BUS) as bus:
        sensor = detect(bus)
        print(f"Chip: {sensor.name} at 0x{ADDR:02X}, bus {BUS}")
        if a.calibrate:
            calibrate(sensor, a.calibrate)
            return
        cal = None if a.raw else load_cal()
        if not a.raw:
            print("Calibration: " + (f"loaded ({cal['date']})" if cal else "NONE (run --calibrate 60)"))
        writer = None
        if a.log:
            fh = open(a.log, "w", newline="")
            writer = csv.writer(fh)
            writer.writerow(["t_s", "x_uT", "y_uT", "z_uT", "norm_uT", "heading_deg"])
        t0 = time.time()
        try:
            while True:
                v = sensor.read()
                if v is None:
                    print("overflow")
                else:
                    c = apply_cal(v, cal)
                    norm = math.sqrt(sum(i * i for i in v))
                    h = heading_deg(c[0], c[1], a.declination, a.flip, a.mount)
                    if a.raw:
                        print(f"X {v[0]:7.1f}  Y {v[1]:7.1f}  Z {v[2]:7.1f}  |B| {norm:5.1f} uT")
                    else:
                        print(f"heading {h:6.1f} deg   |B| {norm:5.1f} uT")
                    if writer:
                        writer.writerow([f"{time.time() - t0:.2f}", f"{v[0]:.2f}", f"{v[1]:.2f}",
                                         f"{v[2]:.2f}", f"{norm:.2f}", f"{h:.1f}"])
                time.sleep(1.0 / a.rate)
        except KeyboardInterrupt:
            pass
        finally:
            if writer:
                fh.close()


if __name__ == "__main__":
    main()
