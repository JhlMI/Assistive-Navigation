"""GPS state filter for macro-navigation: position/velocity Kalman filter + compass heading smoother.

State x = [e, n, ve, vn] in meters / m/s on a local ENU plane (origin = first accepted fix),
constant-velocity model with white acceleration noise (walking pedestrian).

Measurements:
  - GPS position (every GGA). sigma = POS_SIGMA_M scaled up with HDOP and weak signal, and
    inflated during the first SETTLE_S seconds (the receiver is still converging).
  - GPS velocity from RMC speed/course, only when speed >= VEL_MIN_KMH (below that the course is noise).
  - Zero-velocity update (ZUPT) when the GPS speed stays below STILL_KMH for STILL_HOLD_S:
    keeps the position from drifting while the user waits at a corner.
Outlier gating: position innovations beyond the 99.9 % chi-square bound are rejected; after
MAX_REJECT consecutive rejections the filter accepts the fix again (avoids locking onto a bad state).

What it cannot do: remove a constant bias (urban canyon multipath shifts every fix the same way,
so the filter follows it). It removes noise and jumps, not bias.

Heading: the compass (not the GPS course) gives the direction, also when standing still.
HeadingFilter smooths it on the unit circle with time constant HEADING_TAU_S.
"""
import math

EARTH_R = 6371000.0

# ---- defaults (tune with gps_lab data; see kalman_replay.py) ----
POS_SIGMA_M = 3.0          # GPS position noise at HDOP 1 and good signal (static test sigma)
ACCEL_SIGMA = 0.15         # m/s^2, walking accelerations (process noise); tuned on the 2026-11-23 street route
VEL_SIGMA = 0.35           # m/s, GPS Doppler velocity noise
VEL_MIN_KMH = 2.0          # use GPS velocity only above this speed
STILL_KMH = 1.0            # GPS speed below this ...
STILL_HOLD_S = 1.0         # ... for this long -> standing still (ZUPT)
ZUPT_SIGMA = 0.05          # m/s
POS_WALK = 0.5             # m/sqrt(s): position random walk, so the filter keeps following the slow
                           # GPS wander (correlated errors) instead of freezing when standing still
SETTLE_S = 60.0            # inflate the position noise during the first seconds after start
SETTLE_FACTOR = 3.0
GATE_CHI2 = 13.8           # 2 dof, 99.9 %
MAX_REJECT = 5
HEADING_TAU_S = 0.6


class GpsKalman:
    def __init__(self, pos_sigma=POS_SIGMA_M, accel_sigma=ACCEL_SIGMA):
        self.pos_sigma, self.accel_sigma = pos_sigma, accel_sigma
        self.origin = None
        self.x = [0.0, 0.0, 0.0, 0.0]
        self.P = None
        self.t = None
        self.t0 = None
        self.still_since = None
        self.still = False
        self.rejects = 0
        self.n_rejected = 0

    # ---- coordinates ----
    def to_local(self, lat, lon):
        lat0, lon0 = self.origin
        k = math.pi / 180.0 * EARTH_R
        return (lon - lon0) * k * math.cos(math.radians(lat0)), (lat - lat0) * k

    def to_latlon(self, e, n):
        lat0, lon0 = self.origin
        k = math.pi / 180.0 * EARTH_R
        return lat0 + n / k, lon0 + e / (k * math.cos(math.radians(lat0)))

    # ---- small 4x4 linear algebra (no numpy: runs anywhere) ----
    @staticmethod
    def _mm(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

    @staticmethod
    def _t(A):
        return [list(r) for r in zip(*A)]

    @staticmethod
    def _inv2(S):
        a, b, c, d = S[0][0], S[0][1], S[1][0], S[1][1]
        det = a * d - b * c
        return [[d / det, -b / det], [-c / det, a / det]]

    def _predict(self, dt):
        if dt <= 0:
            return
        F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        self.x = [sum(F[i][j] * self.x[j] for j in range(4)) for i in range(4)]
        q = self.accel_sigma ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        w = POS_WALK ** 2 * dt
        Q = [[dt4 / 4 * q + w, 0, dt3 / 2 * q, 0], [0, dt4 / 4 * q + w, 0, dt3 / 2 * q],
             [dt3 / 2 * q, 0, dt2 * q, 0], [0, dt3 / 2 * q, 0, dt2 * q]]
        FP = self._mm(F, self.P)
        self.P = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self._mm(FP, self._t(F)), Q)]

    def _update(self, H, z, R, gate=False):
        """H: 2x4, z: 2, R: 2x2. Returns True if applied."""
        y = [z[i] - sum(H[i][j] * self.x[j] for j in range(4)) for i in range(2)]
        PHt = self._mm(self.P, self._t(H))
        S = self._mm(H, PHt)
        S = [[S[i][j] + R[i][j] for j in range(2)] for i in range(2)]
        Si = self._inv2(S)
        if gate:
            d2 = sum(y[i] * Si[i][j] * y[j] for i in range(2) for j in range(2))
            if d2 > GATE_CHI2 and self.rejects < MAX_REJECT:
                self.rejects += 1
                self.n_rejected += 1
                return False
            self.rejects = 0
        K = self._mm(PHt, Si)
        self.x = [self.x[i] + sum(K[i][j] * y[j] for j in range(2)) for i in range(4)]
        KH = self._mm(K, H)
        IKH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(4)] for i in range(4)]
        self.P = self._mm(IKH, self.P)
        return True

    def meas_sigma(self, hdop=None, cn0=None, t=None):
        s = self.pos_sigma * max(1.0, (hdop or 1.0))
        if cn0 is not None:
            s *= 1.0 if cn0 >= 33 else 1.3 if cn0 >= 28 else 1.7
        if t is not None and self.t0 is not None and t - self.t0 < SETTLE_S:
            s *= SETTLE_FACTOR
        return s

    def step(self, t, lat=None, lon=None, hdop=None, cn0=None, speed_kmh=None, course_deg=None):
        """Feed one GPS epoch (fields may be None when there is no fix). Returns the state dict."""
        if self.origin is None:
            if lat is None:
                return None
            self.origin = (lat, lon)
            self.x = [0.0, 0.0, 0.0, 0.0]
            s = self.meas_sigma(hdop, cn0) * SETTLE_FACTOR
            self.P = [[s * s, 0, 0, 0], [0, s * s, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
            self.t = self.t0 = t
            return self.state()
        self._predict(t - self.t)
        self.t = t
        # standing still?
        if speed_kmh is not None and speed_kmh < STILL_KMH:
            self.still_since = self.still_since if self.still_since is not None else t
        else:
            self.still_since = None
        self.still = self.still_since is not None and t - self.still_since >= STILL_HOLD_S
        H_pos = [[1, 0, 0, 0], [0, 1, 0, 0]]
        H_vel = [[0, 0, 1, 0], [0, 0, 0, 1]]
        if lat is not None:
            e, n = self.to_local(lat, lon)
            s = self.meas_sigma(hdop, cn0, t)
            self._update(H_pos, [e, n], [[s * s, 0], [0, s * s]], gate=True)
        if self.still:
            self._update(H_vel, [0.0, 0.0], [[ZUPT_SIGMA ** 2, 0], [0, ZUPT_SIGMA ** 2]])
        elif speed_kmh is not None and course_deg is not None and speed_kmh >= VEL_MIN_KMH:
            v = speed_kmh / 3.6
            c = math.radians(course_deg)
            self._update(H_vel, [v * math.sin(c), v * math.cos(c)], [[VEL_SIGMA ** 2, 0], [0, VEL_SIGMA ** 2]])
        return self.state()

    def state(self):
        e, n, ve, vn = self.x
        lat, lon = self.to_latlon(e, n)
        spd = math.hypot(ve, vn)
        return {"e": e, "n": n, "lat": lat, "lon": lon, "speed_kmh": spd * 3.6,
                "course_deg": (math.degrees(math.atan2(ve, vn)) % 360.0) if spd > 0.3 else None,
                "pos_sigma_m": math.sqrt(max(self.P[0][0], self.P[1][1])), "still": self.still}


class HeadingFilter:
    """Exponential smoothing of the compass heading on the unit circle."""

    def __init__(self, tau=HEADING_TAU_S):
        self.tau = tau
        self.s = self.c = None
        self.t = None

    def step(self, t, heading_deg):
        if heading_deg is None:
            return self.value()
        hs, hc = math.sin(math.radians(heading_deg)), math.cos(math.radians(heading_deg))
        if self.s is None:
            self.s, self.c, self.t = hs, hc, t
            return heading_deg % 360.0
        a = 1.0 - math.exp(-max(0.0, t - self.t) / self.tau)
        self.s += a * (hs - self.s)
        self.c += a * (hc - self.c)
        self.t = t
        return self.value()

    def value(self):
        if self.s is None:
            return None
        return math.degrees(math.atan2(self.s, self.c)) % 360.0
