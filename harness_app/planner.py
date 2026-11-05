"""Bird's-eye view planner: occupancy grid + path along the sidewalk.

Ground frame (meters): x lateral (right +, 0 = body center), y forward from the feet.

Per frame:
  1. Walkable mask -> bird's-eye grid (5 cm cells), smoothed over time.
  2. Obstacles projected where they touch the ground:
       - detector boxes (person, car, ...): bottom-center of the box
       - poles: lowest pixel of each 'pole' blob from the segmentation
  3. Clearance = distance from every free cell to the nearest non-walkable cell or obstacle.
  4. Path: row by row (near -> far), follow the cell with the most clearance that is
     connected to the previous row. The path is the "center line" of the free space.
  5. Command from the angle to a look-ahead point on the path (hysteresis + debounce):
       F straight, L / R correct, X stop (obstacle on the path too close, or no path).
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import bev
import config as C

POLE_ID = 2                         # [background, building, pole, road, sidewalk, vegetation]


@dataclass
class PlanResult:
    command: str = "S"
    reason: str = "start"
    angle_deg: float = 0.0          # smoothed heading correction, + = turn right
    error: float = 0.0              # angle normalized to -1..1 (kept for the CSV/web)
    sidewalk_frac: float = 0.0      # walkable fraction of the visible grid
    center_x: float = None          # not used in BEV mode (camera overlay draws the path)
    path: list = field(default_factory=list)          # [(x, y)] meters
    path_px: list = field(default_factory=list)       # same path in image pixels (full res)
    look: tuple = None                                # look-ahead point (x, y)
    obstacles_ground: list = field(default_factory=list)   # [(x_left, x_right, y_front, label)] meters
    obstacles: list = field(default_factory=list)     # detector boxes that block the path
    obst_dist: float = None         # distance along the path to the first blocking obstacle
    path_len: float = 0.0           # how far the path reaches (m from the feet)
    grid: np.ndarray = None         # 0 unknown, 1 walkable, 2 not walkable, 3 obstacle
    stop_zone: list = None          # STOP zone outline [(x, y)] meters
    stop_zone_px: list = None       # same outline in image pixels (full res)


class BevPlanner:
    def __init__(self, H=None):
        if H is None:
            H, _ = bev.load_calibration()
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.res = bev.GRID_RES
        self.A = bev.ground_to_grid_matrix()                 # ground -> grid cell
        self.gw, self.gh = bev.grid_size()
        # Cells the camera can see (warp of an all-ones image)
        ones = np.ones((C.SEG_H - 8, C.SEG_W), np.uint8)
        self.visible = bev.warp_to_grid(ones, H, interp=cv2.INTER_NEAREST).astype(bool)
        self.free_ema = np.zeros((self.gh, self.gw), np.float32)
        self.col0 = int(round(self.A[0, 2]))                 # grid column of x = 0
        # First visible row near the feet (blind zone limit) on the center column
        rows = np.where(self.visible[:, self.col0])[0]
        self.row_start = int(rows.max()) if len(rows) else self.gh - 1
        self.y_start = self.row_to_y(self.row_start)
        # Stop zone: the first STOP_RANGE meters the camera sees (beyond the blind zone)
        self.stop_y = self.y_start + C.STOP_RANGE
        # Decision state
        self.angle_ema = 0.0
        self.direction = "F"
        self.candidate, self.candidate_count = "F", 0
        self.block_frames = self.clear_frames = 0
        self.block_alert = False
        self.nopath_frames = 0

    # ---- coordinate helpers --------------------------------------------------
    def row_to_y(self, row):
        return (self.A[1, 2] - row) * self.res

    def col_to_x(self, col):
        return (col - self.A[0, 2]) * self.res

    def ground_to_cell(self, x, y):
        return int(round(self.A[0, 0] * x + self.A[0, 2])), int(round(self.A[1, 1] * y + self.A[1, 2]))

    def ground_to_px(self, x, y):
        p = self.H_inv @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    # ---- obstacles -----------------------------------------------------------
    def _box_obstacles(self, boxes):
        out = []
        for (x1, y1, x2, y2, score, cls) in boxes:
            if cls not in C.OBSTACLE_CLASSES:
                continue
            v = min(y2, 1079.0)
            if v >= 1075:                                    # touches the bottom: inside the blind zone
                gx, _ = bev.image_to_ground(self.H, (x1 + x2) / 2, 1079)
                gy = self.y_start
            else:
                gx, gy = bev.image_to_ground(self.H, (x1 + x2) / 2, v)
            # Ground footprint = the box's bottom edge projected on the floor (left and right ends).
            # The bottom edge is the part touching the ground (feet, wheels), i.e. the closest part.
            xl, _ = bev.image_to_ground(self.H, x1, v)
            xr, _ = bev.image_to_ground(self.H, x2, v)
            xl, xr = min(xl, xr), max(xl, xr)
            if 0 < gy < bev.GRID_Y[1] + 1:
                out.append((xl, xr, gy, C.OBSTACLE_CLASSES[cls], (x1, y1, x2, y2, score, cls)))
        return out

    def _pole_obstacles(self, seg_mask):
        """Lowest pixel of each pole blob = where it meets the ground."""
        if seg_mask is None:
            return []
        poles = (seg_mask == POLE_ID).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(poles, connectivity=8)
        out = []
        sx, sy = 1920.0 / seg_mask.shape[1], 1080.0 / seg_mask.shape[0]
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < C.POLE_MIN_AREA:
                continue
            ys, xs = np.where(labels == i)
            bottom = ys.max()
            u = xs[ys >= bottom - 2].mean() * sx
            v = bottom * sy
            if v >= 1075:
                continue                                     # foot not visible: cannot place it
            gx, gy = bev.image_to_ground(self.H, u, v)
            if 0 < gy < bev.GRID_Y[1]:
                out.append((gx - C.POLE_RADIUS, gx + C.POLE_RADIUS, gy - C.POLE_RADIUS, "pole", None))
        return out

    # ---- main step -----------------------------------------------------------
    def update(self, walkable, boxes, img_shape, seg_mask=None):
        r = PlanResult()
        res = self.res

        # 1. Walkable grid, smoothed over time (only where visible)
        free = bev.warp_to_grid(walkable, self.H, interp=cv2.INTER_NEAREST).astype(np.float32)
        a = C.GRID_EMA
        self.free_ema[self.visible] = a * free[self.visible] + (1 - a) * self.free_ema[self.visible]
        free_s = (self.free_ema > 0.5) & self.visible
        r.sidewalk_frac = float(free_s.sum() / max(1, self.visible.sum()))

        # 2. Obstacles
        obstacles = self._box_obstacles(boxes) + self._pole_obstacles(seg_mask)
        occ = np.zeros((self.gh, self.gw), np.uint8)
        for (xl, xr, yf, label, _) in obstacles:
            c1, r1 = self.ground_to_cell(xl, yf + C.OBST_DEPTH)          # far-left corner
            c2, r2 = self.ground_to_cell(xr, yf)                         # near-right corner
            cv2.rectangle(occ, (c1, r1), (c2, r2), 1, -1)
        r.obstacles_ground = [(xl, xr, yf, label) for (xl, xr, yf, label, _) in obstacles]

        # 3. Clearance: distance to non-walkable visible cells or obstacles
        #    (cells outside the field of view do not repel: they are unknown, not blocked)
        blocked = ((self.visible & ~free_s) | occ.astype(bool))
        clearance = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 3) * res
        navigable = free_s & ~occ.astype(bool) & (clearance >= C.BODY_HALF_WIDTH)

        grid = np.zeros((self.gh, self.gw), np.uint8)
        grid[self.visible] = 2
        grid[free_s] = 1
        grid[occ.astype(bool)] = 3
        r.grid = grid

        # 4. Path, row by row from the blind-zone limit forwards
        max_jump = max(1, int(round(C.PATH_MAX_STEP / res)))
        row_end = int(round(self.A[1, 2] - C.PATH_MAX_Y / res))
        prev = self.col0
        cols, rows = [], []
        misses = 0
        for row in range(self.row_start, row_end - 1, -1):
            idx = np.where(navigable[row])[0]
            if len(idx) == 0:
                misses += 1
                if misses * res > C.PATH_GAP_TOL:
                    break
                continue
            misses = 0
            # runs of consecutive navigable cells; take the run closest to the previous column
            splits = np.where(np.diff(idx) > 1)[0] + 1
            runs = np.split(idx, splits)
            run = min(runs, key=lambda rr: 0 if rr[0] <= prev <= rr[-1] else min(abs(rr[0] - prev), abs(rr[-1] - prev)))
            if not cols and min(abs(run[0] - prev), abs(run[-1] - prev)) > C.START_SEARCH / res \
                    and not (run[0] <= prev <= run[-1]):
                continue                                     # nothing walkable near the body yet
            cl = clearance[row, run]
            if cl.max() >= C.CLEAR_CAP:                    # wide area: center of the "safe enough" plateau
                best = int(round(run[cl >= C.CLEAR_CAP].mean()))
            else:                                            # narrow: the cell farthest from both edges
                best = int(run[np.argmax(cl)])
            best = int(np.clip(best, prev - max_jump, prev + max_jump)) if cols else best
            cols.append(best)
            rows.append(row)
            prev = best

        if len(cols) >= 3:
            # Smooth the center line: low-order polynomial x(y) (steering needs a smooth curve)
            ys_m = np.array([self.row_to_y(rw) for rw in rows])
            xs_m = np.array([self.col_to_x(c) for c in cols])
            deg = min(C.PATH_POLY_DEG, 1 if len(cols) < 10 else (2 if len(cols) < 30 else 3))
            xs_fit = np.polyval(np.polyfit(ys_m, xs_m, deg), ys_m)
            r.path = list(zip(xs_fit.tolist(), ys_m.tolist()))
            r.path_px = [self.ground_to_px(x, y) for x, y in r.path]
            r.path_len = r.path[-1][1]

        # 5. STOP zone: which obstacles count as "in front of the user"
        #    bubble:   half-ellipse in front of the body, centered on the body axis at the blind-zone edge:
        #              BUBBLE_DEPTH forward, BUBBLE_HALF_WIDTH to each side (fixed to the body, not the path)
        #    corridor: the path center +- (body half width + margin), within STOP_RANGE
        r.obst_dist = None
        half = C.BODY_HALF_WIDTH + C.OBST_MARGIN
        for (xl, xr, yf, label, box) in obstacles:
            if C.STOP_ZONE == "bubble":
                dx = 0.0 if xl <= 0.0 <= xr else min(abs(xl), abs(xr))   # closest point of the footprint
                dy = max(0.0, yf - self.y_start)
                inside = (dx / C.BUBBLE_HALF_WIDTH) ** 2 + (dy / C.BUBBLE_DEPTH) ** 2 <= 1.0
            else:
                if not r.path or yf > r.path_len + 0.3 or yf >= self.stop_y:
                    continue
                xp = min(r.path, key=lambda p: abs(p[1] - yf))[0]   # path center at that distance
                inside = xr >= xp - half and xl <= xp + half
            if inside:
                if r.obst_dist is None or yf < r.obst_dist:
                    r.obst_dist = yf
                if box is not None:
                    r.obstacles.append(box)

        # Zone outline (ground meters) for the BEV panel and the camera overlay
        if C.STOP_ZONE == "bubble":
            a = np.linspace(0, np.pi, 25)
            r.stop_zone = [(C.BUBBLE_HALF_WIDTH * np.cos(t), self.y_start + C.BUBBLE_DEPTH * np.sin(t)) for t in a]
        elif r.path:
            xs0 = [x for (x, y) in r.path if y <= self.stop_y] or [r.path[0][0]]
            xc = float(np.mean(xs0))
            r.stop_zone = [(xc - half, self.y_start), (xc - half, self.stop_y),
                           (xc + half, self.stop_y), (xc + half, self.y_start)]
        if r.stop_zone:
            r.stop_zone_px = [self.ground_to_px(x, y) for (x, y) in r.stop_zone]

        # 6. Decision
        if r.obst_dist is not None:                        # something inside the STOP zone
            self.block_frames += 1
            self.clear_frames = 0
        else:
            self.clear_frames += 1
            self.block_frames = 0
        if self.block_frames >= C.FRAMES_OBST_ON:
            self.block_alert = True
        elif self.clear_frames >= C.FRAMES_OBST_OFF:
            self.block_alert = False

        short = (not r.path) or (r.path_len - self.y_start < C.MIN_PATH_LEN)
        self.nopath_frames = self.nopath_frames + 1 if short else 0

        if r.path:
            look_y = min(C.LOOKAHEAD_Y, r.path_len)
            look = min(r.path, key=lambda p: abs(p[1] - look_y))
            r.look = look
            raw = math.degrees(math.atan2(look[0], max(look[1], 0.5)))
            self.angle_ema = C.ANGLE_EMA * raw + (1 - C.ANGLE_EMA) * self.angle_ema
        r.angle_deg = self.angle_ema
        r.error = float(np.clip(self.angle_ema / 30.0, -1, 1))

        if self.block_alert:
            r.command, r.reason = "X", f"obstacle at {r.obst_dist:.1f} m" if r.obst_dist else "obstacle"
        elif self.nopath_frames >= C.FRAMES_NO_SIDEWALK:
            r.command, r.reason = "X", "no walkable path"
        elif short:
            r.command, r.reason = self.direction, "path lost (brief)"
        else:
            r.command = self._direction(self.angle_ema)
            r.reason = {"F": "on path", "L": "path to the left", "R": "path to the right"}[r.command] \
                + f" ({self.angle_ema:+.0f} deg)"

        if r.command == "F" and not C.FORWARD_VIBRATES:
            r.command = "S"
        return r

    def _direction(self, angle):
        cur = self.direction
        if cur == "F":
            prop = "L" if angle < -C.ANGLE_ENTER else "R" if angle > C.ANGLE_ENTER else "F"
        elif cur == "L":
            prop = "L" if angle < -C.ANGLE_EXIT else ("R" if angle > C.ANGLE_ENTER else "F")
        else:
            prop = "R" if angle > C.ANGLE_EXIT else ("L" if angle < -C.ANGLE_ENTER else "F")
        if prop == cur:
            self.candidate_count = 0
            return cur
        if prop == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = prop, 1
        if self.candidate_count >= C.FRAMES_TO_SWITCH:
            self.direction, self.candidate_count = prop, 0
        return self.direction


def render_grid(res, height=360):
    """Bird's-eye panel (BGR) for the web view."""
    if res is None or res.grid is None:
        return None
    colors = np.array([[25, 25, 25], [60, 160, 60], [70, 70, 110], [40, 40, 220]], np.uint8)
    img = colors[res.grid]
    scale = height / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), height), interpolation=cv2.INTER_NEAREST)
    A = bev.ground_to_grid_matrix()

    def to_px(x, y):
        return int((A[0, 0] * x + A[0, 2]) * scale), int((A[1, 1] * y + A[1, 2]) * scale)

    for y in range(1, int(bev.GRID_Y[1]) + 1):
        _, py = to_px(0, y)
        cv2.line(img, (0, py), (img.shape[1], py), (90, 90, 90), 1)
        cv2.putText(img, f"{y}m", (2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(img, to_px(0, 0), to_px(0, bev.GRID_Y[1]), (120, 120, 120), 1)
    for (xl, xr, yf, label) in res.obstacles_ground:
        cv2.rectangle(img, to_px(xl, yf + C.OBST_DEPTH), to_px(xr, yf), (0, 0, 255), 2)
    if res.stop_zone:                                               # STOP zone outline (orange)
        pts = np.array([to_px(x, y) for x, y in res.stop_zone], np.int32)
        cv2.polylines(img, [pts], C.STOP_ZONE != "bubble", (0, 165, 255), 2)
    if len(res.path) >= 2:
        pts = np.array([to_px(x, y) for x, y in res.path], np.int32)
        cv2.polylines(img, [pts], False, (0, 230, 255), 2)
    if res.look:
        cv2.circle(img, to_px(*res.look), 5, (255, 255, 255), -1)
    cv2.circle(img, to_px(0, 0.15), 6, (255, 200, 0), -1)          # the user
    return img
