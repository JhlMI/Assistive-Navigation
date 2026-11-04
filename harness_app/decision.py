"""Direction decision (version without IPM).

1. Close obstacle in the central corridor (person, vehicle...) -> 'X' (stop).
2. No walkable sidewalk for several frames                     -> 'X'.
3. Error = horizontal position of the sidewalk center relative to the image center:
   error < 0 (sidewalk to the left) -> 'L', error > 0 -> 'R', centered -> 'F'.

Assumes the camera is centered on the chest, facing forward: the image center is
the direction the person is walking.
"""
from dataclasses import dataclass, field

import numpy as np

import config as C


@dataclass
class Result:
    command: str = "S"           # F, L, R, X or S
    reason: str = "start"
    error: float = 0.0           # smoothed error, -1 (left) .. 1 (right)
    raw_error: float = 0.0
    sidewalk_frac: float = 0.0
    center_x: float = None       # sidewalk center in the ROI (px at model resolution)
    obstacles: list = field(default_factory=list)   # boxes that trigger the alert


class Decider:
    def __init__(self):
        self.error_ema = 0.0
        self.direction = "F"          # stable direction (F, L or R)
        self.candidate = "F"
        self.candidate_count = 0
        self.obst_frames = 0
        self.clear_frames = 0
        self.obst_alert = False
        self.no_sidewalk_frames = 0

    # ---- Obstacles ----------------------------------------------------------
    def _close_obstacles(self, boxes, height, width):
        close = []
        for (x1, y1, x2, y2, score, cls) in boxes:
            if cls not in C.OBSTACLE_CLASSES:
                continue
            cx = (x1 + x2) / 2 / width
            in_corridor = C.CORRIDOR[0] <= cx <= C.CORRIDOR[1]
            near = (y2 / height >= C.OBST_Y2_MIN) or ((y2 - y1) / height >= C.OBST_HEIGHT_MIN)
            if in_corridor and near:
                close.append((x1, y1, x2, y2, score, cls))
        return close

    # ---- Sidewalk -----------------------------------------------------------
    @staticmethod
    def _measure_sidewalk(walkable):
        h, w = walkable.shape
        roi = walkable[int(h * C.ROI_START):, :]
        frac = float(roi.mean()) if roi.size else 0.0
        if frac < C.SIDEWALK_MIN_FRAC:
            return frac, None, 0.0
        columns = roi.sum(axis=0).astype(np.float64)
        center_x = float((columns * np.arange(w)).sum() / columns.sum())
        error = (center_x - w / 2) / (w / 2)
        return frac, center_x, float(np.clip(error, -1.0, 1.0))

    # ---- Direction with hysteresis and debounce -----------------------------
    def _direction(self, error):
        current = self.direction
        if current == "F":
            proposal = "L" if error < -C.ENTER_THRESHOLD else "R" if error > C.ENTER_THRESHOLD else "F"
        elif current == "L":
            proposal = "L" if error < -C.EXIT_THRESHOLD else ("R" if error > C.ENTER_THRESHOLD else "F")
        else:
            proposal = "R" if error > C.EXIT_THRESHOLD else ("L" if error < -C.ENTER_THRESHOLD else "F")

        if proposal == current:
            self.candidate_count = 0
            return current
        if proposal == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = proposal, 1
        if self.candidate_count >= C.FRAMES_TO_SWITCH:
            self.direction = proposal
            self.candidate_count = 0
        return self.direction

    # ---- Main step ----------------------------------------------------------
    def update(self, walkable, boxes, img_shape):
        height, width = img_shape[:2]
        r = Result()

        # Obstacles: trigger fast, release slowly
        r.obstacles = self._close_obstacles(boxes, height, width)
        if r.obstacles:
            self.obst_frames += 1
            self.clear_frames = 0
        else:
            self.clear_frames += 1
            self.obst_frames = 0
        if self.obst_frames >= C.FRAMES_OBST_ON:
            self.obst_alert = True
        elif self.clear_frames >= C.FRAMES_OBST_OFF:
            self.obst_alert = False

        # Sidewalk
        r.sidewalk_frac, r.center_x, r.raw_error = self._measure_sidewalk(walkable)
        if r.center_x is None:
            self.no_sidewalk_frames += 1
        else:
            self.no_sidewalk_frames = 0
            self.error_ema = C.EMA_ALPHA * r.raw_error + (1 - C.EMA_ALPHA) * self.error_ema
        r.error = self.error_ema

        # Priorities
        if self.obst_alert:
            r.command, r.reason = "X", "obstacle"
        elif self.no_sidewalk_frames >= C.FRAMES_NO_SIDEWALK:
            r.command, r.reason = "X", "no sidewalk"
        elif r.center_x is None:
            r.command, r.reason = self.direction, "no sidewalk (brief)"   # keep the last one
        else:
            r.command = self._direction(self.error_ema)
            r.reason = {"F": "centered", "L": "sidewalk to the left", "R": "sidewalk to the right"}[r.command]

        if r.command == "F" and not C.FORWARD_VIBRATES:
            r.command = "S"
        return r
