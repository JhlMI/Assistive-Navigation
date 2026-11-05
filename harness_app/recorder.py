"""Video recording in background threads (never slows down the main loop: if the disk or the
drawing falls behind, frames are dropped and counted).

Recorder           raw camera frames at 960x540. Keeps enough detail to re-run the models later
                   (they work at 896x504): python main.py --video <file>. Re-timed to a constant
                   FPS from the frame timestamps, so it replays at real speed (no --video-fps).
AnnotatedRecorder  what the system perceived and decided, frame by frame: sidewalk mask, boxes,
                   IPM path, bird's-eye panel, command and a status line. Only frames processed
                   with guidance ON are written. Frames are re-timed to a constant FPS from their
                   timestamps, so the file plays at real speed in any player (evidence videos).
"""
import queue
import threading
import time
from datetime import datetime

import cv2

import config as C


def open_writer(stem, fps, size, codec):
    """codec 'mp4v' (MPEG-4, ~5x smaller, less CPU) or 'mjpg' (one JPEG per frame).
    Falls back to MJPG if this OpenCV build cannot write MPEG-4. Returns (writer, path)."""
    if codec == "mp4v":
        path = C.LOG_DIR / f"{stem}.mp4"
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if w.isOpened():
            return w, path
        print("MPEG-4 writer not available: recording as MJPG")
    path = C.LOG_DIR / f"{stem}.avi"
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size), path


class Recorder:
    def __init__(self, fps=15, size=(960, 540)):
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.fps = fps
        self._writer, self.path = open_writer(f"rec_{datetime.now():%Y%m%d_%H%M%S}", fps, size, C.REC_CODEC)
        self._queue = queue.Queue(maxsize=8)
        self.dropped = 0
        self.frames = 0
        self.written = 0
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def add(self, frame, t=None):
        try:
            self._queue.put_nowait((time.monotonic() if t is None else t, frame))
            self.frames += 1
        except queue.Full:
            self.dropped += 1                 # never slow down the main loop

    def _loop(self):
        period = 1.0 / self.fps
        next_t = None
        while True:
            item = self._queue.get()
            if item is None:
                break
            t, frame = item
            small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            if next_t is None or t - next_t > 1.0:
                next_t = t                    # first frame, or the camera stalled: skip the gap
            while t >= next_t:                # constant FPS: repeat or skip frames by their timestamp
                self._writer.write(small)
                self.written += 1
                next_t += period

    def real_fps(self):
        return self.frames / max(1e-6, time.monotonic() - self._t0)

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._writer.release()


class AnnotatedRecorder:
    """add(t, frame, walkable, boxes, result, info): the drawing happens in this thread."""

    def __init__(self, fps=15, size=(960, 540), stamp=None):
        C.LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = stamp or f"{datetime.now():%Y%m%d_%H%M%S}"
        self.path = C.LOG_DIR / f"ann_{stamp}.avi"      # same stamp as the raw rec_*.avi
        self.fps = fps
        self.size = size                                 # camera part; the BEV panel is added to the right
        self._writer = None                              # opened on the first frame (its width depends on the mode)
        self._queue = queue.Queue(maxsize=4)
        self.dropped = 0
        self.frames = 0
        self.written = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def add(self, t, frame, walkable, boxes, result, info):
        try:
            self._queue.put_nowait((t, frame, walkable, boxes, result, info))
            self.frames += 1
        except queue.Full:
            self.dropped += 1

    def _status_bar(self, img, info):
        h, w = img.shape[0], self.size[0]          # over the camera part only (keeps the BEV panel clear)
        bar = img.copy()
        cv2.rectangle(bar, (0, h - 28), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.75, img, 0.25, 0, img)
        cv2.putText(img, info, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return img

    def _loop(self):
        from web import annotate
        period = 1.0 / self.fps
        next_t = None
        while True:
            item = self._queue.get()
            if item is None:
                break
            t, frame, walkable, boxes, res, info = item
            img = self._status_bar(annotate(frame, walkable, boxes, res, self.size), info)
            if self._writer is None:
                h, w = img.shape[:2]
                self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"),
                                               self.fps, (w, h))
            if next_t is None or t - next_t > 1.0:
                next_t = t                               # first frame, or guidance was off: skip the gap
            # Constant FPS: repeat the frame for every slot it covers (or skip it if it is early)
            while t >= next_t:
                self._writer.write(img)
                self.written += 1
                next_t += period

    def close(self):
        self._queue.put(None)
        self._thread.join(timeout=10)
        if self._writer is not None:
            self._writer.release()
