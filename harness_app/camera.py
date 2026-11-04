"""Image sources: live OAK-D Lite or a recorded video.

Both run in their own thread and keep only the latest frame, so perception always
processes the newest image and latency never builds up.
"""
import threading
import time

import cv2


class _SourceBase:
    def __init__(self):
        self._frame = None
        self._id = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _publish(self, frame):
        with self._cond:
            self._frame = frame
            self._id += 1
            self._cond.notify_all()

    def next(self, last_id, timeout=1.0):
        """Wait for a frame newer than last_id. Returns (frame, id) or (None, last_id)."""
        with self._cond:
            self._cond.wait_for(lambda: self._id != last_id or not self._running, timeout)
            if self._id == last_id:
                return None, last_id
            return self._frame, self._id

    @property
    def active(self):
        return self._running

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()


class OakCamera(_SourceBase):
    """OAK-D Lite at 1080p with the same ISP settings used for the dataset."""

    def __init__(self, fps=30, focus=None, focus_range=(50, 70)):
        super().__init__()
        import depthai as dai

        self._dai = dai
        self._focus = focus
        self._focus_range = focus_range
        self.lens_position = None
        self._t_clamp = 0.0
        self._pipeline = dai.Pipeline()
        cam = self._pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam.setIspScale(1, 2)
        cam.setVideoSize(1920, 1080)
        cam.setFps(fps)
        cam.initialControl.setSharpness(0)
        cam.initialControl.setLumaDenoise(0)
        cam.initialControl.setChromaDenoise(4)
        if focus is not None:
            cam.initialControl.setManualFocus(int(focus))        # fixed focus, same as the GUI default
        else:
            cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            cam.initialControl.setAutoFocusLensRange(*focus_range)
        self._queue = cam.video.createOutputQueue(maxSize=2, blocking=False)
        self._control = cam.inputControl.createInputQueue()
        self._pipeline.start()
        try:
            self.usb_speed = str(self._pipeline.getDefaultDevice().getUsbSpeed()).split(".")[-1]
        except Exception:
            self.usb_speed = "unknown"
        if self.usb_speed not in ("SUPER", "SUPER_PLUS"):
            print(f"WARNING: camera on USB {self.usb_speed} (not SUPER): 1080p will drop below 30 FPS. "
                  "Flip the USB-C plug or use another port/cable.")
        self._thread.start()

    def _loop(self):
        while self._running and self._pipeline.isRunning():
            msg = self._queue.tryGet()
            if msg is None:
                time.sleep(0.002)
                continue
            self.lens_position = msg.getLensPosition()
            self._clamp_focus()
            self._publish(msg.getCvFrame())
        self._running = False

    def _clamp_focus(self):
        """In autofocus mode, if the lens leaves focus_range (the camera may ignore the hint),
        fix it at the nearest limit. From then on the focus stays fixed at that value."""
        if self._focus is not None or self.lens_position is None:
            return
        lo, hi = self._focus_range
        if not lo <= self.lens_position <= hi and time.monotonic() - self._t_clamp > 0.5:
            self._t_clamp = time.monotonic()
            ctrl = self._dai.CameraControl()
            ctrl.setManualFocus(min(max(self.lens_position, lo), hi))
            self._control.send(ctrl)
            self._focus = min(max(self.lens_position, lo), hi)

    def stop(self):
        super().stop()
        self._thread.join(timeout=2.0)       # let the reader leave tryGet() before stopping
        try:
            self._pipeline.stop()
        except Exception:
            pass


class VideoFile(_SourceBase):
    """Plays a video at its original FPS (to test without the camera)."""

    def __init__(self, path, fps=None, loop=False):
        super().__init__()
        self.loop = loop
        self.loops = 0                # times the video restarted (the main loop resets its filters)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0   # fps overrides the file header
        self._period = 1.0 / fps
        self._thread.start()

    def _loop(self):
        next_t = time.perf_counter()
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if self.loop and self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    self.loops += 1
                    ok, frame = self._cap.read()
                if not ok:
                    break
            if frame.shape[1] != 1920:
                frame = cv2.resize(frame, (1920, 1080))
            self._publish(frame)
            next_t += self._period
            wait = next_t - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.perf_counter()
        self._cap.release()
        self.stop()
