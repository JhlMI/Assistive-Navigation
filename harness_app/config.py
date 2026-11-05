"""Assisted navigation system parameters (Jetson Orin Nano)."""
from pathlib import Path

BASE = Path.home() / "Assistive-Navigation"

# ---- Models (TensorRT FP16) -------------------------------------------------
ENGINE_SEG = BASE / "modelos" / "segmentacion" / "engines" / "yolo26n_sem_512x896_fp16.engine"
ENGINE_DET = BASE / "modelos" / "deteccion" / "engines" / "yolov8n_384x640_fp16.engine"
SEG_H, SEG_W = 512, 896
DET_H, DET_W = 384, 640
SIDEWALK_ID = 4                    # [background, building, pole, road, sidewalk, vegetation]
SIDEWALK_MIN_AREA = 120            # px at model resolution

# ---- OAK-D Lite camera ------------------------------------------------------
CAMERA_FPS = 30
CAMERA_FOCUS = 60                  # fixed lens position (0..255). None = autofocus within FOCUS_RANGE
FOCUS_RANGE = (50, 70)             # autofocus limits (enforced in software: the camera only takes it as a hint)

# ---- IPM calibration defaults (tape rectangle on the floor) -----------------
CALIB_RECT = (1.0, 2.0, 1.5)       # width, length, near edge distance from the feet (m)

# ---- Decision mode ----------------------------------------------------------
DECISION_MODE = "bev"              # "bev" (grid + path, needs bev_calibration.json) or "centroid"

# ---- BEV planner (meters unless noted) --------------------------------------
GRID_EMA = 0.5                     # temporal smoothing of the walkable grid
BODY_HALF_WIDTH = 0.30             # minimum clearance from edges/obstacles for the path
CLEAR_CAP = 0.80                   # clearance above this counts as "safe enough" (path takes the middle)
POLE_MIN_AREA = 40                 # px (model resolution) for a pole blob to count
POLE_RADIUS = 0.15
PATH_MAX_Y = 4.0                   # how far the path is traced (slow walking: far range adds noise, not info)
PATH_MAX_STEP = 0.10               # max lateral shift of the path per 5 cm row
PATH_GAP_TOL = 0.30                # allowed gap in the path before it ends
START_SEARCH = 0.60                # search for walkable cells this far left/right of the body
PATH_POLY_DEG = 1                  # path center line fit: 1 = straight line (stable heading), 2 = gentle curve
LOOKAHEAD_Y = 2.0                  # look-ahead point (m from the feet)
ANGLE_EMA = 0.4
ANGLE_ENTER = 10.0                 # deg to start correcting
ANGLE_EXIT = 5.0                   # deg to go back to forward
STOP_ZONE = "bubble"               # "bubble": half-ellipse in front of the body | "corridor": band along the path
BUBBLE_DEPTH = 1.0                 # how far ahead of the blind-zone edge it reaches (m). 0.5 = ~0.5 s of warning at 1 m/s
BUBBLE_HALF_WIDTH = 0.45           # half width at its base (m): body (0.30) + 0.15 margin. Wider -> parked cars trigger it
STOP_RANGE = 1.0                   # corridor mode: obstacle within this distance beyond the blind zone -> stop (m)
OBST_MARGIN = 0.05                 # extra lateral margin (each side) beyond the body: corridor = +-(0.30 + this) m
OBST_DEPTH = 0.40                  # assumed depth of a detected obstacle behind its visible bottom edge (m)
MIN_PATH_LEN = 0.8                 # path shorter than this beyond the blind zone -> "no path"

# ---- Direction decision (centroid mode) -------------------------------------
ROI_START = 0.55                   # use the mask from 55 % of the height downwards
SIDEWALK_MIN_FRAC = 0.03           # less than 3 % sidewalk in the ROI -> "no sidewalk"
ENTER_THRESHOLD = 0.25             # |error| to start correcting (0..1)
EXIT_THRESHOLD = 0.15              # |error| to go back to "forward"
EMA_ALPHA = 0.3                    # error smoothing
FRAMES_TO_SWITCH = 3               # consecutive frames needed to change direction
FORWARD_VIBRATES = True            # False: silence while on track

# Obstacles (COCO classes)
OBSTACLE_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CORRIDOR = (0.30, 0.70)            # central band of the image width
OBST_Y2_MIN = 0.80                 # box bottom below 80 % of the height
OBST_HEIGHT_MIN = 0.45             # or box taller than 45 % of the image
FRAMES_OBST_ON = 2                 # consecutive frames to trigger the alert
FRAMES_OBST_OFF = 5                # clear frames to release it
FRAMES_NO_SIDEWALK = 10            # consecutive frames without sidewalk to alert

# ---- Pico -------------------------------------------------------------------
PICO_PORT = None                   # None = auto-detect /dev/ttyACM*
PICO_BAUD = 115200
PICO_PERIOD_S = 0.15               # command resend period (heartbeat)
PICO_TIMEOUT_MS = 1000
PICO_STALE_S = 0.5                 # no new decision from the main loop for this long -> send S

# ---- 4S battery -------------------------------------------------------------
VBAT_WARNING = 14.0                # 3.5 V per cell
VBAT_CRITICAL = 13.2               # 3.3 V per cell

# ---- Logging and web view ---------------------------------------------------
LOG_DIR = BASE / "logs" / "navigation"
WEB_PORT = 8080
WEB_FPS = 8
WEB_WIDTH, WEB_HEIGHT = 640, 360
CLEAN_FPS = 20                     # /clean: raw video for the person wearing the harness (demos)
CLEAN_WIDTH = 1280
REC_CODEC = "mp4v"                 # raw recording: "mp4v" (~7-15 MB/min) or "mjpg" (~40-75 MB/min, max fidelity)
REC_ANNOTATED = False              # True: the Recording button also saves ann_*.avi (overlays, evidence)
REC_ANNOTATED_SIZE = (960, 540)    # camera part of the annotated video (the BEV panel is added beside it)
REC_ANNOTATED_FPS = 15
