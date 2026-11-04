"""IPM (bird's-eye view) calibration, shared by the main web view (/calib) and calibrate_bev.py.

Floor setup: a tape rectangle centered on the body, WIDTH x LENGTH meters,
its near edge NEAR meters in front of the feet. Click its 4 corners on a frame:
near-left, near-right, far-right, far-left.
"""
import json
import threading
from datetime import datetime

import cv2

import bev
import config as C

DISPLAY_SCALE = 0.5          # the page shows frames at 960x540; clicks are scaled back


def page(base="", grab_label="Take photo"):
    """Calibration page. base = URL prefix of the calibration routes ('' or '/calib')."""
    w, l, n = C.CALIB_RECT
    return """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IPM calibration</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:monospace;padding:12px}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
canvas{border:1px solid #444;cursor:crosshair;max-width:100%}
label{display:inline-block;margin:4px 12px 4px 0}
input{width:70px;background:#222;color:#eee;border:1px solid #555;padding:3px}
button{background:#2a5;color:#fff;border:0;padding:8px 14px;margin:4px 6px 4px 0;cursor:pointer;font:14px monospace}
button.alt{background:#555}
#log{white-space:pre-wrap;margin:8px 0;color:#9cf}
</style></head><body>
<div>
 <label>width (m) <input id="w" value='""" + str(w) + """'></label>
 <label>length (m) <input id="l" value='""" + str(l) + """'></label>
 <label>near edge (m) <input id="n" value='""" + str(n) + """'></label>
</div>
<div>
 <button class="alt" onclick="grab()">""" + grab_label + """</button>
 <button class="alt" onclick="pts=[];draw()">Clear points</button>
 <button onclick="calib()">Calibrate</button>
</div>
<div id="log">Click: near-left, near-right, far-right, far-left</div>
<div class="row">
 <canvas id="c" width="960" height="540"></canvas>
 <img id="bev" style="max-width:100%">
</div>
<script>
const B='""" + base + """';
const c=document.getElementById('c'),g=c.getContext('2d'),log=document.getElementById('log');
let img=new Image(),pts=[],calibrated=false;
const names=['near-left','near-right','far-right','far-left'];
function load(){img=new Image();img.onload=async()=>{draw();try{const p=(await (await fetch(B+'/photo')).json()).photo;
 if(p)log.textContent='Photo: '+p+'\\nClick: near-left, near-right, far-right, far-left';}catch(e){}};img.src=B+'/frame.jpg?'+Date.now();}
function draw(){g.clearRect(0,0,960,540);g.drawImage(img,0,0,960,540);g.fillStyle='#ff0';g.strokeStyle='#ff0';g.font='14px monospace';
 pts.forEach((p,i)=>{g.beginPath();g.arc(p[0],p[1],5,0,7);g.fill();g.fillText(names[i],p[0]+8,p[1]-8);});
 if(pts.length==4){g.beginPath();g.moveTo(...pts[0]);pts.slice(1).forEach(p=>g.lineTo(...p));g.closePath();g.stroke();}}
c.onclick=async e=>{const r=c.getBoundingClientRect(),x=(e.clientX-r.left)*960/r.width,y=(e.clientY-r.top)*540/r.height;
 if(pts.length<4){pts.push([x,y]);draw();
   log.textContent=pts.length<4?'Next: '+names[pts.length]:'4 points set. Check the sizes and press Calibrate.';}
 else if(calibrated){const q=await (await fetch(B+`/project?u=${x}&v=${y}`)).json();
   log.textContent=`Ground point: x = ${q.x.toFixed(2)} m (right +), y = ${q.y.toFixed(2)} m ahead`;
   draw();g.fillStyle='#0ff';g.beginPath();g.arc(x,y,5,0,7);g.fill();}};
async function grab(){const r=await (await fetch(B+'/grab',{method:'POST'})).json();pts=[];calibrated=false;load();
 log.textContent=(r.saved?'Photo: '+r.saved+'\\n':'')+'Click: near-left';}
async function calib(){if(pts.length!=4){log.textContent='Need 4 points';return;}
 const body={points:pts,width:+w.value,length:+l.value,near:+n.value};
 const r=await (await fetch(B+'/compute',{method:'POST',body:JSON.stringify(body)})).json();
 if(!r.ok){log.textContent='Error: '+r.error;return;}
 calibrated=true;document.getElementById('bev').src=B+'/bev.jpg?'+Date.now();
 log.textContent='Saved to '+r.file+'\\nCheck: '+r.info+'\\nNow click any floor point to see its distance.';}
load();
</script></body></html>"""


def save_snapshot(frame, prefix="snap"):
    """Save a full-resolution frame for later calibration. Returns the file path."""
    folder = C.LOG_DIR / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3]      # millisecond resolution
    path = folder / f"{name}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


class CalibSession:
    """frame_provider(): returns the current full-resolution BGR frame (or None)."""

    def __init__(self, frame_provider, save_photos=True, name_provider=None):
        self.frame_provider = frame_provider
        self.save_photos = save_photos
        self.name_provider = name_provider    # name of the current photo (image mode)
        self.photo = None
        self.lock = threading.Lock()
        self.frame = None
        self.H = None
        self.bev_jpg = None

    def grab(self):
        frame = self.frame_provider()
        if frame is None:
            return None
        with self.lock:
            self.frame = frame
        if self.save_photos:
            self.photo = str(save_snapshot(frame, "calib"))
        elif self.name_provider:
            self.photo = self.name_provider()
        return self.photo

    def frame_jpg(self):
        with self.lock:
            frame = self.frame
        if frame is None:
            frame = self.frame_provider()
            if frame is None:
                return None
            with self.lock:
                self.frame = frame
            if self.name_provider:
                self.photo = self.name_provider()
        small = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def compute(self, points, width, length, near):
        full = [[x / DISPLAY_SCALE, y / DISPLAY_SCALE] for x, y in points]
        H = bev.homography_from_points(full, width, length, near)
        self.H = H
        bev.save_calibration(H, {"points_px": full, "rect_m": [width, length, near],
                                 "focus": C.CAMERA_FOCUS, "photo": self.photo})
        bev.append_run({"photo": self.photo, "points_px": full, "rect_m": [width, length, near],
                        "time": datetime.now().isoformat(timespec="seconds")})
        with self.lock:
            top = bev.draw_grid(bev.warp_to_grid(self.frame, H, res=0.02), res=0.02)
        self.bev_jpg = cv2.imencode(".jpg", top)[1].tobytes()
        _, y_bottom = bev.image_to_ground(H, 960, 1079)
        return f"bottom edge of the image is {y_bottom:.2f} m ahead of the feet (blind zone)"

    def project(self, u, v):
        return bev.image_to_ground(self.H, u / DISPLAY_SCALE, v / DISPLAY_SCALE)

    # ---- HTTP helpers (sub-path after the prefix) ---------------------------
    def handle_get(self, sub, query):
        """Returns (body, content_type) or None."""
        if sub == "/frame.jpg":
            jpg = self.frame_jpg()
            return (jpg, "image/jpeg") if jpg else None
        if sub == "/photo":
            return json.dumps({"photo": self.photo}).encode(), "application/json"
        if sub == "/bev.jpg" and self.bev_jpg:
            return self.bev_jpg, "image/jpeg"
        if sub == "/project" and self.H is not None:
            x, y = self.project(float(query["u"][0]), float(query["v"][0]))
            return json.dumps({"x": x, "y": y}).encode(), "application/json"
        return None

    def handle_post(self, sub, body):
        if sub == "/grab":
            saved = self.grab()
            return json.dumps({"ok": True, "saved": saved}).encode(), "application/json"
        if sub == "/compute":
            try:
                d = json.loads(body)
                info = self.compute(d["points"], d["width"], d["length"], d["near"])
                out = {"ok": True, "file": str(bev.CALIB_FILE), "info": info}
            except Exception as e:
                out = {"ok": False, "error": str(e)}
            return json.dumps(out).encode(), "application/json"
        return None
