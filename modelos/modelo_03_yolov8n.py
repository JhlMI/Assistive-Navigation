#!/usr/bin/env python3
"""
modelo_03_yolov8n.py  —  YOLOv8n  (referencia / fallback)
==========================================================
Bloque A — Detección de obstáculos | Taller de Grado I

Entradas:  --fuente  imagen.jpg | carpeta/ | video.mp4 | 0 (webcam)
Salidas:   output/yolov8n/<nombre>_yolov8n.mp4
           output/yolov8n/<nombre>_raw.csv
           output/yolov8n/<nombre>_resumen.csv

Controles: Q/ESC salir | ESPACIO pausa | S captura
"""

import sys, time, argparse, csv, platform
from pathlib import Path

import cv2
import numpy as np

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

# ── Configuración ─────────────────────────────────────────────────────────────

PESO      = "yolov8n.pt"
CONFIANZA = 0.45
IOU       = 0.45
IMGSZ     = 640
CLASES    = None

OUTPUT_DIR = Path("output/yolov8n")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORES = {
    0:  (0,   60,  255),
    1:  (0,   165, 255),
    2:  (255, 120, 0  ),
    3:  (0,   200, 0  ),
    5:  (200, 0,   180),
    7:  (0,   200, 255),
    9:  (0,   230, 230),
    11: (180, 0,   255),
    56: (255, 80,  80 ),
    57: (80,  255, 80 ),
    60: (80,  80,  255),
    39: (0,   180, 180),
    24: (80,  200, 120),
}
COLOR_DEFAULT = (180, 180, 180)


# ── Monitor de recursos ───────────────────────────────────────────────────────

class Monitor:
    def __init__(self):
        if PSUTIL_OK:
            psutil.cpu_percent()
        if NVML_OK:
            self.gpu_handle   = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem               = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            self.gpu_nombre   = pynvml.nvmlDeviceGetName(self.gpu_handle)
            self.gpu_total_mb = mem.total / 1e6
        else:
            self.gpu_handle   = None
            self.gpu_nombre   = "GPU (pynvml no disp.)"
            self.gpu_total_mb = 0

    def leer(self) -> dict:
        cpu    = psutil.cpu_percent() if PSUTIL_OK else 0
        ram_gb = psutil.virtual_memory().used / 1e9 if PSUTIL_OK else 0
        if NVML_OK and self.gpu_handle:
            mem     = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            util    = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            vram_mb = mem.used / 1e6
            gpu_pct = util.gpu
        else:
            vram_mb, gpu_pct = 0, 0
        return {"ram_gb": round(ram_gb,2), "cpu_pct": cpu,
                "gpu_mb": round(vram_mb,1), "gpu_pct": gpu_pct}

    def info_sistema(self) -> dict:
        vm = psutil.virtual_memory() if PSUTIL_OK else None
        return {
            "os":           platform.system() + " " + platform.release(),
            "python":       platform.python_version(),
            "cpu":          platform.processor() or platform.machine(),
            "ram_total_gb": round(vm.total/1e9,1) if vm else 0,
            "gpu":          self.gpu_nombre,
            "gpu_vram_gb":  round(self.gpu_total_mb/1e3,1),
            "cuda":         torch.version.cuda if TORCH_OK else "N/A",
            "pytorch":      torch.__version__  if TORCH_OK else "N/A",
        }


# ── Overlay ───────────────────────────────────────────────────────────────────

def _txt(img, texto, x, y, esc=0.62, col=(255,255,255), fondo=(15,15,15)):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw,th),bl = cv2.getTextSize(texto, f, esc, 1)
    p = 4
    cv2.rectangle(img, (x-p, y-th-p), (x+tw+p, y+bl+p), fondo, -1)
    cv2.putText(img, texto, (x,y), f, esc, col, 1, cv2.LINE_AA)


def dibujar(frame, resultado, lat_ms, recursos, frame_id, monitor):
    vis = frame.copy()
    h, w = vis.shape[:2]
    cajas = resultado.boxes
    n_det = 0

    if cajas is not None:
        n_det = len(cajas)
        for caja in cajas:
            x1,y1,x2,y2 = map(int, caja.xyxy[0])
            cls    = int(caja.cls[0])
            conf   = float(caja.conf[0])
            nombre = resultado.names[cls]
            color  = COLORES.get(cls, COLOR_DEFAULT)

            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            etiq = f"{nombre} {conf:.2f}"
            (tw,th),_ = cv2.getTextSize(etiq, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)
            cv2.rectangle(vis, (x1,y1-th-8), (x1+tw+6,y1), color, -1)
            cv2.putText(vis, etiq, (x1+3,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 1, cv2.LINE_AA)

    fps     = 1000/lat_ms if lat_ms > 0 else 0
    lat_col = (100,255,100) if lat_ms < 300 else (0,60,255)
    fps_col = (100,255,100) if fps >= 15    else (0,60,255)

    PASO, M = 24, 8
    izq = [
        ("YOLOv8n  (referencia)",       (0,220,255)),
        (f"Frame    {frame_id:>5}",     (210,210,210)),
        (f"FPS      {fps:>6.1f}",       fps_col),
        (f"Lat      {lat_ms:>5.1f} ms", lat_col),
        (f"Detec.   {n_det:>5}",        (210,210,210)),
    ]
    for i,(t,c) in enumerate(izq):
        _txt(vis, t, M, M+18+i*PASO, col=c)

    der = [
        ("SISTEMA",                                        (0,220,255)),
        (f"RAM sys {recursos['ram_gb']:>4.1f} GB",        (210,210,210)),
        (f"CPU     {recursos['cpu_pct']:>5.1f} %",        (210,210,210)),
        (f"VRAM    {recursos['gpu_mb']:>6.0f} MB",        (190,140,255)),
        (f"GPU     {recursos['gpu_pct']:>5.1f} %",        (190,140,255)),
    ]
    xd = w - 220
    for i,(t,c) in enumerate(der):
        _txt(vis, t, xd, M+18+i*PASO, col=c)

    _txt(vis, f"{monitor.gpu_nombre}  |  {w}x{h}  |  conf={CONFIANZA}",
         M, h-M-6, esc=0.50, col=(160,160,160))

    return vis


# ── Inferencia ────────────────────────────────────────────────────────────────

def inferir(modelo, frame):
    t0  = time.perf_counter()
    res = modelo(frame, conf=CONFIANZA, iou=IOU, imgsz=IMGSZ,
                 classes=CLASES, verbose=False)
    return res[0], (time.perf_counter()-t0)*1000


# ── CSV ───────────────────────────────────────────────────────────────────────

def guardar_raw(registros, ruta):
    if not registros: return
    with open(ruta,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
        w.writeheader(); w.writerows(registros)
    print(f"  CSV raw     : {ruta}")


def guardar_resumen(registros, ruta, info_sys, fuente_nombre):
    if not registros: return

    def stats(campo):
        v = np.array([r[campo] for r in registros], dtype=float)
        return [round(float(np.mean(v)),2), round(float(np.std(v)),2),
                round(float(np.min(v)),2),  round(float(np.max(v)),2),
                round(float(np.percentile(v,50)),2),
                round(float(np.percentile(v,95)),2)]

    lats  = np.array([r["latencia_ms"]     for r in registros])
    fps_v = np.array([r["fps_instantaneo"] for r in registros])
    lat_m, fps_m, lat_s = np.mean(lats), np.mean(fps_v), np.std(lats)

    with open(ruta,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# RESUMEN YOLOv8n"])
        w.writerow(["# Fuente", fuente_nombre])
        w.writerow(["# Frames", len(registros)])
        w.writerow(["# Fecha",  time.strftime("%Y-%m-%d %H:%M:%S")])
        for k,v in info_sys.items():
            w.writerow([f"# {k}", v])
        w.writerow([])
        w.writerow(["# CRITERIOS DE VALIDACIÓN"])
        w.writerow(["# Lat <300ms", "CUMPLE" if lat_m<300 else "NO CUMPLE", f"{lat_m:.1f}ms"])
        w.writerow(["# FPS >=15",   "CUMPLE" if fps_m>=15 else "NO CUMPLE", f"{fps_m:.1f}"])
        w.writerow(["# Estab <20%", "CUMPLE" if lat_s<lat_m*.20 else "NO CUMPLE",
                    f"std={lat_s:.1f}ms ({lat_s/lat_m*100:.1f}%)"])
        w.writerow([])
        w.writerow(["metrica","media","std","min","max","p50","p95"])
        for m in ["latencia_ms","fps_instantaneo","ram_gb","gpu_mb","cpu_pct","obstaculos_detectados"]:
            if m in registros[0]:
                w.writerow([m]+stats(m))
    print(f"  CSV resumen : {ruta}")


def imprimir_resumen(registros, nombre):
    if not registros: return
    lats  = np.array([r["latencia_ms"]          for r in registros])
    fps_v = np.array([r["fps_instantaneo"]       for r in registros])
    rams  = np.array([r["ram_gb"]                for r in registros])
    dets  = np.array([r["obstaculos_detectados"] for r in registros])
    lat_m, fps_m, lat_s = float(np.mean(lats)), float(np.mean(fps_v)), float(np.std(lats))
    print(f"\n  {'─'*50}")
    print(f"  RESUMEN — YOLOv8n  |  {nombre}")
    print(f"  {'─'*50}")
    print(f"  Frames          : {len(registros)}")
    print(f"  FPS  media      : {fps_m:.1f}   (min {np.min(fps_v):.1f} / max {np.max(fps_v):.1f})")
    print(f"  Lat. media      : {lat_m:.1f} ms  ±{lat_s:.1f}")
    print(f"  Lat. p95        : {float(np.percentile(lats,95)):.1f} ms")
    print(f"  RAM sys pico    : {float(np.max(rams)):.2f} GB")
    print(f"  Detec. media    : {float(np.mean(dets)):.1f} por frame")
    print(f"  {'─'*50}")
    print(f"  Lat <300ms      : {'✅ CUMPLE' if lat_m<300 else '❌ NO CUMPLE'}")
    print(f"  FPS >=15        : {'✅ CUMPLE' if fps_m>=15 else '❌ NO CUMPLE'}")
    ok_std = lat_s < lat_m*.20
    print(f"  Estab. <20%std  : {'✅ CUMPLE' if ok_std else '❌ NO CUMPLE'}  "
          f"({lat_s/lat_m*100:.1f}%)")
    print()


# ── Modos de entrada ──────────────────────────────────────────────────────────

def procesar_imagen(modelo, ruta, monitor):
    frame = cv2.imread(str(ruta))
    if frame is None:
        print(f"  [!] No se pudo leer: {ruta}"); return
    rec = monitor.leer()
    res, lat_ms = inferir(modelo, frame)
    n_det = len(res.boxes) if res.boxes is not None else 0
    print(f"  {ruta.name:40s}  lat={lat_ms:6.1f}ms  det={n_det}")
    vis = dibujar(frame, res, lat_ms, rec, 1, monitor)
    cv2.imwrite(str(OUTPUT_DIR/ruta.name), vis)
    cv2.imshow("YOLOv8n", vis); cv2.waitKey(0); cv2.destroyAllWindows()
    reg = [{"frame_id":1,"latencia_ms":round(lat_ms,2),
             "fps_instantaneo":round(1000/lat_ms,2),**rec,
             "obstaculos_detectados":n_det}]
    guardar_raw(reg, OUTPUT_DIR/f"{ruta.stem}_raw.csv")
    guardar_resumen(reg, OUTPUT_DIR/f"{ruta.stem}_resumen.csv",
                    monitor.info_sistema(), ruta.name)


def procesar_carpeta(modelo, ruta, monitor):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    imgs = sorted([p for p in ruta.iterdir() if p.suffix.lower() in exts])
    if not imgs:
        print(f"  [!] Sin imágenes en {ruta}"); return
    print(f"  {len(imgs)} imágenes."); regs=[]
    for i,ip in enumerate(imgs,1):
        frame=cv2.imread(str(ip))
        if frame is None: continue
        rec=monitor.leer(); res,lat_ms=inferir(modelo,frame)
        n_det=len(res.boxes) if res.boxes is not None else 0
        print(f"  {ip.name:40s}  lat={lat_ms:6.1f}ms  det={n_det}")
        regs.append({"frame_id":i,"latencia_ms":round(lat_ms,2),
                     "fps_instantaneo":round(1000/lat_ms,2),**rec,
                     "obstaculos_detectados":n_det})
        vis=dibujar(frame,res,lat_ms,rec,i,monitor)
        cv2.imwrite(str(OUTPUT_DIR/ip.name),vis)
        cv2.imshow("YOLOv8n",vis)
        if cv2.waitKey(1)&0xFF in (ord("q"),27): break
    cv2.destroyAllWindows()
    guardar_raw(regs, OUTPUT_DIR/f"{ruta.name}_raw.csv")
    guardar_resumen(regs, OUTPUT_DIR/f"{ruta.name}_resumen.csv",
                    monitor.info_sistema(), str(ruta))
    imprimir_resumen(regs, ruta.name)


def procesar_video(modelo, fuente, monitor):
    cap=cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"  [!] No se pudo abrir: {fuente}"); return
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src=cap.get(cv2.CAP_PROP_FPS) or 30
    es_wc=isinstance(fuente,int)
    nom=("webcam" if es_wc else Path(str(fuente)).stem)
    writer=cv2.VideoWriter(str(OUTPUT_DIR/f"{nom}_yolov8n.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"),fps_src,(W,H))
    print(f"  {W}×{H} @ {fps_src:.0f}fps  —  Q/ESC salir  ESPACIO pausa  S captura\n")
    regs=[]; n=0; pausado=False; scap=0; vis=None
    while True:
        if not pausado:
            ret,frame=cap.read()
            if not ret: break
            n+=1
            rec=monitor.leer(); res,lat_ms=inferir(modelo,frame)
            n_det=len(res.boxes) if res.boxes is not None else 0
            regs.append({"frame_id":n,"latencia_ms":round(lat_ms,2),
                          "fps_instantaneo":round(1000/lat_ms,2),
                          "ram_gb":round(rec["ram_gb"],2),
                          "gpu_mb":round(rec["gpu_mb"],1),
                          "cpu_pct":round(rec["cpu_pct"],1),
                          "obstaculos_detectados":n_det})
            vis=dibujar(frame,res,lat_ms,rec,n,monitor)
            writer.write(vis)
            if n%30==0:
                fps_r=float(np.mean([r["fps_instantaneo"] for r in regs[-30:]]))
                print(f"  Frame {n:>5}  lat={lat_ms:5.1f}ms  FPS={fps_r:.1f}"
                      f"  det={n_det}  RAM={rec['ram_gb']:.1f}GB"
                      f"  VRAM={rec['gpu_mb']:.0f}MB  GPU={rec['gpu_pct']:.0f}%")
            cv2.imshow("YOLOv8n",vis)
        key=cv2.waitKey(1)&0xFF
        if key in (ord("q"),27): break
        elif key==ord(" "):
            pausado=not pausado
            print(f"  {'⏸ Pausado' if pausado else '▶ Reanudado'}")
        elif key==ord("s") and vis is not None:
            scap+=1; p=OUTPUT_DIR/f"{nom}_captura_{scap:03d}.jpg"
            cv2.imwrite(str(p),vis); print(f"  📸 {p}")
    cap.release(); writer.release(); cv2.destroyAllWindows()
    guardar_raw(regs, OUTPUT_DIR/f"{nom}_raw.csv")
    guardar_resumen(regs, OUTPUT_DIR/f"{nom}_resumen.csv",
                    monitor.info_sistema(), str(fuente))
    imprimir_resumen(regs, nom)
    print(f"  Video : {OUTPUT_DIR/f'{nom}_yolov8n.mp4'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global CONFIANZA, IOU, IMGSZ
    parser=argparse.ArgumentParser(description="YOLOv8n — referencia")
    parser.add_argument("--fuente", required=True)
    parser.add_argument("--conf",  type=float, default=CONFIANZA)
    parser.add_argument("--iou",   type=float, default=IOU)
    parser.add_argument("--imgsz", type=int,   default=IMGSZ)
    args=parser.parse_args()
    CONFIANZA,IOU,IMGSZ=args.conf,args.iou,args.imgsz

    print(f"\n{'─'*52}")
    print(f"  YOLOv8n  —  Bloque A  |  Taller de Grado I")
    print(f"  conf={CONFIANZA}  iou={IOU}  imgsz={IMGSZ}")
    print(f"{'─'*52}")

    monitor=Monitor()
    info=monitor.info_sistema()
    print(f"  GPU    : {info['gpu']}  ({info['gpu_vram_gb']} GB VRAM)")
    print(f"  RAM    : {info['ram_total_gb']} GB total")
    print(f"  CUDA   : {info['cuda']}\n")

    from ultralytics import YOLO
    print(f"  Cargando {PESO}...")
    t0=time.perf_counter()
    modelo=YOLO(PESO)
    print(f"  Listo en {(time.perf_counter()-t0)*1000:.0f} ms\n")

    fuente=args.fuente
    try:
        fuente=int(fuente); procesar_video(modelo,fuente,monitor); return
    except ValueError: pass

    ruta=Path(fuente)
    if not ruta.exists():
        print(f"  [!] No existe: {ruta}"); sys.exit(1)
    if ruta.is_dir():
        procesar_carpeta(modelo,ruta,monitor)
    elif ruta.suffix.lower() in {".mp4",".avi",".mov",".mkv",".webm"}:
        procesar_video(modelo,str(ruta),monitor)
    else:
        procesar_imagen(modelo,ruta,monitor)

if __name__=="__main__":
    main()
