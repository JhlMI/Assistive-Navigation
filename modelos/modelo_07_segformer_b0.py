#!/usr/bin/env python3
"""
modelo_04_ddrnet.py  —  SegFormer-B0  (Cityscapes real via SegFormer-B3)
==========================================================================
Bloque B — Segmentación de acera | Taller de Grado I

Entradas:  --fuente  imagen.jpg | carpeta/ | video.mp4 | 0 (webcam)
Salidas:   output/segformer_b0/<nombre>_segformer_b0.mp4
           output/segformer_b0/<nombre>_raw.csv
           output/segformer_b0/<nombre>_resumen.csv

Controles: Q/ESC salir | ESPACIO pausa | S captura
           --no-display  para correr sin ventana (Jetson headless)
Req:       pip install transformers
"""

import sys, time, argparse, csv, platform
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from monitor import Monitor

# ── Configuración ─────────────────────────────────────────────────────────────

MODELO_HF  = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
ALFA_SEG   = 0.50
OUTPUT_DIR = Path("output/segformer_b0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NOMBRE_MODELO = "SegFormer-B0"

PALETA = np.array([
    [128, 64,  128], [244, 35,  232], [70,  70,  70 ], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30 ], [220, 220, 0  ],
    [107, 142, 35 ], [152, 251, 152], [70,  130, 180], [220, 20,  60 ],
    [255, 0,   0  ], [0,   0,   142], [0,   0,   70 ], [0,   60,  100],
    [0,   80,  100], [0,   0,   230], [119, 11,  32 ],
], dtype=np.uint8)

NOMBRES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]
CLASES_NAV = {0: "road", 1: "sidewalk", 2: "building", 11: "person", 13: "car"}


# ── Cargar modelo ─────────────────────────────────────────────────────────────

def cargar_modelo():
    try:
        from transformers import (SegformerForSemanticSegmentation,
                                  SegformerImageProcessor)
    except ImportError:
        print("  [!] pip install transformers"); sys.exit(1)
    print(f"  Cargando: {MODELO_HF}")
    proc   = SegformerImageProcessor.from_pretrained(MODELO_HF)
    modelo = SegformerForSemanticSegmentation.from_pretrained(MODELO_HF, use_safetensors=True)
    modelo.eval().to(DEVICE)
    return modelo, proc


# ── Inferencia ────────────────────────────────────────────────────────────────

def inferir(modelo, proc, frame):
    h, w   = frame.shape[:2]
    rgb    = frame[:, :, ::-1].copy()
    inputs = proc(images=rgb, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = modelo(**inputs).logits
        pred   = F.interpolate(logits, size=(h, w),
                               mode="bilinear", align_corners=False)
        pred   = pred.argmax(1).squeeze(0).byte().cpu().numpy()
    return pred, (time.perf_counter() - t0) * 1000


# ── Visualización ─────────────────────────────────────────────────────────────

def _txt(img, t, x, y, esc=0.62, col=(255,255,255), fondo=(15,15,15)):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(t, f, esc, 1)
    p = 4
    cv2.rectangle(img, (x-p, y-th-p), (x+tw+p, y+bl+p), fondo, -1)
    cv2.putText(img, t, (x, y), f, esc, col, 1, cv2.LINE_AA)

def colorear(m):
    h, w = m.shape
    c = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(PALETA)): c[m == i] = PALETA[i]
    return c

def metricas(m, h, w):
    total = h * w
    return {n: round(100.0 * float(np.sum(m == i)) / total, 2)
            for i, n in CLASES_NAV.items() if np.sum(m == i) > 0}

def _reg(n, lat_ms, rec, m):
    h, w = m.shape
    met  = metricas(m, h, w)
    return {
        "frame_id":         n,
        "latencia_ms":      round(lat_ms, 2),
        "fps_instantaneo":  round(1000 / lat_ms, 2),
        "ram_gb":           round(rec["ram_gb"], 2),
        "gpu_mb":           round(rec["gpu_mb"], 1),
        "cpu_pct":          round(rec["cpu_pct"], 1),
        "power_w":          round(rec.get("power_w", 0), 2),
        "temp_c":           round(rec.get("temp_c", 0), 1),
        "clases_detectadas": len(np.unique(m)),
        "pct_sidewalk":     met.get("sidewalk", 0),
        "pct_road":         met.get("road", 0),
        "pct_person":       met.get("person", 0),
    }

def componer(frame, m, lat_ms, rec, fid, monitor):
    h, w   = frame.shape[:2]
    cmask  = colorear(m)
    fusion = cv2.addWeighted(frame, 1-ALFA_SEG, cmask, ALFA_SEG, 0)
    sw     = (m == 1).astype(np.uint8)
    conts, _ = cv2.findContours(sw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fusion, conts, -1, (244, 35, 232), 2)

    fps = 1000 / lat_ms if lat_ms > 0 else 0
    met = metricas(m, h, w)
    pw  = rec.get("power_w", 0)
    PASO, M = 24, 8

    izq = [
        (f"{NOMBRE_MODELO}  [B3]",       (0, 220, 255)),
        (f"Frame    {fid:>5}",            (210, 210, 210)),
        (f"FPS      {fps:>6.1f}",         (100,255,100) if fps>=15 else (0,60,255)),
        (f"Lat      {lat_ms:>5.1f} ms",   (100,255,100) if lat_ms<300 else (0,60,255)),
        (f"Clases   {len(np.unique(m)):>5}", (210, 210, 210)),
        (f"Power    {pw:>5.1f} W",        (200, 255, 200)),
    ]
    for i, (t, c) in enumerate(izq):
        _txt(fusion, t, M, M+18+i*PASO, col=c)

    y_met = M + 18 + len(izq)*PASO + 6
    for nombre, pct in met.items():
        col = (244, 35, 232) if nombre == "sidewalk" else (210, 210, 210)
        _txt(fusion, f"  {nombre}: {pct:.1f}%", M, y_met, esc=0.50, col=col)
        y_met += 20

    der = [("SISTEMA", (0,220,255)),
           (f"RAM sys {rec['ram_gb']:>4.1f} GB", (210,210,210)),
           (f"CPU     {rec['cpu_pct']:>5.1f} %",  (210,210,210)),
           (f"VRAM    {rec['gpu_mb']:>6.0f} MB",  (190,140,255)),
           (f"GPU     {rec['gpu_pct']:>5.1f} %",  (190,140,255))]
    xd = w - 220
    for i, (t, c) in enumerate(der):
        _txt(fusion, t, xd, M+18+i*PASO, col=c)

    clases = [c for c in np.unique(m) if c < len(NOMBRES)]
    for i, cls_id in enumerate(clases[:8]):
        color = tuple(int(c) for c in PALETA[cls_id])
        yp = h - M - 10 - (min(len(clases), 8)-1-i)*18
        cv2.rectangle(fusion, (M, yp-12), (M+12, yp+2), color, -1)
        _txt(fusion, NOMBRES[cls_id], M+16, yp, esc=0.43,
             col=(255,255,255) if cls_id == 1 else (180,180,180))

    _txt(fusion, f"{monitor.gpu_nombre}  |  {w}x{h}  |  Cityscapes 19cls",
         M, h-M-6, esc=0.47, col=(160,160,160))

    sep   = np.full((h, 4, 3), 60, dtype=np.uint8)
    return np.hstack([frame, sep, fusion]), fusion


# ── CSV ───────────────────────────────────────────────────────────────────────

def guardar_raw(regs, ruta):
    if not regs: return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(regs[0].keys()))
        w.writeheader(); w.writerows(regs)
    print(f"  CSV raw     : {ruta}")

def guardar_resumen(regs, ruta, info_sys, fuente_nombre):
    if not regs: return

    def st(campo):
        v = np.array([r.get(campo, 0) for r in regs], dtype=float)
        return [round(float(x), 2) for x in [np.mean(v), np.std(v),
                np.min(v), np.max(v), np.percentile(v,50), np.percentile(v,95)]]

    lats  = np.array([r["latencia_ms"]     for r in regs])
    fps_v = np.array([r["fps_instantaneo"] for r in regs])
    lat_m, fps_m, lat_s = np.mean(lats), np.mean(fps_v), np.std(lats)

    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# RESUMEN {NOMBRE_MODELO}"])
        w.writerow(["# Fuente", fuente_nombre])
        w.writerow(["# Frames", len(regs)])
        w.writerow(["# Fecha",  time.strftime("%Y-%m-%d %H:%M:%S")])
        for k, v in info_sys.items(): w.writerow([f"# {k}", v])
        w.writerow([])
        w.writerow(["# CRITERIOS DE VALIDACIÓN"])
        w.writerow(["# Lat <300ms", "CUMPLE" if lat_m<300 else "NO CUMPLE", f"{lat_m:.1f}ms"])
        w.writerow(["# FPS >=15",   "CUMPLE" if fps_m>=15 else "NO CUMPLE", f"{fps_m:.1f}"])
        w.writerow(["# Estab <20%", "CUMPLE" if lat_s<lat_m*.20 else "NO CUMPLE",
                    f"std={lat_s:.1f}ms ({lat_s/lat_m*100:.1f}%)"])
        w.writerow([])
        w.writerow(["metrica","media","std","min","max","p50","p95"])
        for m in ["latencia_ms","fps_instantaneo","ram_gb","gpu_mb","cpu_pct",
                  "power_w","temp_c","clases_detectadas","pct_sidewalk","pct_road","pct_person"]:
            w.writerow([m] + st(m))
    print(f"  CSV resumen : {ruta}")

def imprimir_resumen(regs, nombre):
    if not regs: return
    lats   = np.array([r["latencia_ms"]     for r in regs])
    fps_v  = np.array([r["fps_instantaneo"] for r in regs])
    rams   = np.array([r["ram_gb"]          for r in regs])
    sw     = np.array([r.get("pct_sidewalk",0) for r in regs])
    rd     = np.array([r.get("pct_road",0)     for r in regs])
    powers = np.array([r.get("power_w",0)      for r in regs])
    temps  = np.array([r.get("temp_c",0)       for r in regs])
    lat_m, fps_m, lat_s = float(np.mean(lats)), float(np.mean(fps_v)), float(np.std(lats))

    print(f"\n  {'─'*52}")
    print(f"  RESUMEN — {NOMBRE_MODELO}  |  {nombre}")
    print(f"  {'─'*52}")
    print(f"  Frames          : {len(regs)}")
    print(f"  FPS  media      : {fps_m:.1f}  (min {np.min(fps_v):.1f} / max {np.max(fps_v):.1f})")
    print(f"  Lat. media      : {lat_m:.1f} ms  ±{lat_s:.1f}")
    print(f"  Lat. p95        : {float(np.percentile(lats,95)):.1f} ms")
    print(f"  RAM sys pico    : {float(np.max(rams)):.2f} GB")
    print(f"  Sidewalk medio  : {float(np.mean(sw)):.1f}% del frame")
    print(f"  Road medio      : {float(np.mean(rd)):.1f}% del frame")
    print(f"  Consumo medio   : {float(np.mean(powers)):.1f} W  (pico {float(np.max(powers)):.1f} W)")
    print(f"  Temp GPU media  : {float(np.mean(temps)):.1f} °C  (pico {float(np.max(temps)):.1f} °C)")
    print(f"  {'─'*52}")
    print(f"  Lat <300ms      : {'✅ CUMPLE' if lat_m<300 else '❌ NO CUMPLE'}")
    print(f"  FPS >=15        : {'✅ CUMPLE' if fps_m>=15 else '❌ NO CUMPLE'}")
    ok = lat_s < lat_m*.20
    print(f"  Estab. <20%std  : {'✅ CUMPLE' if ok else '❌ NO CUMPLE'}  ({lat_s/lat_m*100:.1f}%)")
    print()


# ── Modos de entrada ──────────────────────────────────────────────────────────

def procesar_imagen(modelo, proc, ruta, monitor, headless=False):
    frame = cv2.imread(str(ruta))
    if frame is None: print(f"  [!] {ruta}"); return
    rec = monitor.leer(); m, lat_ms = inferir(modelo, proc, frame)
    print(f"  {ruta.name:40s}  lat={lat_ms:6.1f}ms  {metricas(m,*frame.shape[:2])}")
    panel, fusion = componer(frame, m, lat_ms, rec, 1, monitor)
    cv2.imwrite(str(OUTPUT_DIR / ruta.name), fusion)
    if not headless:
        cv2.imshow(NOMBRE_MODELO, panel); cv2.waitKey(0); cv2.destroyAllWindows()
    guardar_raw([_reg(1,lat_ms,rec,m)], OUTPUT_DIR/f"{ruta.stem}_raw.csv")
    guardar_resumen([_reg(1,lat_ms,rec,m)], OUTPUT_DIR/f"{ruta.stem}_resumen.csv",
                    monitor.info_sistema(), ruta.name)

def procesar_carpeta(modelo, proc, ruta, monitor, headless=False):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    imgs = sorted([p for p in ruta.iterdir() if p.suffix.lower() in exts])
    if not imgs: print(f"  [!] Sin imágenes en {ruta}"); return
    print(f"  {len(imgs)} imágenes."); regs = []
    for i, ip in enumerate(imgs, 1):
        frame = cv2.imread(str(ip))
        if frame is None: continue
        rec = monitor.leer(); m, lat_ms = inferir(modelo, proc, frame)
        print(f"  {ip.name:40s}  lat={lat_ms:6.1f}ms  {metricas(m,*frame.shape[:2])}")
        regs.append(_reg(i, lat_ms, rec, m))
        panel, fusion = componer(frame, m, lat_ms, rec, i, monitor)
        cv2.imwrite(str(OUTPUT_DIR/ip.name), fusion)
        if not headless:
            cv2.imshow(NOMBRE_MODELO, panel)
            if cv2.waitKey(1)&0xFF in (ord("q"),27): break
    if not headless: cv2.destroyAllWindows()
    guardar_raw(regs, OUTPUT_DIR/f"{ruta.name}_raw.csv")
    guardar_resumen(regs, OUTPUT_DIR/f"{ruta.name}_resumen.csv",
                    monitor.info_sistema(), str(ruta))
    imprimir_resumen(regs, ruta.name)

def procesar_video(modelo, proc, fuente, monitor, headless=False):
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened(): print(f"  [!] {fuente}"); return
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    es_wc   = isinstance(fuente, int)
    nom     = "webcam" if es_wc else Path(str(fuente)).stem
    writer  = cv2.VideoWriter(str(OUTPUT_DIR/f"{nom}_segformer_b0.mp4"),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W,H))
    print(f"  {W}×{H} @ {fps_src:.0f}fps  —  headless={headless}\n")
    regs=[]; n=0; pausado=False; scap=0; vis=None

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret: break
            n += 1
            rec = monitor.leer(); m, lat_ms = inferir(modelo, proc, frame)
            regs.append(_reg(n, lat_ms, rec, m))
            panel, fusion = componer(frame, m, lat_ms, rec, n, monitor)
            vis = fusion; writer.write(fusion)
            if n % 30 == 0:
                fps_r = float(np.mean([r["fps_instantaneo"] for r in regs[-30:]]))
                sw = regs[-1].get("pct_sidewalk",0)
                pw = rec.get("power_w",0); tc = rec.get("temp_c",0)
                print(f"  Frame {n:>5}  lat={lat_ms:5.1f}ms  FPS={fps_r:.1f}"
                      f"  sw={sw:.1f}%  VRAM={rec['gpu_mb']:.0f}MB"
                      f"  GPU={rec['gpu_pct']:.0f}%  {pw:.1f}W  {tc:.0f}°C")
            if not headless: cv2.imshow(NOMBRE_MODELO, panel)
        key = (cv2.waitKey(1)&0xFF) if not headless else 255
        if key in (ord("q"),27): break
        elif key == ord(" "):
            pausado = not pausado
            print(f"  {'⏸ Pausado' if pausado else '▶ Reanudado'}")
        elif key == ord("s") and vis is not None:
            scap+=1; p=OUTPUT_DIR/f"{nom}_captura_{scap:03d}.jpg"
            cv2.imwrite(str(p),vis); print(f"  📸 {p}")

    cap.release(); writer.release()
    if not headless: cv2.destroyAllWindows()
    guardar_raw(regs, OUTPUT_DIR/f"{nom}_raw.csv")
    guardar_resumen(regs, OUTPUT_DIR/f"{nom}_resumen.csv",
                    monitor.info_sistema(), str(fuente))
    imprimir_resumen(regs, nom)
    print(f"  Video : {OUTPUT_DIR/f'{nom}_segformer_b0.mp4'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global ALFA_SEG
    parser = argparse.ArgumentParser(description=f"{NOMBRE_MODELO} — Cityscapes")
    parser.add_argument("--fuente",     required=True)
    parser.add_argument("--no-display", action="store_true", help="Headless")
    parser.add_argument("--alfa",       type=float, default=ALFA_SEG)
    args = parser.parse_args()
    ALFA_SEG = args.alfa

    print(f"\n{'─'*55}")
    print(f"  {NOMBRE_MODELO}  —  Cityscapes 19 clases  |  Taller I")
    print(f"  Modelo: {MODELO_HF}")
    print(f"  device={DEVICE}  alfa={ALFA_SEG}  headless={args.no_display}")
    print(f"{'─'*55}")

    monitor = Monitor()
    info    = monitor.info_sistema()
    print(f"  Plataforma : {info['plataforma']}")
    print(f"  GPU        : {info['gpu']}  ({info['gpu_vram_gb']} GB VRAM)")
    print(f"  RAM        : {info['ram_total_gb']} GB total")
    print(f"  CUDA       : {info['cuda']}\n")

    print("  Cargando modelo...")
    t0 = time.perf_counter()
    modelo, proc = cargar_modelo()
    print(f"  Listo en {(time.perf_counter()-t0)*1000:.0f} ms\n")

    fuente = args.fuente
    try:
        fuente = int(fuente)
        procesar_video(modelo, proc, fuente, monitor, headless=args.no_display); return
    except ValueError: pass

    ruta = Path(fuente)
    if not ruta.exists(): print(f"  [!] No existe: {ruta}"); sys.exit(1)
    if ruta.is_dir():
        procesar_carpeta(modelo, proc, ruta, monitor, headless=args.no_display)
    elif ruta.suffix.lower() in {".mp4",".avi",".mov",".mkv",".webm"}:
        procesar_video(modelo, proc, str(ruta), monitor, headless=args.no_display)
    else:
        procesar_imagen(modelo, proc, ruta, monitor, headless=args.no_display)


if __name__ == "__main__":
    main()