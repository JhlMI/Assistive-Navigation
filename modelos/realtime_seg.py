#!/usr/bin/env python3
"""
realtime_seg.py  —  Segmentación Cityscapes en tiempo real
===========================================================
Sin guardado. Sin CSV. Máxima velocidad posible.

Uso:
    python realtime_seg.py --fuente 0              # webcam
    python realtime_seg.py --fuente video.mp4
    python realtime_seg.py --fuente video.mp4 --modelo b0   # más rápido
    python realtime_seg.py --fuente video.mp4 --modelo b3   # más preciso
    python realtime_seg.py --fuente video.mp4 --skip 2      # saltar frames
    python realtime_seg.py --fuente video.mp4 --alfa 0.6    # más opacidad

Modelos disponibles (de más rápido a más preciso):
    b0  ~15MB   — máxima velocidad
    b3  ~80MB   — balance (DDRNet proxy)
    b5  ~180MB  — máxima precisión

Controles:
    Q / ESC     — salir
    ESPACIO     — pausar
    A / D       — bajar/subir opacidad máscara
    0-5         — resaltar clase específica
                  0=todas  1=sidewalk  2=road  3=building  4=person  5=car
"""

import sys, time, argparse
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── Configuración ─────────────────────────────────────────────────────────────

MODELOS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    "b3": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paleta Cityscapes BGR
PALETA = np.array([
    [128, 64,  128],   # 0  road
    [244, 35,  232],   # 1  sidewalk   ← ACERA (magenta)
    [70,  70,  70 ],   # 2  building
    [102, 102, 156],   # 3  wall
    [190, 153, 153],   # 4  fence
    [153, 153, 153],   # 5  pole
    [250, 170, 30 ],   # 6  traffic light
    [220, 220, 0  ],   # 7  traffic sign
    [107, 142, 35 ],   # 8  vegetation
    [152, 251, 152],   # 9  terrain
    [70,  130, 180],   # 10 sky
    [220, 20,  60 ],   # 11 person
    [255, 0,   0  ],   # 12 rider
    [0,   0,   142],   # 13 car
    [0,   0,   70 ],   # 14 truck
    [0,   60,  100],   # 15 bus
    [0,   80,  100],   # 16 train
    [0,   0,   230],   # 17 motorcycle
    [119, 11,  32 ],   # 18 bicycle
], dtype=np.uint8)

NOMBRES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain","sky",
    "person","rider","car","truck","bus","train","motorcycle","bicycle"
]

# Clases resaltables con tecla numérica
CLASE_KEYS = {
    ord("1"): 1,   # sidewalk
    ord("2"): 0,   # road
    ord("3"): 2,   # building
    ord("4"): 11,  # person
    ord("5"): 13,  # car
    ord("0"): -1,  # todas
}


# ── Cargar modelo ─────────────────────────────────────────────────────────────

def cargar_modelo(nombre):
    from transformers import (SegformerForSemanticSegmentation,
                              SegformerImageProcessor)
    model_id = MODELOS[nombre]
    print(f"  Cargando: {model_id}")
    proc   = SegformerImageProcessor.from_pretrained(model_id)
    modelo = SegformerForSemanticSegmentation.from_pretrained(model_id)
    modelo.eval().to(DEVICE)
    # Warmup
    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    for _ in range(3):
        _inferir(modelo, proc, dummy)
    print("  Listo.\n")
    return modelo, proc


# ── Inferencia ────────────────────────────────────────────────────────────────

def _inferir(modelo, proc, frame):
    h, w = frame.shape[:2]
    rgb  = frame[:, :, ::-1].copy()
    inp  = proc(images=rgb, return_tensors="pt")
    inp  = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        logits = modelo(**inp).logits
        pred   = F.interpolate(logits, size=(h, w),
                               mode="bilinear", align_corners=False)
        pred   = pred.argmax(1).squeeze(0).byte().cpu().numpy()
    return pred


# ── Colorear máscara ──────────────────────────────────────────────────────────

def colorear(mascara, clase_resaltada=-1):
    h, w  = mascara.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    if clase_resaltada == -1:
        # Todas las clases
        for i in range(len(PALETA)):
            color[mascara == i] = PALETA[i]
    else:
        # Solo la clase seleccionada, resto en gris oscuro
        color[:] = 30
        color[mascara == clase_resaltada] = PALETA[clase_resaltada]

    return color


# ── Overlay ───────────────────────────────────────────────────────────────────

def _txt(img, texto, x, y, esc=0.60, col=(255,255,255), fondo=(15,15,15)):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(texto, f, esc, 1)
    p = 3
    cv2.rectangle(img, (x-p, y-th-p), (x+tw+p, y+bl+p), fondo, -1)
    cv2.putText(img, texto, (x, y), f, esc, col, 1, cv2.LINE_AA)


def dibujar_overlay(fusion, mascara, lat_ms, fps_avg,
                    clase_resaltada, alfa, frame_id):
    h, w = fusion.shape[:2]
    fps_col = (100, 255, 100) if fps_avg >= 15 else (0, 60, 255)
    lat_col = (100, 255, 100) if lat_ms  < 300 else (0, 60, 255)

    # Calcular % de sidewalk y road
    total = h * w
    pct_sw = 100.0 * float(np.sum(mascara == 1)) / total
    pct_rd = 100.0 * float(np.sum(mascara == 0)) / total

    PASO, M = 22, 8
    lineas = [
        (f"SegFormer  Cityscapes",           (0, 220, 255)),
        (f"Frame   {frame_id:>6}",           (200, 200, 200)),
        (f"FPS     {fps_avg:>6.1f}",         fps_col),
        (f"Lat     {lat_ms:>5.1f} ms",       lat_col),
        (f"sidewalk {pct_sw:>5.1f}%",        (244, 35, 232)),
        (f"road     {pct_rd:>5.1f}%",        (128, 100, 128)),
    ]
    for i, (t, c) in enumerate(lineas):
        _txt(fusion, t, M, M + 16 + i*PASO, col=c)

    # Clase resaltada activa
    if clase_resaltada >= 0:
        nombre = NOMBRES[clase_resaltada] if clase_resaltada < len(NOMBRES) else "?"
        color  = tuple(int(c) for c in PALETA[clase_resaltada])
        _txt(fusion, f"[RESALTADO: {nombre}]",
             M, M + 16 + len(lineas)*PASO + 4,
             esc=0.55, col=color)

    # Controles en pie
    controles = "Q=salir  SPC=pausa  A/D=opacidad  1=sidewalk 2=road 3=building 4=person 0=todas"
    _txt(fusion, controles, M, h-M-6, esc=0.42, col=(140,140,140))

    # Alfa actual
    _txt(fusion, f"alfa={alfa:.2f}", w-90, h-M-6, esc=0.48, col=(160,160,160))


# ── Loop principal ────────────────────────────────────────────────────────────

def run(modelo, proc, fuente, skip, alfa_init):
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"  [!] No se pudo abrir: {fuente}"); return

    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    es_webcam = isinstance(fuente, int)

    # Tiempo entre frames para respetar velocidad del video
    # En webcam no hay timing — mostrar tan rápido como sea posible
    frame_duration = 0.0 if es_webcam else (1.0 / fps_src)

    print(f"  {W}×{H} @ {fps_src:.0f}fps  —  CUDA: {torch.cuda.is_available()}\n")

    fps_window    = deque(maxlen=30)
    alfa          = alfa_init
    clase_res     = -1
    pausado       = False
    n_frame       = 0
    n_procesado   = 0
    ultima_mascara = None
    lat_ultima     = 0.0
    t_ultimo_frame = time.perf_counter()

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                break
            n_frame += 1

            # ── Inferencia (en frames que no se saltan) ──
            if skip <= 1 or n_frame % skip == 0:
                t0  = time.perf_counter()
                msc = _inferir(modelo, proc, frame)
                lat = (time.perf_counter() - t0) * 1000
                fps_window.append(lat)
                ultima_mascara = msc
                lat_ultima     = lat
                n_procesado   += 1
            # En frames saltados reusar la última máscara

            fps_avg = 1000 / np.mean(fps_window) if fps_window else 0

            if ultima_mascara is not None:
                # Contorno de acera
                sw_bin = (ultima_mascara == 1).astype(np.uint8)
                conts, _ = cv2.findContours(sw_bin, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                cmask  = colorear(ultima_mascara, clase_res)
                fusion = cv2.addWeighted(frame, 1-alfa, cmask, alfa, 0)
                cv2.drawContours(fusion, conts, -1, (255, 255, 255), 2)
                dibujar_overlay(fusion, ultima_mascara, lat_ultima,
                                fps_avg, clase_res, alfa, n_frame)
                cv2.imshow("Segmentación RT", fusion)

            # ── Timing: esperar lo necesario para respetar FPS fuente ──
            if frame_duration > 0:
                ahora    = time.perf_counter()
                t_sig    = t_ultimo_frame + frame_duration
                espera   = t_sig - ahora
                if espera > 0:
                    # waitKey con el tiempo restante (mínimo 1ms)
                    cv2.waitKey(max(1, int(espera * 1000)))
                else:
                    cv2.waitKey(1)
                t_ultimo_frame = time.perf_counter()
            else:
                cv2.waitKey(1)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            pausado = not pausado
            print(f"  {'⏸' if pausado else '▶'}")
        elif key == ord("a"):
            alfa = max(0.1, alfa - 0.05)
        elif key == ord("d"):
            alfa = min(0.9, alfa + 0.05)
        elif key in CLASE_KEYS:
            clase_res = CLASE_KEYS[key]
            nombre = NOMBRES[clase_res] if clase_res >= 0 else "TODAS"
            print(f"  Resaltando: {nombre}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Frames totales  : {n_frame}")
    print(f"  Frames procesados: {n_procesado}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Segmentación RT Cityscapes")
    parser.add_argument("--fuente", required=True,
                        help="video.mp4 | 0 (webcam)")
    parser.add_argument("--modelo", default="b0",
                        choices=["b0","b3","b5"],
                        help="b0=rápido  b3=balance  b5=preciso (default: b0)")
    parser.add_argument("--skip",   type=int, default=1,
                        help="Procesar 1 de cada N frames (default: 1=todos)")
    parser.add_argument("--alfa",   type=float, default=0.55,
                        help="Opacidad inicial máscara 0.0-1.0 (default: 0.55)")
    args = parser.parse_args()

    print(f"\n{'─'*50}")
    print(f"  Segmentación RT  —  Cityscapes 19 clases")
    print(f"  Modelo: {MODELOS[args.modelo]}")
    print(f"  Device: {DEVICE}  skip={args.skip}  alfa={args.alfa}")
    print(f"{'─'*50}\n")

    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        print("  [!] pip install transformers"); sys.exit(1)

    modelo, proc = cargar_modelo(args.modelo)

    fuente = args.fuente
    try:
        fuente = int(fuente)
    except ValueError:
        pass

    run(modelo, proc, fuente, args.skip, args.alfa)


if __name__ == "__main__":
    main()
