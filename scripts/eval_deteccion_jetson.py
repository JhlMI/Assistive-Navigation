"""
Benchmark del detector YOLOv8n (TensorRT FP16) en el Jetson Orin Nano.

Para cada resolucion:
  - Tiempos por etapa (preprocesamiento, inferencia, postprocesamiento con NMS)
  - Verificacion contra el modelo original yolov8n.pt (Ultralytics, PyTorch):
    porcentaje de detecciones del .pt que el engine reproduce (misma clase, IoU >= 0.5).
    No es mAP (no hay cajas anotadas de La Paz); mide que la conversion a TensorRT
    FP16 no cambio el comportamiento del detector.
  - Imagenes anotadas para revision visual

Uso:
    python eval_deteccion_jetson.py
    python eval_deteccion_jetson.py --resoluciones 384x640 --guardar 10
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# Ultralytics desactiva el multihilo de OpenCV al importarse (cv2.setNumThreads(0)).
# Guardamos el valor por defecto para restaurarlo y medir en las mismas condiciones
# que eval_segmentacion_jetson.py (que no importa ultralytics).
HILOS_OPENCV_ORIGINAL = cv2.getNumThreads()

BASE = Path.home() / "Assistive-Navigation"
DIR_TEST = BASE / "data" / "test_lapaz" / "test"
DIR_ENGINES = BASE / "modelos" / "deteccion" / "engines"
DIR_LOGS = BASE / "logs" / "benchmark_deteccion"
RUTA_PT = BASE / "yolov8n.pt"


# ---------------------------------------------------------------------------
# Motor TensorRT
# ---------------------------------------------------------------------------
class MotorTRT:
    def __init__(self, ruta_engine):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.ERROR)
        with open(ruta_engine, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.contexto = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buffers = {}

        for i in range(self.engine.num_io_tensors):
            nombre = self.engine.get_tensor_name(i)
            forma = tuple(self.engine.get_tensor_shape(nombre))
            np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(nombre)))
            torch_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
            buffer = torch.empty(forma, dtype=torch_dtype, device="cuda")
            self.buffers[nombre] = buffer
            self.contexto.set_tensor_address(nombre, buffer.data_ptr())
            if self.engine.get_tensor_mode(nombre) == trt.TensorIOMode.INPUT:
                self.entrada = nombre
            else:
                self.salida = nombre

    def __call__(self, x):
        with self.torch.cuda.stream(self.stream):
            self.buffers[self.entrada].copy_(self.torch.from_numpy(x))
            self.contexto.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buffers[self.salida].cpu().numpy()


# ---------------------------------------------------------------------------
# Pre y postprocesamiento (equivalentes a Ultralytics)
# ---------------------------------------------------------------------------
def letterbox(img_bgr, h, w):
    h0, w0 = img_bgr.shape[:2]
    r = min(h / h0, w / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    dw, dh = (w - nw) / 2, (h - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    x = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, left, top


def postprocesar(salida, r, pad_x, pad_y, forma_original, conf_min, iou_nms):
    """Salida YOLOv8 (1, 84, N) -> cajas xyxy en coordenadas de la imagen original, scores, clases."""
    pred = salida[0].T                      # (N, 84): cx, cy, w, h, 80 scores
    scores_clase = pred[:, 4:]
    clases = scores_clase.argmax(axis=1)
    scores = scores_clase[np.arange(len(pred)), clases]
    keep = scores >= conf_min
    if not keep.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    pred, scores, clases = pred[keep], scores[keep], clases[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

    # NMS por clase: desplaza las cajas segun la clase para que no se supriman entre clases distintas
    offset = clases * 4096.0
    cajas_nms = np.stack([cx - bw / 2 + offset, cy - bh / 2 + offset, bw, bh], axis=1)
    indices = cv2.dnn.NMSBoxes(cajas_nms.tolist(), scores.tolist(), conf_min, iou_nms)
    indices = np.array(indices).flatten() if len(indices) else np.array([], dtype=int)

    x1 = (cx - bw / 2 - pad_x) / r
    y1 = (cy - bh / 2 - pad_y) / r
    x2 = (cx + bw / 2 - pad_x) / r
    y2 = (cy + bh / 2 - pad_y) / r
    cajas = np.stack([x1, y1, x2, y2], axis=1)
    alto, ancho = forma_original
    cajas[:, [0, 2]] = cajas[:, [0, 2]].clip(0, ancho)
    cajas[:, [1, 3]] = cajas[:, [1, 3]].clip(0, alto)
    return cajas[indices], scores[indices], clases[indices]


# ---------------------------------------------------------------------------
# Verificacion contra el .pt
# ---------------------------------------------------------------------------
def iou_matriz(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def contar_coincidencias(cajas_ref, clases_ref, cajas_trt, clases_trt, umbral=0.5):
    """Cuantas detecciones de referencia (.pt) tienen una del engine con misma clase e IoU >= umbral."""
    if len(cajas_ref) == 0:
        return 0
    iou = iou_matriz(cajas_ref, cajas_trt)
    usadas = set()
    coincidencias = 0
    for i in np.argsort(-iou.max(axis=1) if iou.size else np.zeros(len(cajas_ref))):
        for j in np.argsort(-iou[i]) if iou.size else []:
            if j in usadas or iou[i, j] < umbral:
                continue
            if clases_ref[i] == clases_trt[j]:
                usadas.add(j)
                coincidencias += 1
                break
    return coincidencias


def resumen_tiempos(lista_ms):
    a = np.array(lista_ms)
    return {"mediana_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


def dibujar(img, cajas, scores, clases, nombres):
    salida = img.copy()
    for (x1, y1, x2, y2), s, c in zip(cajas.astype(int), scores, clases):
        cv2.rectangle(salida, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(salida, f"{nombres[int(c)]} {s:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return salida


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resoluciones", nargs="+", default=["384x640", "512x896"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--guardar", type=int, default=6, help="Cuantas imagenes anotadas guardar")
    parser.add_argument("--sin-verificacion", action="store_true",
                        help="No corre el .pt de PyTorch en el bucle: usar para medir tiempos limpios")
    args = parser.parse_args()

    from ultralytics import YOLO
    print(f"Hilos OpenCV: original {HILOS_OPENCV_ORIGINAL} | tras importar ultralytics {cv2.getNumThreads()}")
    cv2.setNumThreads(HILOS_OPENCV_ORIGINAL)
    print(f"Hilos OpenCV restaurados a: {cv2.getNumThreads()}")
    modelo_pt = YOLO(str(RUTA_PT))
    nombres = modelo_pt.names
    if args.sin_verificacion:
        del modelo_pt
        import gc
        gc.collect()

    imagenes = sorted(DIR_TEST.glob("*.jpg"))
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    print(f"Imagenes de test: {len(imagenes)} | conf {args.conf} | iou NMS {args.iou}\n")

    resultados = {}
    for res in args.resoluciones:
        h, w = map(int, res.split("x"))
        print(f"== yolov8n {res} ==")
        motor = MotorTRT(DIR_ENGINES / f"yolov8n_{res}_fp16.engine")

        x_warm, *_ = letterbox(cv2.imread(str(imagenes[0])), h, w)
        for _ in range(args.warmup):
            motor(x_warm)

        t_pre, t_inf, t_post, t_total = [], [], [], []
        n_ref_total, n_coinciden, n_trt_total = 0, 0, 0
        dir_anotadas = DIR_LOGS / f"anotadas_{res}"
        dir_anotadas.mkdir(exist_ok=True)

        for k, ruta in enumerate(imagenes):
            img = cv2.imread(str(ruta))

            t0 = time.perf_counter()
            x, r, pad_x, pad_y = letterbox(img, h, w)
            t1 = time.perf_counter()
            salida = motor(x)
            t2 = time.perf_counter()
            cajas, scores, clases = postprocesar(salida, r, pad_x, pad_y, img.shape[:2], args.conf, args.iou)
            t3 = time.perf_counter()

            t_pre.append((t1 - t0) * 1000)
            t_inf.append((t2 - t1) * 1000)
            t_post.append((t3 - t2) * 1000)
            t_total.append((t3 - t0) * 1000)

            n_trt_total += len(cajas)
            if not args.sin_verificacion:
                ref = modelo_pt.predict(str(ruta), imgsz=[h, w], conf=args.conf, iou=args.iou, verbose=False)[0]
                cajas_ref = ref.boxes.xyxy.cpu().numpy()
                clases_ref = ref.boxes.cls.cpu().numpy().astype(int)
                n_ref_total += len(cajas_ref)
                n_coinciden += contar_coincidencias(cajas_ref, clases_ref, cajas, clases)

            if k < args.guardar:
                cv2.imwrite(str(dir_anotadas / ruta.name), dibujar(img, cajas, scores, clases, nombres))

        coincidencia = None if args.sin_verificacion else n_coinciden / max(n_ref_total, 1)
        resultados[res] = {
            "detecciones_pt": None if args.sin_verificacion else n_ref_total,
            "detecciones_trt": n_trt_total,
            "coincidencia_pt_vs_trt": None if coincidencia is None else round(coincidencia, 4),
            "tiempos": {
                "preprocesamiento": resumen_tiempos(t_pre),
                "inferencia": resumen_tiempos(t_inf),
                "postprocesamiento": resumen_tiempos(t_post),
                "total": resumen_tiempos(t_total),
            },
        }
        tt = resultados[res]["tiempos"]
        if coincidencia is None:
            print(f"   detecciones engine {n_trt_total} (modo solo tiempos, sin verificacion)")
        else:
            print(f"   detecciones .pt {n_ref_total} | engine {n_trt_total} | coincidencia {coincidencia:.4f}")
        print(f"   tiempos mediana (ms): pre {tt['preprocesamiento']['mediana_ms']:.2f} | "
              f"inferencia {tt['inferencia']['mediana_ms']:.2f} | post {tt['postprocesamiento']['mediana_ms']:.2f} | "
              f"total {tt['total']['mediana_ms']:.2f}")
        print(f"   imagenes anotadas en: {dir_anotadas}\n")
        del motor

    salida_json = DIR_LOGS / ("deteccion_trt_tiempos.json" if args.sin_verificacion else "deteccion_trt.json")
    with open(salida_json, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"Resultados guardados en: {salida_json}")


if __name__ == "__main__":
    main()