"""
Evaluacion de precision (mIoU, IoU por clase, metricas de zona transitable) y
latencia por etapa de los modelos de segmentacion en el Jetson Orin Nano.

Backends:
    trt  -> engines TensorRT FP16 (lo que corre en despliegue)
    onnx -> modelos ONNX FP32 con ONNX Runtime (referencia para aislar el efecto de FP16)

Ambos usan exactamente el mismo preprocesamiento y postprocesamiento, asi que la
diferencia entre ellos se atribuye solo a la precision reducida.

Uso:
    python eval_segmentacion_jetson.py --backend trt
    python eval_segmentacion_jetson.py --backend onnx
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

H, W = 512, 896
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE = Path.home() / "Assistive-Navigation"
DIR_TEST = BASE / "data" / "test_lapaz" / "test"
DIR_MODELOS = BASE / "modelos" / "segmentacion"
DIR_LOGS = BASE / "logs" / "benchmark_segmentacion"

MODELOS = {
    "segformer_b0": ("segformer_b0_512x896_mask", "segformer"),
    "segformer_b1": ("segformer_b1_512x896_mask", "segformer"),
    "yolo26n_sem": ("yolo26n_sem_512x896", "yolo"),
}


# ---------------------------------------------------------------------------
# Clases
# ---------------------------------------------------------------------------
def cargar_clases(ruta_csv):
    clases_por_valor = {}
    with open(ruta_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for fila in reader:
            clases_por_valor[int(fila[0].strip())] = fila[1].strip()
    nombres = []
    for n in clases_por_valor.values():
        if n not in nombres:
            nombres.append(n)
    tabla = np.zeros(256, dtype=np.uint8)
    for valor, n in clases_por_valor.items():
        tabla[valor] = nombres.index(n)
    return nombres, tabla


# ---------------------------------------------------------------------------
# Pre y postprocesamiento (identicos para trt y onnx)
# ---------------------------------------------------------------------------
def preprocesar(img_bgr, tipo):
    """Devuelve el tensor de entrada (1,3,H,W) float32 y la info de padding (solo YOLO)."""
    if tipo == "segformer":
        # Igual que SegformerImageProcessor: resize PIL bilineal, /255, normalizacion ImageNet
        rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        x = np.asarray(rgb.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), None

    # YOLO: letterbox igual que Ultralytics (mantiene proporcion, relleno gris 114, /255)
    h0, w0 = img_bgr.shape[:2]
    r = min(H / h0, W / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    dw, dh = (W - nw) / 2, (H - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    x = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.copyMakeBorder(x, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (top, left, nh, nw)


def postprocesar(mascara, tipo, pad, forma_original):
    """Lleva la mascara (H,W) de vuelta a la resolucion original de la imagen."""
    m = np.asarray(mascara).reshape(H, W).astype(np.uint8)
    if tipo == "yolo":
        top, left, nh, nw = pad
        m = m[top:top + nh, left:left + nw]
    alto, ancho = forma_original
    return cv2.resize(m, (ancho, alto), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class MotorTRT:
    def __init__(self, ruta_engine):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
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
            self.buffers[self.entrada].copy_(self.torch.from_numpy(x), non_blocking=False)
            self.contexto.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buffers[self.salida].cpu().numpy()


class MotorONNX:
    def __init__(self, ruta_onnx):
        import onnxruntime as ort

        disponibles = ort.get_available_providers()
        proveedores = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in disponibles]
        self.sesion = ort.InferenceSession(str(ruta_onnx), providers=proveedores)
        self.entrada = self.sesion.get_inputs()[0].name
        print(f"    ONNX Runtime usando: {self.sesion.get_providers()[0]}")

    def __call__(self, x):
        return self.sesion.run(None, {self.entrada: x})[0]


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def metricas(conf, nombres):
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    i_side, i_road, i_bg = nombres.index("sidewalk"), nombres.index("road"), nombres.index("background")
    tp = conf[i_side, i_side]
    fp = conf[:, i_side].sum() - tp
    fn = conf[i_side].sum() - tp
    norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    return {
        "miou": float(iou.mean()),
        "iou_por_clase": {n: float(iou[i]) for i, n in enumerate(nombres)},
        "precision_transitable": float(tp / max(tp + fp, 1)),
        "recall_transitable": float(tp / max(tp + fn, 1)),
        "calle_como_acera": float(norm[i_road, i_side]),
        "background_como_acera": float(norm[i_bg, i_side]),
    }


def resumen_tiempos(lista_ms):
    a = np.array(lista_ms)
    return {"mediana_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["trt", "onnx"], required=True)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    nombres, tabla_remapeo = cargar_clases(DIR_TEST / "_classes.csv")
    n_clases = len(nombres)
    imagenes = sorted(f for f in DIR_TEST.glob("*.jpg"))
    print(f"Clases: {nombres}")
    print(f"Imagenes de test: {len(imagenes)} | backend: {args.backend}\n")

    resultados = {}
    for nombre_modelo, (base_archivo, tipo) in MODELOS.items():
        print(f"== {nombre_modelo} ==")
        if args.backend == "trt":
            motor = MotorTRT(DIR_MODELOS / "engines" / f"{base_archivo}_fp16.engine")
        else:
            motor = MotorONNX(DIR_MODELOS / "onnx" / f"{base_archivo}.onnx")

        # Calentamiento con la primera imagen (no se mide)
        x_warm, _ = preprocesar(cv2.imread(str(imagenes[0])), tipo)
        for _ in range(args.warmup):
            motor(x_warm)

        conf = np.zeros((n_clases, n_clases), dtype=np.int64)
        t_pre, t_inf, t_post, t_total = [], [], [], []

        for ruta in imagenes:
            img = cv2.imread(str(ruta))
            gt_raw = np.array(Image.open(ruta.with_name(ruta.stem + "_mask.png")).convert("L"))
            gt = tabla_remapeo[gt_raw].astype(np.int64)

            t0 = time.perf_counter()
            x, pad = preprocesar(img, tipo)
            t1 = time.perf_counter()
            salida = motor(x)
            t2 = time.perf_counter()
            pred = postprocesar(salida, tipo, pad, img.shape[:2]).astype(np.int64)
            t3 = time.perf_counter()

            t_pre.append((t1 - t0) * 1000)
            t_inf.append((t2 - t1) * 1000)
            t_post.append((t3 - t2) * 1000)
            t_total.append((t3 - t0) * 1000)

            idx = gt.flatten() * n_clases + pred.flatten()
            conf += np.bincount(idx, minlength=n_clases ** 2).reshape(n_clases, n_clases)

        met = metricas(conf, nombres)
        met["tiempos"] = {
            "preprocesamiento": resumen_tiempos(t_pre),
            "inferencia": resumen_tiempos(t_inf),
            "postprocesamiento": resumen_tiempos(t_post),
            "total": resumen_tiempos(t_total),
        }
        resultados[nombre_modelo] = met

        print(f"   mIoU {met['miou']:.4f} | sidewalk {met['iou_por_clase']['sidewalk']:.4f} | "
              f"pole {met['iou_por_clase']['pole']:.4f} | precision transitable {met['precision_transitable']:.4f} | "
              f"calle->acera {met['calle_como_acera']:.4f}")
        tt = met["tiempos"]
        print(f"   tiempos mediana (ms): pre {tt['preprocesamiento']['mediana_ms']:.2f} | "
              f"inferencia {tt['inferencia']['mediana_ms']:.2f} | post {tt['postprocesamiento']['mediana_ms']:.2f} | "
              f"total {tt['total']['mediana_ms']:.2f}\n")

        del motor

    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    salida_json = DIR_LOGS / f"precision_{args.backend}.json"
    with open(salida_json, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"Resultados guardados en: {salida_json}")


if __name__ == "__main__":
    main()
