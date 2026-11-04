"""Perception: YOLO26n-sem segmentation + YOLOv8n detection with TensorRT FP16.

Same pre/post-processing used in the power tests.
Does not import ultralytics (it drops OpenCV to 1 thread and adds latency).
"""
import time

import cv2
import numpy as np

import config as C


class TrtEngine:
    def __init__(self, engine_path):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            torch_dtype = torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
            buffer = torch.empty(shape, dtype=torch_dtype, device="cuda")
            self.buffers[name] = buffer
            self.context.set_tensor_address(name, buffer.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input = name
            else:
                self.output = name

    def __call__(self, x):
        with self.torch.cuda.stream(self.stream):
            self.buffers[self.input].copy_(self.torch.from_numpy(x))
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.buffers[self.output].cpu().numpy()


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
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, left, top, nh, nw


_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def walkable_sidewalk(mask):
    """Walkable area = 'sidewalk' components touching the bottom edge."""
    sidewalk = (mask == C.SIDEWALK_ID).astype(np.uint8)
    sidewalk = cv2.morphologyEx(sidewalk, cv2.MORPH_OPEN, _KERNEL)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sidewalk, connectivity=8)
    if n <= 1:
        return np.zeros_like(sidewalk)
    valid = [i for i in np.unique(labels[-1])
             if i != 0 and stats[i, cv2.CC_STAT_AREA] >= C.SIDEWALK_MIN_AREA]
    if not valid:
        return np.zeros_like(sidewalk)
    return np.isin(labels, valid).astype(np.uint8)


def postprocess_detection(output, r, pad_x, pad_y, shape, conf_min=0.25, iou_nms=0.45):
    """Returns a list of (x1, y1, x2, y2, score, class) in original image coordinates."""
    pred = output[0].T
    class_scores = pred[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(pred)), classes]
    keep = scores >= conf_min
    if not keep.any():
        return []
    pred, scores, classes = pred[keep], scores[keep], classes[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    offset = classes * 4096.0
    nms_boxes = np.stack([cx - bw / 2 + offset, cy - bh / 2 + offset, bw, bh], axis=1)
    idx = cv2.dnn.NMSBoxes(nms_boxes.tolist(), scores.tolist(), conf_min, iou_nms)
    idx = np.array(idx).flatten() if len(idx) else np.array([], dtype=int)
    height, width = shape
    x1 = ((cx - bw / 2 - pad_x) / r).clip(0, width)
    y1 = ((cy - bh / 2 - pad_y) / r).clip(0, height)
    x2 = ((cx + bw / 2 - pad_x) / r).clip(0, width)
    y2 = ((cy + bh / 2 - pad_y) / r).clip(0, height)
    return [(float(x1[k]), float(y1[k]), float(x2[k]), float(y2[k]), float(scores[k]), int(classes[k]))
            for k in idx]


class Perception:
    def __init__(self):
        self.seg = TrtEngine(C.ENGINE_SEG)
        self.det = TrtEngine(C.ENGINE_DET)
        self.last_mask = None

    def process(self, img):
        """Returns (walkable, boxes, times_ms).
        walkable: uint8 mask at model resolution (504x896 for 1080p)."""
        t = [time.perf_counter()]
        x, _, l, tp, nh, nw = letterbox(img, C.SEG_H, C.SEG_W)
        out = self.seg(x)
        mask = np.asarray(out).reshape(C.SEG_H, C.SEG_W).astype(np.uint8)[tp:tp + nh, l:l + nw]
        walkable = walkable_sidewalk(mask)
        self.last_mask = mask                 # raw class mask (used for poles in BEV mode)
        t.append(time.perf_counter())
        xd, rd, ld, td, _, _ = letterbox(img, C.DET_H, C.DET_W)
        boxes = postprocess_detection(self.det(xd), rd, ld, td, img.shape[:2])
        t.append(time.perf_counter())
        times = {"seg_ms": (t[1] - t[0]) * 1000, "det_ms": (t[2] - t[1]) * 1000}
        return walkable, boxes, times
