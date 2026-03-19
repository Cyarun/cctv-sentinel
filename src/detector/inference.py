#!/usr/bin/env python3
"""ONNX inference helper for cctv-sentinel YOLO detector.

Usage:
    python3 inference.py <raw_rgb_path> <model_path> <img_width> <img_height> <model_size> <conf_threshold>

Reads a raw RGB image file, runs ONNX inference, and prints JSON detections
to stdout. Each detection has: class, class_id, confidence, x, y, w, h
(all coordinates normalized 0.0-1.0, center-based).

Exit codes:
    0 = success (even if no detections — prints empty array [])
    1 = error (details on stderr)
"""

import sys
import json
import numpy as np

CLASSES = [
    'person', 'car', 'motorcycle', 'auto_rickshaw', 'dog',
    'cat', 'cow', 'bicycle', 'truck', 'bus',
]


def load_raw_rgb(path, width, height):
    """Load raw RGB bytes into a numpy array (H, W, 3)."""
    expected = width * height * 3
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) != expected:
        raise ValueError(
            f'Expected {expected} bytes, got {len(data)} '
            f'for {width}x{height} RGB image'
        )
    return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))


def preprocess(img, model_size):
    """Preprocess image for YOLO ONNX model.

    - Resize to model_size x model_size (letterbox not needed for square CIF)
    - Normalize to 0.0-1.0 float32
    - Transpose to NCHW format
    """
    from PIL import Image

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((model_size, model_size), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    # HWC -> CHW -> NCHW
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def postprocess(output, conf_threshold, num_classes):
    """Parse YOLO output tensor into detections.

    Supports two common YOLO output shapes:
    - (1, num_detections, 4+num_classes) — YOLOv8 format
    - (1, 4+num_classes, num_detections) — transposed format
    """
    pred = output[0]  # first output tensor

    if len(pred.shape) == 3:
        pred = pred[0]  # remove batch dim -> (rows, cols)

    # Determine orientation: if cols == 4+num_classes, rows are detections
    # If rows == 4+num_classes, need to transpose
    expected_attrs = 4 + num_classes
    if pred.shape[0] == expected_attrs and pred.shape[1] != expected_attrs:
        pred = pred.T  # now (num_detections, 4+num_classes)

    detections = []
    for row in pred:
        # row: [cx, cy, w, h, cls0_conf, cls1_conf, ...]
        box = row[:4]
        class_scores = row[4:4 + num_classes]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < conf_threshold:
            continue

        cx, cy, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        detections.append({
            'class': CLASSES[class_id] if class_id < len(CLASSES) else f'class_{class_id}',
            'class_id': class_id,
            'confidence': round(confidence, 4),
            'x': round(cx, 4),
            'y': round(cy, 4),
            'w': round(bw, 4),
            'h': round(bh, 4),
        })

    # NMS — simple greedy IoU-based
    detections = nms(detections, iou_threshold=0.5)
    return detections


def iou(a, b):
    """Compute IoU between two center-format boxes."""
    ax1 = a['x'] - a['w'] / 2
    ay1 = a['y'] - a['h'] / 2
    ax2 = a['x'] + a['w'] / 2
    ay2 = a['y'] + a['h'] / 2

    bx1 = b['x'] - b['w'] / 2
    by1 = b['y'] - b['h'] / 2
    bx2 = b['x'] + b['w'] / 2
    by2 = b['y'] + b['h'] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


def nms(detections, iou_threshold=0.5):
    """Greedy NMS: keep highest confidence, suppress overlapping same-class."""
    if not detections:
        return []

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    keep = []

    for det in detections:
        suppressed = False
        for kept in keep:
            if kept['class_id'] == det['class_id'] and iou(kept, det) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)

    return keep


def main():
    if len(sys.argv) != 7:
        print(
            'Usage: inference.py <raw_rgb_path> <model_path> '
            '<width> <height> <model_size> <conf_threshold>',
            file=sys.stderr,
        )
        sys.exit(1)

    raw_path = sys.argv[1]
    model_path = sys.argv[2]
    img_width = int(sys.argv[3])
    img_height = int(sys.argv[4])
    model_size = int(sys.argv[5])
    conf_threshold = float(sys.argv[6])

    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed. pip install onnxruntime', file=sys.stderr)
        sys.exit(1)

    # Load image
    img = load_raw_rgb(raw_path, img_width, img_height)

    # Preprocess
    input_tensor = preprocess(img, model_size)

    # Run inference
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider'],
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    # Postprocess
    detections = postprocess(outputs, conf_threshold, len(CLASSES))

    # Normalize coordinates to 0.0-1.0 (model outputs are in model_size pixels)
    for det in detections:
        det['x'] = round(det['x'] / model_size, 4)
        det['y'] = round(det['y'] / model_size, 4)
        det['w'] = round(det['w'] / model_size, 4)
        det['h'] = round(det['h'] / model_size, 4)

    # Output JSON to stdout
    print(json.dumps(detections))


if __name__ == '__main__':
    main()
