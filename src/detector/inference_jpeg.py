#!/usr/bin/env python3
"""ONNX inference helper that accepts JPEG input directly.

Usage:
    python3 inference_jpeg.py <jpeg_path> <model_path> <model_size> <conf_threshold>

Reads a JPEG image, runs ONNX YOLO inference, prints JSON detections to stdout.
Coordinates are normalized 0.0-1.0 (center x, center y, width, height).

Exit codes:
    0 = success (prints JSON array, possibly empty)
    1 = error (details on stderr)
"""

import sys
import json
import numpy as np

CLASSES = [
    'person', 'car', 'motorcycle', 'auto_rickshaw', 'dog',
    'cat', 'cow', 'bicycle', 'truck', 'bus',
]


def preprocess(img, model_size):
    """Resize, normalize, transpose to NCHW float32."""
    from PIL import Image

    pil_img = img.resize((model_size, model_size), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def postprocess(output, conf_threshold, num_classes):
    """Parse YOLO output tensor into detections."""
    pred = output[0]
    if len(pred.shape) == 3:
        pred = pred[0]

    expected_attrs = 4 + num_classes
    if pred.shape[0] == expected_attrs and pred.shape[1] != expected_attrs:
        pred = pred.T

    detections = []
    for row in pred:
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

    detections = nms(detections, iou_threshold=0.5)
    return detections


def iou(a, b):
    ax1, ay1 = a['x'] - a['w'] / 2, a['y'] - a['h'] / 2
    ax2, ay2 = a['x'] + a['w'] / 2, a['y'] + a['h'] / 2
    bx1, by1 = b['x'] - b['w'] / 2, b['y'] - b['h'] / 2
    bx2, by2 = b['x'] + b['w'] / 2, b['y'] + b['h'] / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0


def nms(detections, iou_threshold=0.5):
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
    if len(sys.argv) != 5:
        print(
            'Usage: inference_jpeg.py <jpeg_path> <model_path> <model_size> <conf_threshold>',
            file=sys.stderr,
        )
        sys.exit(1)

    jpeg_path = sys.argv[1]
    model_path = sys.argv[2]
    model_size = int(sys.argv[3])
    conf_threshold = float(sys.argv[4])

    try:
        from PIL import Image
    except ImportError:
        print('Pillow not installed. pip install Pillow', file=sys.stderr)
        sys.exit(1)

    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed. pip install onnxruntime', file=sys.stderr)
        sys.exit(1)

    # Load JPEG image
    try:
        img = Image.open(jpeg_path).convert('RGB')
    except Exception as e:
        print(f'Failed to open image: {e}', file=sys.stderr)
        sys.exit(1)

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

    # Normalize coordinates to 0.0-1.0
    for det in detections:
        det['x'] = round(det['x'] / model_size, 4)
        det['y'] = round(det['y'] / model_size, 4)
        det['w'] = round(det['w'] / model_size, 4)
        det['h'] = round(det['h'] / model_size, 4)

    print(json.dumps(detections))


if __name__ == '__main__':
    main()
