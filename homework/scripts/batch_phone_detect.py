#!/usr/bin/env python3
"""
Batch phone detection for Homework 1.

Mirrors the browser pipeline in:
  lib/object-detection/phone-detector.ts
  lib/object-detection/phone-decode.ts
  lib/object-detection/nms.ts

Uses the same ONNX weights: public/models/yolov8n.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "public" / "models" / "yolov8n.onnx"
DEFAULT_IMAGES = ROOT / "homework" / "test-images"
DEFAULT_OUT = ROOT / "homework" / "results"

CELL_PHONE_CLASS_ID = 67
CLASS_NAME = "cell phone"
INPUT_SIZE = 640
LETTERBOX_PAD = (114, 114, 114)
MAX_DETECTIONS = 20


@dataclass
class Box:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Detection:
    box: Box
    classId: int
    className: str
    score: float


def letterbox(image: Image.Image, input_size: int = INPUT_SIZE) -> tuple[np.ndarray, dict[str, float]]:
    """Match PhoneDetector.letterboxSnapshot: scale, center pad with rgb(114,114,114)."""
    src_w, src_h = image.size
    scale = min(input_size / src_w, input_size / src_h)
    draw_w = int(round(src_w * scale))
    draw_h = int(round(src_h * scale))
    pad_x = (input_size - draw_w) / 2.0
    pad_y = (input_size - draw_h) / 2.0

    canvas = Image.new("RGB", (input_size, input_size), LETTERBOX_PAD)
    resized = image.convert("RGB").resize((draw_w, draw_h), Image.Resampling.BILINEAR)
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC RGB
    chw = np.transpose(arr, (2, 0, 1))  # CHW
    tensor = np.expand_dims(chw, 0)  # NCHW
    meta = {
        "scale": float(scale),
        "padX": float(pad_x),
        "padY": float(pad_y),
        "srcWidth": float(src_w),
        "srcHeight": float(src_h),
    }
    return tensor, meta


def letterbox_box_to_source(cx: float, cy: float, w: float, h: float, meta: dict[str, float]) -> Box:
    x1 = (cx - w / 2 - meta["padX"]) / meta["scale"]
    y1 = (cy - h / 2 - meta["padY"]) / meta["scale"]
    x2 = (cx + w / 2 - meta["padX"]) / meta["scale"]
    y2 = (cy + h / 2 - meta["padY"]) / meta["scale"]
    left = max(0.0, min(meta["srcWidth"], x1))
    top = max(0.0, min(meta["srcHeight"], y1))
    right = max(0.0, min(meta["srcWidth"], x2))
    bottom = max(0.0, min(meta["srcHeight"], y2))
    return Box(x=left, y=top, width=max(0.0, right - left), height=max(0.0, bottom - top))


def iou(a: Box, b: Box) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(
    candidates: list[Detection],
    *,
    iou_threshold: float,
    enabled: bool,
    max_detections: int,
) -> list[Detection]:
    """Match lib/object-detection/nms.ts greedy NMS."""
    if not enabled:
        return candidates[:max_detections] if len(candidates) > max_detections else list(candidates)

    sorted_cands = sorted(candidates, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    while sorted_cands:
        best = sorted_cands.pop(0)
        kept.append(best)
        if len(kept) >= max_detections:
            break
        sorted_cands = [d for d in sorted_cands if iou(best.box, d.box) < iou_threshold]
    return kept


def decode_phone_candidates(
    data: np.ndarray,
    dims: list[int],
    meta: dict[str, float],
    score_threshold: float,
) -> tuple[list[Detection], float]:
    """Match decodePhoneCandidates for YOLOv8 output [1, 84, N]."""
    if len(dims) == 3:
        num_preds = int(dims[2])
    elif len(dims) >= 2 and dims[1] == 84:
        num_preds = int(dims[2])
    else:
        raise ValueError(f"Unexpected YOLO output shape: {dims}")

    flat = np.asarray(data, dtype=np.float32).reshape(84, num_preds)
    class_channel = 4 + CELL_PHONE_CLASS_ID
    scores = flat[class_channel]
    best_score = float(scores.max()) if num_preds else 0.0

    candidates: list[Detection] = []
    for i in range(num_preds):
        score = float(scores[i])
        if score < score_threshold:
            continue
        cx, cy, w, h = map(float, flat[0:4, i])
        box = letterbox_box_to_source(cx, cy, w, h, meta)
        if box.width <= 1 or box.height <= 1:
            continue
        candidates.append(
            Detection(
                box=box,
                classId=CELL_PHONE_CLASS_ID,
                className=CLASS_NAME,
                score=score,
            )
        )
    return candidates, best_score


def detection_to_dict(d: Detection) -> dict[str, Any]:
    return {
        "box": asdict(d.box),
        "classId": d.classId,
        "className": d.className,
        "score": d.score,
    }


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    *,
    title: str,
) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in detections:
        x, y, w, h = d.box.x, d.box.y, d.box.width, d.box.height
        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=3)
        draw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=1)
        label = f"CELL PHONE {d.score:.2f}"
        ty = max(0, y - 14)
        draw.rectangle([x, ty, x + 8 * len(label), ty + 14], fill=(20, 20, 20))
        draw.text((x + 2, ty + 1), label, fill=(255, 255, 255), font=font)

    banner = f"{title} | phones={len(detections)}"
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    draw.text((4, 2), banner, fill=(255, 255, 255), font=font)
    return img


class PhoneBatchDetector:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(
        self,
        image: Image.Image,
        *,
        score_threshold: float,
        iou_threshold: float,
        nms_enabled: bool,
    ) -> dict[str, Any]:
        tensor, meta = letterbox(image)
        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        infer_ms = (time.perf_counter() - t0) * 1000.0
        out = outputs[0]
        dims = list(out.shape)
        candidates, best_score = decode_phone_candidates(
            out, dims, meta, score_threshold
        )
        detections = non_max_suppression(
            candidates,
            iou_threshold=iou_threshold,
            enabled=nms_enabled,
            max_detections=MAX_DETECTIONS,
        )
        return {
            "candidates": [detection_to_dict(d) for d in candidates],
            "detections": [detection_to_dict(d) for d in detections],
            "candidate_objs": candidates,
            "detection_objs": detections,
            "bestScore": best_score,
            "beforeNms": len(candidates),
            "afterNms": len(detections),
            "phoneCount": len(detections),
            "inferMs": infer_ms,
            "meta": meta,
        }


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def run_batch(
    *,
    images_dir: Path,
    out_dir: Path,
    model_path: Path,
    score_threshold: float,
    iou_threshold: float,
    nms_enabled: bool,
    tag: str,
    save_images: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = out_dir / "annotated"
    if save_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    detector = PhoneBatchDetector(model_path)
    results: list[dict[str, Any]] = []
    title = (
        f"conf={score_threshold:.2f} iou={iou_threshold:.2f} "
        f"nms={'on' if nms_enabled else 'off'}"
    )

    for path in list_images(images_dir):
        image = Image.open(path)
        raw = detector.detect(
            image,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            nms_enabled=nms_enabled,
        )
        entry = {
            "image": path.name,
            "scoreThreshold": score_threshold,
            "iouThreshold": iou_threshold,
            "nmsEnabled": nms_enabled,
            "bestScore": raw["bestScore"],
            "beforeNms": raw["beforeNms"],
            "afterNms": raw["afterNms"],
            "phoneCount": raw["phoneCount"],
            "inferMs": round(raw["inferMs"], 2),
            "detections": raw["detections"],
            "candidates": raw["candidates"],
        }
        results.append(entry)

        if save_images:
            drawn = draw_detections(
                image, raw["detection_objs"], title=f"{path.name} | {title}"
            )
            drawn.save(annotated_dir / f"{path.stem}__{tag}.jpg", quality=90)

        print(
            f"[{tag}] {path.name}: phones={entry['phoneCount']} "
            f"beforeNMS={entry['beforeNms']} best={entry['bestScore']:.3f} "
            f"infer={entry['inferMs']:.0f}ms"
        )

    summary = {
        "tag": tag,
        "model": str(model_path.relative_to(ROOT)),
        "imagesDir": str(images_dir.relative_to(ROOT)),
        "scoreThreshold": score_threshold,
        "iouThreshold": iou_threshold,
        "nmsEnabled": nms_enabled,
        "imageCount": len(results),
        "imagesWithPhones": sum(1 for r in results if r["phoneCount"] > 0),
        "totalPhonesDetected": sum(r["phoneCount"] for r in results),
        "meanInferMs": round(
            float(np.mean([r["inferMs"] for r in results])) if results else 0.0, 2
        ),
        "results": results,
    }
    out_json = out_dir / f"{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch YOLOv8n phone detection")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--score", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--nms", choices=["on", "off"], default="on")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    nms_enabled = args.nms == "on"
    tag = args.tag or (
        f"score{args.score:.2f}_iou{args.iou:.2f}_nms{'on' if nms_enabled else 'off'}"
    )
    run_batch(
        images_dir=args.images,
        out_dir=args.out,
        model_path=args.model,
        score_threshold=args.score,
        iou_threshold=args.iou,
        nms_enabled=nms_enabled,
        tag=tag,
        save_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
