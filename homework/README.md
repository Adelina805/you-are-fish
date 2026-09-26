# Homework 1 — Object Detection experiments

**Final write-up:** [`REPORT.md`](REPORT.md)

Python batch runner mirroring the live browser phone detector
(`public/models/yolov8n.onnx` + the same letterbox / decode / NMS logic).

## Setup

Requires Python 3 with:

```bash
pip3 install onnxruntime pillow numpy
```

(Already available on the machine used to generate committed results.)

## Run default live-app settings (conf 0.15, IoU 0.45, NMS on)

```bash
python3 homework/scripts/batch_phone_detect.py \
  --score 0.15 --iou 0.45 --nms on --tag live_defaults
```

## Confidence threshold experiment (Task 5)

```bash
python3 homework/scripts/run_experiments.py --confidence
```

## NMS experiment (Task 6)

```bash
python3 homework/scripts/run_experiments.py --nms
```

## Outputs

Written under [`../results/`](../results/):

- `*.json` — per-image boxes, class, score, phone counts
- `annotated/` — images with drawn boxes
- `examples/` — curated success / partial / failure cases
- `EXPERIMENT_SUMMARY.md` — measured tables (generated, not invented)
