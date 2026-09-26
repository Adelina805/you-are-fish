#!/usr/bin/env python3
"""Run Homework 1 confidence + NMS experiments and write measured summaries."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from batch_phone_detect import ROOT, run_batch

RESULTS = ROOT / "homework" / "results"
IMAGES = ROOT / "homework" / "test-images"
MODEL = ROOT / "public" / "models" / "yolov8n.onnx"
MANIFEST = ROOT / "homework" / "test-images-manifest.json"


def load_manifest() -> dict[str, dict]:
    rows = json.loads(MANIFEST.read_text())
    return {r["filename"]: r for r in rows}


def summarize_vs_expected(summary: dict, manifest: dict[str, dict]) -> dict:
    tp = fp = tn = fn = 0
    per_image = []
    for r in summary["results"]:
        expected = bool(manifest.get(r["image"], {}).get("expected_phone", False))
        predicted = r["phoneCount"] > 0
        if expected and predicted:
            tp += 1
            label = "TP"
        elif expected and not predicted:
            fn += 1
            label = "FN"
        elif not expected and predicted:
            fp += 1
            label = "FP"
        else:
            tn += 1
            label = "TN"
        per_image.append(
            {
                "image": r["image"],
                "expected_phone": expected,
                "phoneCount": r["phoneCount"],
                "bestScore": r["bestScore"],
                "beforeNms": r["beforeNms"],
                "afterNms": r["afterNms"],
                "label": label,
            }
        )
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "detection_rate_on_expected": round(tp / (tp + fn), 3) if (tp + fn) else None,
        "false_positive_rate_on_negatives": round(fp / (fp + tn), 3)
        if (fp + tn)
        else None,
        "per_image": per_image,
    }


def run_confidence(manifest: dict[str, dict]) -> list[dict]:
    thresholds = [0.20, 0.50, 0.80]
    rows = []
    for thr in thresholds:
        tag = f"conf_{thr:.2f}"
        summary = run_batch(
            images_dir=IMAGES,
            out_dir=RESULTS,
            model_path=MODEL,
            score_threshold=thr,
            iou_threshold=0.45,
            nms_enabled=True,
            tag=tag,
            save_images=True,
        )
        metrics = summarize_vs_expected(summary, manifest)
        (RESULTS / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
        rows.append(
            {
                "scoreThreshold": thr,
                "tag": tag,
                "imagesWithPhones": summary["imagesWithPhones"],
                "totalPhonesDetected": summary["totalPhonesDetected"],
                "meanInferMs": summary["meanInferMs"],
                **{k: metrics[k] for k in ("tp", "fp", "tn", "fn", "detection_rate_on_expected", "false_positive_rate_on_negatives")},
            }
        )
    (RESULTS / "confidence_experiment.json").write_text(json.dumps(rows, indent=2))
    return rows


def run_nms(manifest: dict[str, dict]) -> list[dict]:
    configs = [
        {"tag": "nms_on_iou0.45", "nms": True, "iou": 0.45},
        {"tag": "nms_off", "nms": False, "iou": 0.45},
        {"tag": "nms_on_iou0.20", "nms": True, "iou": 0.20},
        {"tag": "nms_on_iou0.80", "nms": True, "iou": 0.80},
    ]
    rows = []
    for cfg in configs:
        summary = run_batch(
            images_dir=IMAGES,
            out_dir=RESULTS,
            model_path=MODEL,
            score_threshold=0.20,  # homework-suggested mid-low threshold for NMS study
            iou_threshold=cfg["iou"],
            nms_enabled=cfg["nms"],
            tag=cfg["tag"],
            save_images=True,
        )
        metrics = summarize_vs_expected(summary, manifest)
        mean_before = sum(r["beforeNms"] for r in summary["results"]) / max(
            1, len(summary["results"])
        )
        mean_after = sum(r["afterNms"] for r in summary["results"]) / max(
            1, len(summary["results"])
        )
        row = {
            "tag": cfg["tag"],
            "nmsEnabled": cfg["nms"],
            "iouThreshold": cfg["iou"],
            "scoreThreshold": 0.20,
            "meanBeforeNms": round(mean_before, 2),
            "meanAfterNms": round(mean_after, 2),
            "totalPhonesDetected": summary["totalPhonesDetected"],
            "imagesWithPhones": summary["imagesWithPhones"],
            **{k: metrics[k] for k in ("tp", "fp", "tn", "fn")},
        }
        rows.append(row)
        (RESULTS / f"{cfg['tag']}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / "nms_experiment.json").write_text(json.dumps(rows, indent=2))
    return rows


def pick_examples(manifest: dict[str, dict]) -> dict:
    """
    Curate Task 7 examples from measured conf_0.20 results (not invented).
    Categories use detector output vs expected_phone + qualitative notes.
    """
    conf_path = RESULTS / "conf_0.20.json"
    data = json.loads(conf_path.read_text())
    by_name = {r["image"]: r for r in data["results"]}

    def rank_success():
        # Prefer a single high-confidence box on an expected-phone image.
        cands = []
        for name, r in by_name.items():
            m = manifest.get(name, {})
            if m.get("expected_phone") and r["phoneCount"] == 1 and r["bestScore"] >= 0.5:
                cands.append((r["bestScore"], name, r))
        cands.sort(reverse=True)
        if len(cands) < 2:
            # Fall back to any strong detection if too few singles.
            for name, r in by_name.items():
                m = manifest.get(name, {})
                if m.get("expected_phone") and r["phoneCount"] >= 1 and r["bestScore"] >= 0.5:
                    if not any(name == c[1] for c in cands):
                        cands.append((r["bestScore"], name, r))
            cands.sort(reverse=True)
        return cands[:2]

    def rank_partial():
        # detected but low-ish score, or multiple boxes, or augmented hard case
        cands = []
        for name, r in by_name.items():
            m = manifest.get(name, {})
            if not m.get("expected_phone"):
                continue
            if r["phoneCount"] >= 1 and (
                r["bestScore"] < 0.55
                or r["phoneCount"] > 1
                or "scale_small" in name
                or "dark" in name
                or "crop" in name
            ):
                cands.append((r["bestScore"], name, r, m))
        # prefer harder ones (lower score) but still detected
        cands.sort(key=lambda t: t[0])
        return cands[:2]

    def rank_failure():
        failures = []
        # FN: expected but none detected
        for name, r in by_name.items():
            m = manifest.get(name, {})
            if m.get("expected_phone") and r["phoneCount"] == 0:
                failures.append(("fn", r["bestScore"], name, r, m))
        # FP: not expected but detected
        for name, r in by_name.items():
            m = manifest.get(name, {})
            if not m.get("expected_phone") and r["phoneCount"] > 0:
                failures.append(("fp", r["bestScore"], name, r, m))
        # sort FN by highest near-miss score, FP by score
        fns = sorted([f for f in failures if f[0] == "fn"], key=lambda t: t[1], reverse=True)
        fps = sorted([f for f in failures if f[0] == "fp"], key=lambda t: t[1], reverse=True)
        picked = (fns[:1] + fps[:1]) if fns or fps else failures[:2]
        if len(picked) < 2:
            # pad with additional FNs/FPs
            rest = [f for f in failures if f not in picked]
            picked += rest[: 2 - len(picked)]
        return picked[:2]

    examples_dir = RESULTS / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = RESULTS / "annotated"

    def copy_ann(name: str, label: str) -> str | None:
        src = ann_dir / f"{Path(name).stem}__conf_0.20.jpg"
        if not src.exists():
            # try live defaults style tag variants
            matches = list(ann_dir.glob(f"{Path(name).stem}__conf_0.20.jpg"))
            if not matches:
                return None
            src = matches[0]
        dest_name = f"{label}__{name}"
        if dest_name.lower().endswith(".png"):
            dest_name = dest_name[:-4] + ".jpg"
        dest = examples_dir / dest_name
        shutil.copy2(src, dest)
        return str(dest.relative_to(ROOT))

    success = []
    for best, name, r in rank_success():
        path = copy_ann(name, "success")
        success.append(
            {
                "image": name,
                "category": "successful",
                "phoneCount": r["phoneCount"],
                "bestScore": r["bestScore"],
                "detections": r["detections"],
                "annotated": path,
                "why": (
                    f"Expected phone present; detector returned {r['phoneCount']} "
                    f"box(es) with bestScore={r['bestScore']:.3f} at conf=0.20."
                ),
            }
        )

    partial = []
    for best, name, r, m in rank_partial():
        path = copy_ann(name, "partial")
        tags = ",".join(m.get("variety_tags", [])[:4])
        partial.append(
            {
                "image": name,
                "category": "partially_successful",
                "phoneCount": r["phoneCount"],
                "bestScore": r["bestScore"],
                "detections": r["detections"],
                "annotated": path,
                "why": (
                    f"Phone found (count={r['phoneCount']}, bestScore={r['bestScore']:.3f}) "
                    f"but under harder conditions ({tags or 'low score / multi-box'})."
                ),
            }
        )

    failure = []
    for kind, best, name, r, m in rank_failure():
        path = copy_ann(name, "failure")
        if kind == "fn":
            why = (
                f"False negative: expected phone but phoneCount=0 "
                f"(bestScore={r['bestScore']:.3f} stayed below threshold)."
            )
        else:
            why = (
                f"False positive: no phone expected but phoneCount={r['phoneCount']} "
                f"(bestScore={r['bestScore']:.3f})."
            )
        failure.append(
            {
                "image": name,
                "category": "failure",
                "failureType": kind,
                "phoneCount": r["phoneCount"],
                "bestScore": r["bestScore"],
                "detections": r["detections"],
                "annotated": path,
                "why": why,
            }
        )

    payload = {"confidenceUsed": 0.20, "success": success, "partial": partial, "failure": failure}
    (RESULTS / "examples.json").write_text(json.dumps(payload, indent=2))
    return payload


def write_summary_md(conf_rows: list[dict], nms_rows: list[dict], examples: dict) -> None:
    lines = [
        "# Experiment summary (measured)",
        "",
        "Generated by `homework/scripts/run_experiments.py` using `public/models/yolov8n.onnx`",
        "on `homework/test-images/` (40 images). Metrics vs `expected_phone` in the manifest",
        "are a coarse proxy (manifest intent ≠ exhaustive human boxes).",
        "",
        "## Confidence threshold experiment",
        "",
        "| Threshold | Images with ≥1 phone | Total phones | TP | FN | FP | TN | Det rate (expected) | FP rate (negatives) | Mean infer ms |",
        "|----------:|---------------------:|-------------:|---:|---:|---:|---:|--------------------:|--------------------:|--------------:|",
    ]
    for r in conf_rows:
        lines.append(
            f"| {r['scoreThreshold']:.2f} | {r['imagesWithPhones']} | {r['totalPhonesDetected']} | "
            f"{r['tp']} | {r['fn']} | {r['fp']} | {r['tn']} | {r['detection_rate_on_expected']} | "
            f"{r['false_positive_rate_on_negatives']} | {r['meanInferMs']} |"
        )
    lines += [
        "",
        "## NMS experiment (score threshold fixed at 0.20)",
        "",
        "| Config | NMS | IoU | Mean before NMS | Mean after NMS | Total phones | Images with ≥1 | TP | FN | FP | TN |",
        "|--------|:---:|----:|----------------:|---------------:|-------------:|---------------:|---:|---:|---:|---:|",
    ]
    for r in nms_rows:
        lines.append(
            f"| {r['tag']} | {'on' if r['nmsEnabled'] else 'off'} | {r['iouThreshold']:.2f} | "
            f"{r['meanBeforeNms']} | {r['meanAfterNms']} | {r['totalPhonesDetected']} | "
            f"{r['imagesWithPhones']} | {r['tp']} | {r['fn']} | {r['fp']} | {r['tn']} |"
        )
    lines += [
        "",
        "## Example cases (conf=0.20)",
        "",
        "### Successful",
    ]
    for e in examples["success"]:
        lines.append(f"- `{e['image']}` — {e['why']} — annotated: `{e['annotated']}`")
    lines += ["", "### Partially successful"]
    for e in examples["partial"]:
        lines.append(f"- `{e['image']}` — {e['why']} — annotated: `{e['annotated']}`")
    lines += ["", "### Failures"]
    for e in examples["failure"]:
        lines.append(f"- `{e['image']}` — {e['why']} — annotated: `{e['annotated']}`")
    lines.append("")
    (RESULTS / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--live-defaults", action="store_true")
    args = parser.parse_args()

    if not any([args.confidence, args.nms, args.examples, args.all, args.live_defaults]):
        args.all = True

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    conf_rows: list[dict] = []
    nms_rows: list[dict] = []

    if args.all or args.live_defaults:
        run_batch(
            images_dir=IMAGES,
            out_dir=RESULTS,
            model_path=MODEL,
            score_threshold=0.15,
            iou_threshold=0.45,
            nms_enabled=True,
            tag="live_defaults",
            save_images=True,
        )

    if args.all or args.confidence:
        conf_rows = run_confidence(manifest)

    if args.all or args.nms:
        nms_rows = run_nms(manifest)

    examples = {}
    if args.all or args.examples:
        # examples need conf_0.20.json
        if not (RESULTS / "conf_0.20.json").exists():
            conf_rows = run_confidence(manifest)
        examples = pick_examples(manifest)

    if conf_rows or (RESULTS / "confidence_experiment.json").exists():
        if not conf_rows and (RESULTS / "confidence_experiment.json").exists():
            conf_rows = json.loads((RESULTS / "confidence_experiment.json").read_text())
    if nms_rows or (RESULTS / "nms_experiment.json").exists():
        if not nms_rows and (RESULTS / "nms_experiment.json").exists():
            nms_rows = json.loads((RESULTS / "nms_experiment.json").read_text())
    if examples or (RESULTS / "examples.json").exists():
        if not examples and (RESULTS / "examples.json").exists():
            examples = json.loads((RESULTS / "examples.json").read_text())

    if conf_rows and nms_rows and examples:
        write_summary_md(conf_rows, nms_rows, examples)
        print("wrote", RESULTS / "EXPERIMENT_SUMMARY.md")


if __name__ == "__main__":
    main()
