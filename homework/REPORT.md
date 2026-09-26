# Homework 1: Object Detection for a Real-World Application

**Project:** [You Are Fish](https://you-are-fish.vercel.app) (`you-are-fish`)  
**Detector:** YOLOv8n ONNX (`public/models/yolov8n.onnx`), class `cell phone` (COCO id 67)  
**Artifacts:** `[homework/test-images/](test-images/)`, `[homework/results/](results/)`, batch scripts in `[homework/scripts/](scripts/)`

---

## Task 1 — Application and Motivation

### What is the application?

**You Are Fish** is a browser webcam experience: the user’s face is tracked and mapped onto a fish that swims according to head pose. Separately, a pretrained object detector watches each camera frame for a **cell phone**.

### What problem does it attempt to solve?

It discourages phone use during the session. When a phone is stably visible in the camera, the fish flees and scolds the user; when the phone is put away, the fish returns.

### What object class is important?

COCO `cell phone` (class ID **67**).

### Why is object detection useful?

The app needs to know **whether** a phone is in view (and, for debugging, **where**). Classification alone would only yield a whole-image label; detection supplies boxes and scores that drive presence logic and visualization.

### What information does each detection provide?

For every kept proposal: axis-aligned box `{x, y, width, height}`, `classId`, `className` (`"cell phone"`), and `score` (see `lib/object-detection/phone-decode.ts`).

### How does a phone detection affect the interactive experience?

1. Post-NMS detections → `rawPhone = detections.length > 0`
2. `PhonePresenceTracker` converts raw detections into stable `phonePresent` (400 ms to activate, 600 ms to deactivate)
3. When `phonePresent` becomes true during play: fish **fleeing**
4. When the fish is mostly off-screen: **hidden** + message **"Put that away. You're a fish."**
5. When the phone is gone: **returning** + **"That's better."** (1.5 s), then **normal**



### Implemented vs planned


| Implemented now                                                                   | Only planned (README “LATER”, not part of this homework detector) |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Client YOLO phone detect, NMS, debug boxes, presence hysteresis, flee/hide/return | Cheek puff, hand gestures, fluid ocean, Fishsona persistence      |


---



## Task 2 — Test Images

**40 images** in `[homework/test-images/](test-images/)` (within 20–50). Provenance: `[test-images/README.md](test-images/README.md)`, `[test-images-manifest.json](test-images-manifest.json)`.


| Subset                      | Count | Source                                                 |
| --------------------------- | ----- | ------------------------------------------------------ |
| COCO128 phone originals     | 5     | Ultralytics COCO128 (YOLO class 67 labeled)            |
| COCO128 phone augmentations | 8     | Scale / brightness / crop variants                     |
| Wikimedia Commons phones    | 18    | Public Commons search (smartphone / in-hand / product) |
| COCO128 negatives           | 9     | No class-67 label                                      |


**Variety:** viewpoint (crops, hand-held), scale (downscales + product close-ups), lighting (bright/dark), occlusion risk (in-hand), cluttered COCO backgrounds, and negatives without phones.

---



## Task 3 — Pretrained Object Detector


| Detail                           | Value                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------- |
| Model                            | **YOLOv8n** (nano)                                                                            |
| File                             | `public/models/yolov8n.onnx`                                                                  |
| Live runtime                     | `onnxruntime-web` WASM in a Web Worker (`lib/object-detection/phone-detector.worker.ts`)      |
| Batch runtime (this homework)    | `onnxruntime` CPU (`homework/scripts/batch_phone_detect.py`), same weights + decode/NMS logic |
| Training data                    | Not documented for this ONNX file; code assumes **COCO-80** layout                            |
| Target class                     | `"cell phone"`, ID **67**                                                                     |
| Input                            | **640×640** letterbox, pad `rgb(114,114,114)`, RGB CHW floats in **[0, 1]**                   |
| Raw output                       | `[1, 84, N]` — cx,cy,w,h + 80 class scores                                                    |
| Decode                           | Keep phone channel scores ≥ threshold; map boxes to source; drop tiny boxes                   |
| Live confidence                  | **0.15** (`PHONE_SCORE_THRESHOLD`)                                                            |
| Live NMS IoU                     | **0.45**; `nmsEnabled` default **true** (can disable via config)                              |
| Where                            | **Client-side** (webcam); batch eval is offline on static files                               |
| Cadence (live)                   | ≥ **160 ms** start-to-start; HUD shows live `Infer ms`                                        |
| Batch mean infer (measured, CPU) | ~**100–110 ms**/image on the machine that ran experiments                                     |


---



## Task 4 — Process Multiple Images / Detection Results

Batch processing is implemented in `homework/scripts/batch_phone_detect.py`.

For **every** test image, each JSON result records:

- object detection (YOLO forward pass)
- bounding boxes
- object class (`cell phone` / 67)
- confidence score(s)
- `phoneCount` (number of post-NMS phones)

Primary multi-image run at live-like settings: `[results/live_defaults.json](results/live_defaults.json)`  
Annotated images: `[results/annotated/](results/annotated/)`

Webcam path also shows boxes/class/score in the debug overlay (`drawPhoneDetections`); counts appear as Before/After NMS in the HUD.

---



## Task 5 — Confidence Threshold Experiment

**Live app threshold:** `0.15`  
**Experiment thresholds run:** **0.20, 0.50, 0.80** (assignment-suggested values)  
**Results saved:** `[results/confidence_experiment.json](results/confidence_experiment.json)`, per-threshold JSON + annotated images, summary in `[results/EXPERIMENT_SUMMARY.md](results/EXPERIMENT_SUMMARY.md)`

Measured on the 40-image set (NMS on, IoU 0.45):


| Threshold | Images with ≥1 phone | Total phones | Det rate on expected-phone images | FP rate on negatives |
| --------- | -------------------- | ------------ | --------------------------------- | -------------------- |
| 0.20      | 14                   | 44           | 0.452                             | 0.0                  |
| 0.50      | 11                   | 32           | 0.355                             | 0.0                  |
| 0.80      | 5                    | 13           | 0.161                             | 0.0                  |


**Observation (from measured data only):** Raising the threshold reduces both the number of images with detections and total phone boxes. Negatives stayed at **0** false positives across these three thresholds on this set. Many COCO128 “small phone in clutter” cases fall below 0.20–0.50 and are counted as misses—so higher thresholds trade recall for stricter confidence.

---



## Task 6 — Non-Maximum Suppression Experiment



### Implementation (app)

- **IoU:** axis-aligned intersection / union (`lib/object-detection/nms.ts` → `iou`)
- **Greedy NMS:** sort by score descending; keep best; suppress others with IoU ≥ threshold; cap at 20
- **Live IoU:** 0.45; **can disable** with `nmsEnabled: false`
- **Pre-NMS** candidates and **post-NMS** detections are both retained (`candidates` / `detections`)



### Experiment (measured)

Score fixed at **0.20**; configs: NMS on IoU 0.45 / 0.20 / 0.80, and **NMS off**.  
Saved: `[results/nms_experiment.json](results/nms_experiment.json)`


| Config           | Mean before NMS | Mean after NMS | Total phones |
| ---------------- | --------------- | -------------- | ------------ |
| NMS on, IoU 0.45 | 9.95            | 1.10           | 44           |
| **NMS off**      | 9.95            | 3.58*          | **143**      |
| NMS on, IoU 0.20 | 9.95            | 1.07           | 43           |
| NMS on, IoU 0.80 | 9.95            | 1.12           | 45           |


When NMS is off, `afterNms` is still capped at `maxDetections=20` per image (same as the app), so mean after ≠ mean before on busy frames.

**Observation:** Disabling NMS multiplies duplicate boxes on the same phones (143 vs 44 total). Changing IoU among 0.20 / 0.45 / 0.80 barely changed image-level TP/FN on this set, but NMS-off clearly shows duplicate proposals—exactly why NMS is needed for counting/presence.

---



## Task 7 — Successful and Failed Detection Examples

Selected from **measured** `conf_0.20` outputs. Images in `[results/examples/](results/examples/)`; metadata in `[results/examples.json](results/examples.json)`.

### Successful (2)

1. `commons_08.jpg` — single box, bestScore ≈ **0.92**
  Annotated: `results/examples/success__commons_08.jpg`
2. `commons_06.png` — single box, bestScore ≈ **0.87**
  Annotated: `results/examples/success__commons_06.jpg`



### Partially successful (2)

1. `commons_09.png` — detected, but low score (~**0.26**) and two overlapping-ish boxes
  `results/examples/partial__commons_09.jpg`
2. `commons_14.jpg` — hand-held / occlusion-risk scene; one weak detection (~**0.26**)
  `results/examples/partial__commons_14.jpg`



### Failures (2)

1. `coco128_phone_000000000395.jpg` — labeled phone in COCO128, but bestScore ≈ **0.199** < 0.20 → **miss**
  `results/examples/failure__coco128_phone_000000000395.jpg`
2. `coco128_phone_000000000328.jpg` — labeled phone, bestScore ≈ **0.074** → **miss**
  `results/examples/failure__coco128_phone_000000000328.jpg`

---



## Task 8 — Application-Level Analysis



### Pipeline

```
webcam canvas
  → letterbox 640 + RGB/255
  → YOLOv8n ONNX (WASM worker)
  → decode cell-phone scores
  → confidence filter
  → NMS
  → rawPhone = (detections.length > 0)
  → PhonePresenceTracker (400 ms / 600 ms)
  → phonePresent → fish flee / hide / return + messages
```



### Stable `phonePresent`

`PhonePresenceTracker` (`lib/phone-presence.ts`):

- raw true must accumulate **400 ms** before present
- raw false must accumulate **600 ms** before absent
- per-update `dt` clamped to **100 ms** (avoids huge jumps after tab pauses)



### When a phone becomes present

Fish starts **fleeing** (nearest side). No scolding text yet.

### When it disappears

If fleeing or hidden → **returning** with **"That's better."** for **1500 ms**, then normal steering resumes once on-screen.

### Messages


| State                                | Message                         |
| ------------------------------------ | ------------------------------- |
| Fish mostly off-screen while fleeing | `Put that away. You're a fish.` |
| Phone cleared after flee/hide        | `That's better.` (1.5 s)        |


---



## Conclusion

This homework wired a **pretrained YOLOv8n** phone detector into a real interactive webcam app and evaluated it on a **40-image** static set.

- Detection is useful here because **presence of a phone in the frame** should change gameplay, not merely classify the whole session.
- **Confidence threshold** strongly controls recall: 0.20 → 0.50 → 0.80 steadily reduced detections on this set.
- **NMS** is essential to collapse duplicate proposals; disabling it roughly **tripled** total phone boxes.
- Clear product-style phones detect well; **small/cluttered phones** often miss—important for a selfie-webcam product where phones may be small or partly occluded.
- Application success also depends on **temporal filtering** and UX (flee / messages), not boxes alone.

**Deliverable media:** annotated batch outputs under `[homework/results/annotated/](results/annotated/)` and curated examples under `[homework/results/examples/](results/examples/)`. Live demo: enable camera on the deployed app and open Tracking Info (ⓘ) with `DEBUG=true` to see boxes and infer timing.