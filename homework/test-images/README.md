# Homework 1 test images (40)

Static evaluation set for phone object detection. Count is within the assignment’s 20–50 range.

## Provenance

| Subset | Count | Source |
|--------|------:|--------|
| `coco128_phone_*.jpg` (original) | 5 | [Ultralytics COCO128](https://github.com/ultralytics/assets) subset of COCO; YOLO label class **67** (`cell phone`) present |
| `coco128_phone_*_{scale_small,bright,dark,crop}.jpg` | 8 | Deterministic augmentations of the COCO128 phone images (scale / lighting / center-crop) |
| `commons_*.jpg|png` | 18 | [Wikimedia Commons](https://commons.wikimedia.org) search hits for smartphone / mobile phone / hand-held use |
| `coco128_nophone_*.jpg` | 9 | COCO128 scenes with **no** class-67 label (negatives / distractors) |

Per-file URLs and tags: [`../test-images-manifest.json`](../test-images-manifest.json).

## Variety represented

- **Viewpoint / crop:** center-cropped COCO variants; Commons hand-held angles
- **Scale:** original vs `scale_small` downscales; product close-ups vs cluttered scenes
- **Lighting:** `bright` / `dark` augmentations; mixed Commons lighting
- **Occlusion / grasp:** Commons “in hand” searches (fingers may cover phone)
- **Backgrounds:** cluttered COCO multi-object scenes; product-style Commons shots; negatives without phones
- **Negatives:** 9 images expected to contain no cell phone (false-positive stress)

## Notes

- `expected_phone` in the manifest is based on source labels/search intent, **not** on detector output.
- Augmentations are documented derivatives, not additional independent photographs.
