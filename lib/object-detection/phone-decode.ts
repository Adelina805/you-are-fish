import { CELL_PHONE_CLASS_ID, cocoLabel } from "@/lib/object-detection/coco-labels";
import { nonMaxSuppression } from "@/lib/object-detection/nms";

export type Detection = {
  box: { x: number; y: number; width: number; height: number };
  classId: number;
  className: string;
  score: number;
};

export type LetterboxMeta = {
  scale: number;
  padX: number;
  padY: number;
  srcWidth: number;
  srcHeight: number;
};

export type DecodePhoneOptions = {
  scoreThreshold: number;
  iouThreshold: number;
  nmsEnabled: boolean;
  maxDetections: number;
};

export type DecodePhoneResult = {
  /** Phone-class boxes above score threshold, before NMS. */
  candidates: Detection[];
  /**
   * Max cell-phone class score across all YOLO predictions this frame,
   * including those below the score threshold (for debug near-misses).
   */
  bestScore: number;
};

/**
 * Decode YOLOv8 raw output `[1, 84, N]` → phone-class boxes above score threshold.
 * Channels 0–3 are cx, cy, w, h in letterbox pixels; 4–83 are COCO class scores.
 */
export function decodePhoneCandidates(
  data: Float32Array,
  dims: readonly number[],
  meta: LetterboxMeta,
  scoreThreshold: number,
): DecodePhoneResult {
  const numPreds = dims.length === 3 ? dims[2]! : dims[1] === 84 ? dims[2]! : 0;
  if (!numPreds) {
    throw new Error(`Unexpected YOLO output shape: [${dims.join(", ")}]`);
  }

  const classChannel = 4 + CELL_PHONE_CLASS_ID;
  const candidates: Detection[] = [];
  let bestScore = 0;

  for (let i = 0; i < numPreds; i += 1) {
    const score = data[classChannel * numPreds + i]!;
    if (score > bestScore) {
      bestScore = score;
    }
    if (score < scoreThreshold) {
      continue;
    }

    const cx = data[0 * numPreds + i]!;
    const cy = data[1 * numPreds + i]!;
    const w = data[2 * numPreds + i]!;
    const h = data[3 * numPreds + i]!;

    const box = letterboxBoxToSource(cx, cy, w, h, meta);
    if (box.width <= 1 || box.height <= 1) {
      continue;
    }

    candidates.push({
      box,
      classId: CELL_PHONE_CLASS_ID,
      className: cocoLabel(CELL_PHONE_CLASS_ID),
      score,
    });
  }

  return { candidates, bestScore };
}

export function decodeAndNmsPhoneDetections(
  data: Float32Array,
  dims: readonly number[],
  meta: LetterboxMeta,
  options: DecodePhoneOptions,
): DecodePhoneResult & { detections: Detection[] } {
  const { candidates, bestScore } = decodePhoneCandidates(
    data,
    dims,
    meta,
    options.scoreThreshold,
  );
  const detections = nonMaxSuppression(candidates, {
    iouThreshold: options.iouThreshold,
    enabled: options.nmsEnabled,
    maxDetections: options.maxDetections,
  });
  return { candidates, detections, bestScore };
}

export function letterboxBoxToSource(
  cx: number,
  cy: number,
  w: number,
  h: number,
  meta: LetterboxMeta,
): Detection["box"] {
  const x1 = (cx - w / 2 - meta.padX) / meta.scale;
  const y1 = (cy - h / 2 - meta.padY) / meta.scale;
  const x2 = (cx + w / 2 - meta.padX) / meta.scale;
  const y2 = (cy + h / 2 - meta.padY) / meta.scale;

  const left = Math.max(0, Math.min(meta.srcWidth, x1));
  const top = Math.max(0, Math.min(meta.srcHeight, y1));
  const right = Math.max(0, Math.min(meta.srcWidth, x2));
  const bottom = Math.max(0, Math.min(meta.srcHeight, y2));

  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

/** Pack RGBA ImageData into a reused CHW float32 buffer in [0, 1]. */
export function packRgbFloatChw(
  rgba: Uint8ClampedArray,
  inputSize: number,
  out: Float32Array,
): void {
  const plane = inputSize * inputSize;
  for (let i = 0; i < plane; i += 1) {
    const px = i * 4;
    out[i] = rgba[px]! / 255;
    out[plane + i] = rgba[px + 1]! / 255;
    out[2 * plane + i] = rgba[px + 2]! / 255;
  }
}
