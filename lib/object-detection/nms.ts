export type NmsBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type NmsDetection = {
  box: NmsBox;
  score: number;
};

/**
 * Intersection-over-union for axis-aligned boxes in `{x,y,width,height}` form.
 */
export function iou(a: NmsBox, b: NmsBox): number {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;

  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) {
    return 0;
  }

  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

export type NmsOptions = {
  /** Suppress boxes with IoU >= this value relative to a kept box. */
  iouThreshold: number;
  /**
   * When false, return candidates unchanged (homework: inspect duplicates).
   * Default true.
   */
  enabled?: boolean;
  /** Optional cap on kept detections after NMS. */
  maxDetections?: number;
};

/**
 * Greedy score-sorted Non-Maximum Suppression.
 * Returns a new array; does not mutate `candidates`.
 */
export function nonMaxSuppression<T extends NmsDetection>(
  candidates: readonly T[],
  options: NmsOptions,
): T[] {
  const enabled = options.enabled !== false;
  if (!enabled) {
    const max = options.maxDetections;
    if (max !== undefined && candidates.length > max) {
      return candidates.slice(0, max);
    }
    return candidates.slice();
  }

  const iouThreshold = options.iouThreshold;
  const sorted = candidates.slice().sort((a, b) => b.score - a.score);
  const kept: T[] = [];

  while (sorted.length > 0) {
    const best = sorted.shift()!;
    kept.push(best);
    if (
      options.maxDetections !== undefined &&
      kept.length >= options.maxDetections
    ) {
      break;
    }

    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (iou(best.box, sorted[i].box) >= iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return kept;
}
