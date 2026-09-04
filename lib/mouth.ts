import type { Classifications } from "@mediapipe/tasks-vision";

/** Enter MOUTH_OPEN when jawOpen rises above this. */
export const MOUTH_OPEN_THRESHOLD = 0.05;

/** Leave MOUTH_OPEN when jawOpen falls below this. */
export const MOUTH_CLOSE_THRESHOLD = 0.03;

export type MouthState = "MOUTH_OPEN" | "MOUTH_CLOSED";

export type MouthStatus = {
  openness: number;
  state: MouthState;
  /** True only on the closed → open edge this frame. */
  justOpened: boolean;
};

/**
 * Pull the jawOpen blendshape score (0–1). Returns 0 when missing.
 */
export function extractJawOpen(
  blendshapes: Classifications | undefined | null,
): number {
  if (!blendshapes?.categories?.length) {
    return 0;
  }
  for (const category of blendshapes.categories) {
    if (category.categoryName === "jawOpen") {
      const score = category.score;
      if (typeof score !== "number" || Number.isNaN(score)) {
        return 0;
      }
      return Math.min(1, Math.max(0, score));
    }
  }
  return 0;
}

/**
 * Dual-threshold hysteresis classifier for mouth open/closed.
 * Face loss forces MOUTH_CLOSED.
 */
export class MouthTracker {
  private state: MouthState = "MOUTH_CLOSED";
  private openThreshold: number;
  private closeThreshold: number;

  constructor(
    openThreshold = MOUTH_OPEN_THRESHOLD,
    closeThreshold = MOUTH_CLOSE_THRESHOLD,
  ) {
    this.openThreshold = openThreshold;
    this.closeThreshold = Math.min(closeThreshold, openThreshold);
  }

  reset(): void {
    this.state = "MOUTH_CLOSED";
  }

  update(
    blendshapes: Classifications | undefined | null,
    faceDetected: boolean,
  ): MouthStatus {
    if (!faceDetected) {
      this.state = "MOUTH_CLOSED";
      return { openness: 0, state: "MOUTH_CLOSED", justOpened: false };
    }

    const openness = extractJawOpen(blendshapes);
    const wasOpen = this.state === "MOUTH_OPEN";

    if (wasOpen) {
      if (openness < this.closeThreshold) {
        this.state = "MOUTH_CLOSED";
      }
    } else if (openness > this.openThreshold) {
      this.state = "MOUTH_OPEN";
    }

    const justOpened = !wasOpen && this.state === "MOUTH_OPEN";
    return { openness, state: this.state, justOpened };
  }
}
