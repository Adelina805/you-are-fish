/**
 * Wall-clock hysteresis for raw YOLO phone detections.
 * Do not drive gameplay from a single inference frame.
 */

/** Raw phone must stay true this long before phonePresent activates. */
export const PHONE_PRESENT_CONFIRM_MS = 400;

/** Raw phone must stay false this long before phonePresent deactivates. */
export const PHONE_ABSENT_CONFIRM_MS = 600;

/** How long “That’s better.” stays on screen after the phone leaves. */
export const PHONE_BETTER_MESSAGE_MS = 1500;

export class PhonePresenceTracker {
  private phonePresent = false;
  private presentAccumMs = 0;
  private absentAccumMs = 0;
  private lastNowMs: number | null = null;

  get present(): boolean {
    return this.phonePresent;
  }

  reset(): void {
    this.phonePresent = false;
    this.presentAccumMs = 0;
    this.absentAccumMs = 0;
    this.lastNowMs = null;
  }

  /**
   * @param rawDetected - true when latest successful infer has ≥1 post-NMS phone
   * @param nowMs - performance.now() (or equivalent wall clock)
   */
  update(rawDetected: boolean, nowMs: number): boolean {
    if (this.lastNowMs === null) {
      this.lastNowMs = nowMs;
      // Seed accumulators from the first sample without advancing time.
      if (rawDetected) {
        this.presentAccumMs = 0;
        this.absentAccumMs = 0;
      }
      return this.phonePresent;
    }

    const dtMs = Math.max(0, Math.min(nowMs - this.lastNowMs, 100));
    this.lastNowMs = nowMs;

    if (rawDetected) {
      this.presentAccumMs += dtMs;
      this.absentAccumMs = 0;
      if (!this.phonePresent && this.presentAccumMs >= PHONE_PRESENT_CONFIRM_MS) {
        this.phonePresent = true;
      }
    } else {
      this.absentAccumMs += dtMs;
      this.presentAccumMs = 0;
      if (this.phonePresent && this.absentAccumMs >= PHONE_ABSENT_CONFIRM_MS) {
        this.phonePresent = false;
      }
    }

    return this.phonePresent;
  }
}
