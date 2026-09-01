import type { HeadPose } from "@/lib/head-pose";

export const DURATION_SEC = 1.5;

export class NeutralPoseCalibrator {
  durationSec: number;
  private yawSamples: number[] = [];
  private pitchSamples: number[] = [];
  private collectedSec = 0;
  private lastUpdateTime: number | null = null;
  private _baselineYaw: number | null = null;
  private _baselinePitch: number | null = null;
  private active = false;

  constructor(durationSec = DURATION_SEC) {
    this.durationSec = durationSec;
  }

  get isActive(): boolean {
    return this.active;
  }

  get isComplete(): boolean {
    return this._baselineYaw !== null && this._baselinePitch !== null;
  }

  get baselineYaw(): number | null {
    return this._baselineYaw;
  }

  get baselinePitch(): number | null {
    return this._baselinePitch;
  }

  start(): void {
    this.yawSamples = [];
    this.pitchSamples = [];
    this.collectedSec = 0;
    this.lastUpdateTime = null;
    this._baselineYaw = null;
    this._baselinePitch = null;
    this.active = true;
  }

  update(pose: HeadPose | null, now: number): void {
    if (!this.active || this.isComplete) {
      return;
    }

    if (pose === null) {
      this.lastUpdateTime = null;
      return;
    }

    if (this.lastUpdateTime !== null) {
      const delta = now - this.lastUpdateTime;
      if (delta > 0) {
        this.collectedSec += delta;
      }
    }
    this.lastUpdateTime = now;

    this.yawSamples.push(pose.yawDeg);
    this.pitchSamples.push(pose.pitchDeg);

    if (this.collectedSec >= this.durationSec && this.yawSamples.length > 0) {
      this._baselineYaw =
        this.yawSamples.reduce((sum, value) => sum + value, 0) / this.yawSamples.length;
      this._baselinePitch =
        this.pitchSamples.reduce((sum, value) => sum + value, 0) / this.pitchSamples.length;
      this.active = false;
    }
  }

  apply(pose: HeadPose): HeadPose {
    if (!this.isComplete || this._baselineYaw === null || this._baselinePitch === null) {
      return pose;
    }

    return {
      yawDeg: pose.yawDeg - this._baselineYaw,
      pitchDeg: pose.pitchDeg - this._baselinePitch,
      rollDeg: pose.rollDeg,
    };
  }
}
