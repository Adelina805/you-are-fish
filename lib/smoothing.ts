import type { HeadPose } from "@/lib/head-pose";

export const SMOOTHING_ALPHA = 0.25;

export class PoseSmoother {
  alpha: number;
  private yaw: number | null = null;
  private pitch: number | null = null;
  private roll: number | null = null;

  constructor(alpha = SMOOTHING_ALPHA) {
    this.alpha = alpha;
  }

  reset(): void {
    this.yaw = null;
    this.pitch = null;
    this.roll = null;
  }

  update(pose: HeadPose | null): HeadPose | null {
    if (pose === null) {
      return null;
    }

    if (this.yaw === null || this.pitch === null || this.roll === null) {
      this.yaw = pose.yawDeg;
      this.pitch = pose.pitchDeg;
      this.roll = pose.rollDeg;
    } else {
      this.yaw = this.alpha * pose.yawDeg + (1 - this.alpha) * this.yaw;
      this.pitch = this.alpha * pose.pitchDeg + (1 - this.alpha) * this.pitch;
      this.roll = this.alpha * pose.rollDeg + (1 - this.alpha) * this.roll;
    }

    return {
      yawDeg: this.yaw,
      pitchDeg: this.pitch,
      rollDeg: this.roll,
    };
  }
}
