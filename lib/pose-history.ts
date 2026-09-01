import type { HeadPose } from "@/lib/head-pose";

export const HISTORY_MAXLEN = 120;

export class PoseHistory {
  private yaw: number[] = [];
  private pitch: number[] = [];
  private maxlen: number;

  constructor(maxlen = HISTORY_MAXLEN) {
    this.maxlen = maxlen;
  }

  append(pose: HeadPose): void {
    this.yaw.push(pose.yawDeg);
    this.pitch.push(pose.pitchDeg);
    if (this.yaw.length > this.maxlen) {
      this.yaw.shift();
      this.pitch.shift();
    }
  }

  yawSamples(): number[] {
    return this.yaw;
  }

  pitchSamples(): number[] {
    return this.pitch;
  }

  clear(): void {
    this.yaw = [];
    this.pitch = [];
  }
}
