import type { HeadPose } from "@/lib/head-pose";
import type { Detection } from "@/lib/object-detection/phone-decode";

export const SMOOTHING_ALPHA = 0.25;

/** Per-frame EMA toward the latest YOLO boxes (visual only; cheap). */
export const PHONE_BOX_SMOOTHING_ALPHA = 0.2;

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

function cloneDetection(detection: Detection): Detection {
  return {
    box: { ...detection.box },
    classId: detection.classId,
    className: detection.className,
    score: detection.score,
  };
}

/**
 * Ease drawn phone boxes toward the latest YOLO results each animation frame.
 * Does not change inference rate or results used for debug counts.
 */
export class PhoneBoxSmoother {
  alpha: number;
  private current: Detection[] = [];

  constructor(alpha = PHONE_BOX_SMOOTHING_ALPHA) {
    this.alpha = alpha;
  }

  reset(): void {
    this.current = [];
  }

  update(detections: readonly Detection[]): Detection[] {
    if (detections.length === 0) {
      this.current = [];
      return [];
    }

    const targets = [...detections].sort((a, b) => b.score - a.score);

    if (this.current.length !== targets.length) {
      this.current = targets.map(cloneDetection);
      return this.current.map(cloneDetection);
    }

    const a = this.alpha;
    const inv = 1 - a;
    this.current = targets.map((target, index) => {
      const prev = this.current[index]!;
      return {
        classId: target.classId,
        className: target.className,
        score: target.score,
        box: {
          x: a * target.box.x + inv * prev.box.x,
          y: a * target.box.y + inv * prev.box.y,
          width: a * target.box.width + inv * prev.box.width,
          height: a * target.box.height + inv * prev.box.height,
        },
      };
    });

    return this.current.map(cloneDetection);
  }
}
