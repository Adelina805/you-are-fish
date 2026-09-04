import type { HeadPose } from "@/lib/head-pose";

export const YAW_DEAD_ZONE_DEG = 3;
export const PITCH_DEAD_ZONE_DEG = 3;

/**
 * Screen-space look direction after calibration.
 * +x = look toward screen right, +y = look up.
 * angleDeg: atan2(y, x); 0° = right, 90° = up; null at center.
 * magnitude: 0 in the dead zone; hypot(nx, ny) - 1 outside it.
 */
export type LookDirection = {
  x: number;
  y: number;
  angleDeg: number | null;
  magnitude: number;
};

export function classifyDirection(
  pose: HeadPose | null,
  yawDeadZoneDeg = YAW_DEAD_ZONE_DEG,
  pitchDeadZoneDeg = PITCH_DEAD_ZONE_DEG,
): LookDirection | null {
  if (pose === null) {
    return null;
  }

  const yawDz = yawDeadZoneDeg > 0 ? yawDeadZoneDeg : 1;
  const pitchDz = pitchDeadZoneDeg > 0 ? pitchDeadZoneDeg : 1;

  // Normalize into screen space: +yaw is look left → -x; +pitch is look up → +y.
  const nx = -pose.yawDeg / yawDz;
  const ny = pose.pitchDeg / pitchDz;
  const r = Math.hypot(nx, ny);

  if (r <= 1) {
    return { x: 0, y: 0, angleDeg: null, magnitude: 0 };
  }

  const x = nx / r;
  const y = ny / r;
  return {
    x,
    y,
    angleDeg: (Math.atan2(y, x) * 180) / Math.PI,
    magnitude: r - 1,
  };
}
