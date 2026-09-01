import type { HeadPose } from "@/lib/head-pose";

export const YAW_DEAD_ZONE_DEG = 8;
export const PITCH_DEAD_ZONE_DEG = 8;

export type Direction = "CENTER" | "LEFT" | "RIGHT" | "UP" | "DOWN";

export function classifyDirection(
  pose: HeadPose | null,
  yawDeadZoneDeg = YAW_DEAD_ZONE_DEG,
  pitchDeadZoneDeg = PITCH_DEAD_ZONE_DEG,
): Direction | null {
  if (pose === null) {
    return null;
  }

  const yaw = pose.yawDeg;
  const pitch = pose.pitchDeg;
  const yawInZone = Math.abs(yaw) <= yawDeadZoneDeg;
  const pitchInZone = Math.abs(pitch) <= pitchDeadZoneDeg;

  if (yawInZone && pitchInZone) {
    return "CENTER";
  }

  const yawRatio = yawDeadZoneDeg > 0 ? Math.abs(yaw) / yawDeadZoneDeg : 0;
  const pitchRatio = pitchDeadZoneDeg > 0 ? Math.abs(pitch) / pitchDeadZoneDeg : 0;

  if (yawRatio >= pitchRatio) {
    return yaw > 0 ? "RIGHT" : "LEFT";
  }
  return pitch > 0 ? "DOWN" : "UP";
}
