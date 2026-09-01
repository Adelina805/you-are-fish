"""Directional classifier from calibrated, smoothed yaw/pitch."""

from __future__ import annotations

from enum import Enum

from head_pose import HeadPose

YAW_DEAD_ZONE_DEG = 8.0
PITCH_DEAD_ZONE_DEG = 8.0


class Direction(Enum):
    CENTER = "CENTER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"


def classify_direction(
    pose: HeadPose | None,
    yaw_dead_zone_deg: float = YAW_DEAD_ZONE_DEG,
    pitch_dead_zone_deg: float = PITCH_DEAD_ZONE_DEG,
) -> Direction | None:
    """Classify smoothed calibrated pose into one of five directions."""
    if pose is None:
        return None

    yaw = pose.yaw_deg
    pitch = pose.pitch_deg
    yaw_in_zone = abs(yaw) <= yaw_dead_zone_deg
    pitch_in_zone = abs(pitch) <= pitch_dead_zone_deg

    if yaw_in_zone and pitch_in_zone:
        return Direction.CENTER

    yaw_ratio = abs(yaw) / yaw_dead_zone_deg if yaw_dead_zone_deg > 0 else 0.0
    pitch_ratio = abs(pitch) / pitch_dead_zone_deg if pitch_dead_zone_deg > 0 else 0.0

    if yaw_ratio >= pitch_ratio:
        return Direction.RIGHT if yaw > 0 else Direction.LEFT
    return Direction.DOWN if pitch > 0 else Direction.UP
