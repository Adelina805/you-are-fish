"""Exponential moving average smoothing for calibrated head pose."""

from __future__ import annotations

from dataclasses import dataclass

from head_pose import HeadPose

SMOOTHING_ALPHA = 0.25


@dataclass
class PoseSmoother:
    """Per-axis EMA on calibrated yaw/pitch."""

    alpha: float = SMOOTHING_ALPHA
    _yaw: float | None = None
    _pitch: float | None = None
    _roll: float | None = None

    def reset(self) -> None:
        self._yaw = None
        self._pitch = None
        self._roll = None

    def update(self, pose: HeadPose | None) -> HeadPose | None:
        if pose is None:
            return None

        if self._yaw is None:
            self._yaw = pose.yaw_deg
            self._pitch = pose.pitch_deg
            self._roll = pose.roll_deg
        else:
            self._yaw = self.alpha * pose.yaw_deg + (1.0 - self.alpha) * self._yaw
            self._pitch = self.alpha * pose.pitch_deg + (1.0 - self.alpha) * self._pitch
            self._roll = self.alpha * pose.roll_deg + (1.0 - self.alpha) * self._roll

        return HeadPose(
            yaw_deg=self._yaw,
            pitch_deg=self._pitch,
            roll_deg=self._roll,
        )
