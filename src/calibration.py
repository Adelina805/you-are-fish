"""Per-user neutral head-pose calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

from head_pose import HeadPose

DURATION_SEC = 1.5


@dataclass
class NeutralPoseCalibrator:
    """Collects a short rest-pose sample and subtracts it from later poses."""

    duration_sec: float = DURATION_SEC
    _yaw_samples: list[float] = field(default_factory=list)
    _pitch_samples: list[float] = field(default_factory=list)
    _collected_sec: float = 0.0
    _last_update_time: float | None = None
    _baseline_yaw: float | None = None
    _baseline_pitch: float | None = None
    _active: bool = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_complete(self) -> bool:
        return self._baseline_yaw is not None and self._baseline_pitch is not None

    @property
    def baseline_yaw(self) -> float | None:
        return self._baseline_yaw

    @property
    def baseline_pitch(self) -> float | None:
        return self._baseline_pitch

    @property
    def progress(self) -> float:
        if self.is_complete:
            return 1.0
        if self.duration_sec <= 0:
            return 1.0
        return min(1.0, self._collected_sec / self.duration_sec)

    def start(self) -> None:
        self._yaw_samples.clear()
        self._pitch_samples.clear()
        self._collected_sec = 0.0
        self._last_update_time = None
        self._baseline_yaw = None
        self._baseline_pitch = None
        self._active = True

    def update(self, pose: HeadPose | None, now: float) -> None:
        if not self._active or self.is_complete:
            return

        if pose is None:
            self._last_update_time = None
            return

        if self._last_update_time is not None:
            delta = now - self._last_update_time
            if delta > 0:
                self._collected_sec += delta
        self._last_update_time = now

        self._yaw_samples.append(pose.yaw_deg)
        self._pitch_samples.append(pose.pitch_deg)

        if self._collected_sec >= self.duration_sec:
            self._baseline_yaw = sum(self._yaw_samples) / len(self._yaw_samples)
            self._baseline_pitch = sum(self._pitch_samples) / len(self._pitch_samples)
            self._active = False

    def apply(self, pose: HeadPose) -> HeadPose:
        if not self.is_complete:
            return pose

        return HeadPose(
            yaw_deg=pose.yaw_deg - self._baseline_yaw,
            pitch_deg=pose.pitch_deg - self._baseline_pitch,
            roll_deg=pose.roll_deg,
        )
