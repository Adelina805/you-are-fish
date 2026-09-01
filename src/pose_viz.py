"""Debug visualization for raw head-pose signals (no smoothing)."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from head_pose import HeadPose

YAW_RANGE_DEG = 45.0
PITCH_RANGE_DEG = 45.0
HISTORY_MAXLEN = 120

PANEL_MARGIN = 20
YAW_BAR_LEN = 260
PITCH_BAR_LEN = 180
PANEL_PAD = 14
AXIS_THICKNESS = 2
TICK_LEN = 6
ZERO_CROSS_LEN = 8
MARKER_RADIUS = 8
HISTORY_RADIUS = 3
LABEL_FONT_SCALE = 0.55

_AXIS_COLOR = (140, 140, 140)
_ZERO_COLOR = (220, 220, 220)
_HISTORY_COLOR = (180, 200, 80)
_MARKER_COLOR = (80, 220, 255)
_LABEL_COLOR = (200, 200, 200)
_BACKING_ALPHA = 0.55


class PoseHistory:
    """Rolling buffer of raw yaw/pitch samples."""

    def __init__(self, maxlen: int = HISTORY_MAXLEN) -> None:
        self._yaw: deque[float] = deque(maxlen=maxlen)
        self._pitch: deque[float] = deque(maxlen=maxlen)

    def append(self, pose: HeadPose) -> None:
        self._yaw.append(pose.yaw_deg)
        self._pitch.append(pose.pitch_deg)

    def yaw_samples(self) -> list[float]:
        return list(self._yaw)

    def pitch_samples(self) -> list[float]:
        return list(self._pitch)

    def clear(self) -> None:
        self._yaw.clear()
        self._pitch.clear()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _yaw_to_x(yaw_deg: float, center_x: int, half_len: int) -> int:
    ratio = _clamp(yaw_deg / YAW_RANGE_DEG, -1.0, 1.0)
    # Positive yaw = looking left → marker moves left on the bar.
    return int(round(center_x - ratio * half_len))


def _pitch_to_y(pitch_deg: float, center_y: int, half_len: int) -> int:
    ratio = _clamp(pitch_deg / PITCH_RANGE_DEG, -1.0, 1.0)
    # Positive pitch = looking up → marker moves up on the bar.
    return int(round(center_y - ratio * half_len))


def draw_pose_signal_viz(
    frame: np.ndarray,
    pose: HeadPose | None,
    history: PoseHistory,
    top_offset: int = 0,
) -> None:
    """Draw yaw (horizontal) and pitch (vertical) bars with raw history trails."""
    _, frame_w = frame.shape[:2]
    yaw_half = YAW_BAR_LEN // 2
    pitch_half = PITCH_BAR_LEN // 2

    corner_x = frame_w - PANEL_MARGIN - yaw_half
    corner_y = PANEL_MARGIN + top_offset + pitch_half

    yaw_left = corner_x - yaw_half
    yaw_right = corner_x + yaw_half
    pitch_top = corner_y - pitch_half
    pitch_bottom = corner_y + pitch_half

    panel_left = yaw_left - PANEL_PAD
    panel_top = pitch_top - PANEL_PAD
    panel_right = yaw_right + PANEL_PAD
    panel_bottom = pitch_bottom + PANEL_PAD

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.addWeighted(overlay, _BACKING_ALPHA, frame, 1.0 - _BACKING_ALPHA, 0, frame)

    cv2.line(
        frame,
        (yaw_left, corner_y),
        (yaw_right, corner_y),
        _AXIS_COLOR,
        AXIS_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (corner_x, pitch_top),
        (corner_x, pitch_bottom),
        _AXIS_COLOR,
        AXIS_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (corner_x - ZERO_CROSS_LEN, corner_y),
        (corner_x + ZERO_CROSS_LEN, corner_y),
        _ZERO_COLOR,
        AXIS_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (corner_x, corner_y - ZERO_CROSS_LEN),
        (corner_x, corner_y + ZERO_CROSS_LEN),
        _ZERO_COLOR,
        AXIS_THICKNESS,
        cv2.LINE_AA,
    )

    for tick in (-1.0, 1.0):
        tick_x = _yaw_to_x(tick * YAW_RANGE_DEG, corner_x, yaw_half)
        cv2.line(
            frame,
            (tick_x, corner_y - TICK_LEN),
            (tick_x, corner_y + TICK_LEN),
            _AXIS_COLOR,
            AXIS_THICKNESS,
            cv2.LINE_AA,
        )
        tick_y = _pitch_to_y(tick * PITCH_RANGE_DEG, corner_y, pitch_half)
        cv2.line(
            frame,
            (corner_x - TICK_LEN, tick_y),
            (corner_x + TICK_LEN, tick_y),
            _AXIS_COLOR,
            AXIS_THICKNESS,
            cv2.LINE_AA,
        )

    yaw_samples = history.yaw_samples()
    pitch_samples = history.pitch_samples()
    sample_count = min(len(yaw_samples), len(pitch_samples))
    for index in range(sample_count):
        fade = (index + 1) / sample_count
        color = tuple(int(channel * (0.25 + 0.75 * fade)) for channel in _HISTORY_COLOR)
        yaw_x = _yaw_to_x(yaw_samples[index], corner_x, yaw_half)
        pitch_y = _pitch_to_y(pitch_samples[index], corner_y, pitch_half)
        cv2.circle(frame, (yaw_x, corner_y), HISTORY_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (corner_x, pitch_y), HISTORY_RADIUS, color, -1, cv2.LINE_AA)

    if pose is not None:
        marker_x = _yaw_to_x(pose.yaw_deg, corner_x, yaw_half)
        marker_y = _pitch_to_y(pose.pitch_deg, corner_y, pitch_half)
        cv2.circle(frame, (marker_x, corner_y), MARKER_RADIUS, _MARKER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, (corner_x, marker_y), MARKER_RADIUS, _MARKER_COLOR, -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        "yaw",
        (yaw_left, corner_y + 22),
        font,
        LABEL_FONT_SCALE,
        _LABEL_COLOR,
        AXIS_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "pitch",
        (corner_x + 10, pitch_top + 16),
        font,
        LABEL_FONT_SCALE,
        _LABEL_COLOR,
        AXIS_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "0",
        (corner_x + 8, corner_y + 20),
        font,
        LABEL_FONT_SCALE,
        _LABEL_COLOR,
        AXIS_THICKNESS,
        cv2.LINE_AA,
    )
