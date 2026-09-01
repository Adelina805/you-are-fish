"""Head orientation from MediaPipe's facial transformation matrix.

MediaPipe Face Landmarker can output a 4x4 matrix T that maps the static
canonical 3D face into the detected face. That matrix is a weighted Procrustes
alignment (uniform scale + rotation + translation) and is meant to follow head
pose rather than expression.

T has the form:

    [ sR  t ]
    [  0  1 ]

MediaPipe's metric space is right-handed, with the camera at the origin looking
down -Z. This module:

1. Takes M = T[:3, :3] (scaled rotation).
2. Recovers the nearest rotation R via SVD (Orthogonal Procrustes).
3. Converts R to intrinsic Tait-Bryan YXZ Euler angles in degrees:
   - yaw:   rotation about vertical Y (looking left/right)
   - pitch: rotation about X (looking up/down)
   - roll:  rotation about Z (head tilt)

Sign convention (unmirrored webcam):

- Positive yaw: looking left
- Positive pitch: looking up
- Positive roll: tilting clockwise from the camera's view
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GIMBAL_LOCK_EPS = 1e-6

# Applied after YXZ conversion so displayed signs match the convention above.
# Tuned against the unmirrored webcam; do not flip signs in the overlay.
_YAW_SIGN = 1.0
_PITCH_SIGN = 1.0
_ROLL_SIGN = 1.0


@dataclass(frozen=True)
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def estimate_head_pose(matrix: np.ndarray | None) -> HeadPose | None:
    """Return yaw/pitch/roll in degrees from a 4x4 facial transformation matrix."""
    rotation = _rotation_from_transform(matrix)
    if rotation is None:
        return None

    yaw, pitch, roll = _rotation_to_yxz_degrees(rotation)
    return HeadPose(
        yaw_deg=float(_YAW_SIGN * yaw),
        pitch_deg=float(_PITCH_SIGN * pitch),
        roll_deg=float(_ROLL_SIGN * roll),
    )


def _rotation_from_transform(matrix: np.ndarray | None) -> np.ndarray | None:
    if matrix is None:
        return None

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return None

    scaled_rotation = transform[:3, :3]
    try:
        u, _, vt = np.linalg.svd(scaled_rotation)
    except np.linalg.LinAlgError:
        return None

    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rotation = u @ vt

    if not np.isfinite(rotation).all():
        return None
    return rotation


def _rotation_to_yxz_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
    """Extract intrinsic YXZ Euler angles from a 3x3 rotation matrix.

    R = Ry(yaw) @ Rx(pitch) @ Rz(roll), so:

        pitch = atan2(-R[1, 2], hypot(R[1, 0], R[1, 1]))
        yaw   = atan2( R[0, 2], R[2, 2])
        roll  = atan2( R[1, 0], R[1, 1])
    """
    pitch = np.arctan2(-rotation[1, 2], np.hypot(rotation[1, 0], rotation[1, 1]))
    cos_pitch = np.cos(pitch)

    if abs(cos_pitch) > _GIMBAL_LOCK_EPS:
        yaw = np.arctan2(rotation[0, 2], rotation[2, 2])
        roll = np.arctan2(rotation[1, 0], rotation[1, 1])
    else:
        yaw = np.arctan2(-rotation[2, 0], rotation[0, 0])
        roll = 0.0

    return (
        float(np.degrees(yaw)),
        float(np.degrees(pitch)),
        float(np.degrees(roll)),
    )
