export type HeadPose = {
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
};

const GIMBAL_LOCK_EPS = 1e-6;
const YAW_SIGN = 1;
const PITCH_SIGN = 1;
const ROLL_SIGN = 1;

export function estimateHeadPose(matrix: ArrayLike<number> | undefined): HeadPose | null {
  const rotation = rotationFromTransform(matrix);
  if (!rotation) {
    return null;
  }

  const [yaw, pitch, roll] = rotationToYxzDegrees(rotation);
  return {
    yawDeg: YAW_SIGN * yaw,
    pitchDeg: PITCH_SIGN * pitch,
    rollDeg: ROLL_SIGN * roll,
  };
}

function rotationFromTransform(matrix: ArrayLike<number> | undefined): number[] | null {
  if (!matrix || matrix.length < 16) {
    return null;
  }

  for (let i = 0; i < 16; i += 1) {
    if (!Number.isFinite(matrix[i])) {
      return null;
    }
  }

  // MediaPipe JS stores 4x4 matrices in column-major order (Three.js-compatible).
  const scaled = [
    matrix[0],
    matrix[4],
    matrix[8],
    matrix[1],
    matrix[5],
    matrix[9],
    matrix[2],
    matrix[6],
    matrix[10],
  ];

  const rotation = polarRotation(scaled);
  if (!rotation || !rotation.every(Number.isFinite)) {
    return null;
  }

  if (det3(rotation) < 0) {
    rotation[2] *= -1;
    rotation[5] *= -1;
    rotation[8] *= -1;
  }

  return rotation;
}

function polarRotation(matrix: number[]): number[] | null {
  let current = [...matrix];

  for (let i = 0; i < 10; i += 1) {
    const inverse = invert3(current);
    if (!inverse) {
      return null;
    }
    const invT = transpose3(inverse);
    current = [
      0.5 * (current[0] + invT[0]),
      0.5 * (current[1] + invT[1]),
      0.5 * (current[2] + invT[2]),
      0.5 * (current[3] + invT[3]),
      0.5 * (current[4] + invT[4]),
      0.5 * (current[5] + invT[5]),
      0.5 * (current[6] + invT[6]),
      0.5 * (current[7] + invT[7]),
      0.5 * (current[8] + invT[8]),
    ];
  }

  return current;
}

function rotationToYxzDegrees(rotation: number[]): [number, number, number] {
  const pitch = Math.atan2(-rotation[5], Math.hypot(rotation[3], rotation[4]));
  const cosPitch = Math.cos(pitch);

  let yaw: number;
  let roll: number;
  if (Math.abs(cosPitch) > GIMBAL_LOCK_EPS) {
    yaw = Math.atan2(rotation[2], rotation[8]);
    roll = Math.atan2(rotation[3], rotation[4]);
  } else {
    yaw = Math.atan2(-rotation[6], rotation[0]);
    roll = 0;
  }

  return [
    (yaw * 180) / Math.PI,
    (pitch * 180) / Math.PI,
    (roll * 180) / Math.PI,
  ];
}

function transpose3(m: number[]): number[] {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

function det3(m: number[]): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

function invert3(m: number[]): number[] | null {
  const det = det3(m);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) {
    return null;
  }
  const inv = 1 / det;
  return [
    (m[4] * m[8] - m[5] * m[7]) * inv,
    (m[2] * m[7] - m[1] * m[8]) * inv,
    (m[1] * m[5] - m[2] * m[4]) * inv,
    (m[5] * m[6] - m[3] * m[8]) * inv,
    (m[0] * m[8] - m[2] * m[6]) * inv,
    (m[2] * m[3] - m[0] * m[5]) * inv,
    (m[3] * m[7] - m[4] * m[6]) * inv,
    (m[1] * m[6] - m[0] * m[7]) * inv,
    (m[0] * m[4] - m[1] * m[3]) * inv,
  ];
}
