import type { LookDirection } from "@/lib/direction";

/**
 * Pixels per second² toward the look direction (scaled by look magnitude).
 * Keep roughly FISH_MAX_SPEED * FISH_DRAG so mag≈1 settles near the cap quickly.
 */
export const FISH_ACCEL = 6000;

/**
 * Per-second velocity decay. Higher = velocity tracks head more tightly
 * (less accel ramp / less coast).
 */
export const FISH_DRAG = 10;

/** Maximum speed in pixels per second. */
export const FISH_MAX_SPEED = 600;

/**
 * Gentle cruise speed (px/s) along the last heading when look is CENTER
 * or unavailable. Settles via thrust = IDLE * DRAG against drag.
 */
export const FISH_IDLE_SPEED = 80;

export const FISH_RADIUS = 18;

/** Ignore tiny speeds when updating facing / zeroing residual velocity. */
const FISH_SPEED_EPSILON = 1;

export type FishState = {
  x: number;
  y: number;
  /** Velocity in canvas space (px/s; +y down). */
  vx: number;
  vy: number;
  /** Last heading (canvas space: +y down). */
  headingX: number;
  headingY: number;
};

export function createFish(width: number, height: number): FishState {
  return {
    x: width / 2,
    y: height / 2,
    vx: 0,
    vy: 0,
    headingX: 1,
    headingY: 0,
  };
}

function clampFish(fish: FishState, width: number, height: number): void {
  const r = FISH_RADIUS;
  const minX = r;
  const maxX = width - r;
  const minY = r;
  const maxY = height - r;

  if (fish.x < minX) {
    fish.x = minX;
    fish.vx = 0;
  } else if (fish.x > maxX) {
    fish.x = maxX;
    fish.vx = 0;
  }

  if (fish.y < minY) {
    fish.y = minY;
    fish.vy = 0;
  } else if (fish.y > maxY) {
    fish.y = maxY;
    fish.vy = 0;
  }
}

export function updateFish(
  fish: FishState,
  direction: LookDirection | null,
  dt: number,
  width: number,
  height: number,
): void {
  if (dt <= 0) {
    clampFish(fish, width, height);
    return;
  }

  if (direction && direction.magnitude > 0) {
    // Thrust toward look direction when outside the dead zone.
    const desiredX = direction.x;
    const desiredY = -direction.y;
    const thrust = FISH_ACCEL * direction.magnitude;
    fish.vx += desiredX * thrust * dt;
    fish.vy += desiredY * thrust * dt;
  } else {
    // Keep a gentle cruise along the last heading at CENTER / no pose.
    const len = Math.hypot(fish.headingX, fish.headingY) || 1;
    const ux = fish.headingX / len;
    const uy = fish.headingY / len;
    const idleThrust = FISH_IDLE_SPEED * FISH_DRAG;
    fish.vx += ux * idleThrust * dt;
    fish.vy += uy * idleThrust * dt;
  }

  // Linear drag; with idle thrust this settles near FISH_IDLE_SPEED at CENTER.
  const dragFactor = Math.max(0, 1 - FISH_DRAG * dt);
  fish.vx *= dragFactor;
  fish.vy *= dragFactor;

  // Cap maximum speed.
  const speed = Math.hypot(fish.vx, fish.vy);
  if (speed > FISH_MAX_SPEED) {
    const scale = FISH_MAX_SPEED / speed;
    fish.vx *= scale;
    fish.vy *= scale;
  } else if (speed < FISH_SPEED_EPSILON) {
    fish.vx = 0;
    fish.vy = 0;
  }

  // Face travel direction while moving.
  const travelSpeed = Math.hypot(fish.vx, fish.vy);
  if (travelSpeed >= FISH_SPEED_EPSILON) {
    fish.headingX = fish.vx / travelSpeed;
    fish.headingY = fish.vy / travelSpeed;
  }

  fish.x += fish.vx * dt;
  fish.y += fish.vy * dt;
  clampFish(fish, width, height);
}

export function drawFish(
  ctx: CanvasRenderingContext2D,
  fish: FishState,
): void {
  const { x, y, headingX, headingY } = fish;
  const len = Math.hypot(headingX, headingY) || 1;
  const ux = headingX / len;
  const uy = headingY / len;
  const px = -uy;
  const py = ux;
  const r = FISH_RADIUS;

  ctx.save();

  // Solid V-tail behind the body.
  const rearX = x - ux * r * 0.85;
  const rearY = y - uy * r * 0.85;
  const tipX = x - ux * r * 1.85;
  const tipY = y - uy * r * 1.85;
  const flare = r * 0.95;
  ctx.fillStyle = "#e09020";
  ctx.beginPath();
  ctx.moveTo(rearX, rearY);
  ctx.lineTo(tipX + px * flare, tipY + py * flare);
  ctx.lineTo(tipX - px * flare, tipY - py * flare);
  ctx.closePath();
  ctx.fill();

  // Body
  ctx.fillStyle = "#f0a030";
  ctx.beginPath();
  ctx.ellipse(x, y, r * 1.15, r * 0.75, Math.atan2(uy, ux), 0, Math.PI * 2);
  ctx.fill();

  // Eye centered on the forward face
  const eyeX = x + ux * r * 0.5;
  const eyeY = y + uy * r * 0.5;
  ctx.fillStyle = "#1a1a1a";
  ctx.beginPath();
  ctx.arc(eyeX, eyeY, r * 0.14, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}
