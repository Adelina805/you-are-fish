import type { LookDirection } from "@/lib/direction";

/** Pixels per second at LookDirection magnitude 1. */
export const FISH_SPEED = 220;

/** Slow coast speed when look is CENTER / unavailable. */
export const FISH_IDLE_SPEED = 28;

export const FISH_RADIUS = 18;

export type FishState = {
  x: number;
  y: number;
  /** Last heading (canvas space: +y down). */
  headingX: number;
  headingY: number;
};

export function createFish(width: number, height: number): FishState {
  return {
    x: width / 2,
    y: height / 2,
    headingX: 1,
    headingY: 0,
  };
}

function clampFish(fish: FishState, width: number, height: number): void {
  const r = FISH_RADIUS;
  fish.x = Math.min(width - r, Math.max(r, fish.x));
  fish.y = Math.min(height - r, Math.max(r, fish.y));
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
    fish.headingX = direction.x;
    fish.headingY = -direction.y;
  }

  const len = Math.hypot(fish.headingX, fish.headingY) || 1;
  const ux = fish.headingX / len;
  const uy = fish.headingY / len;
  const speed =
    direction && direction.magnitude > 0
      ? direction.magnitude * FISH_SPEED
      : FISH_IDLE_SPEED;

  fish.x += ux * speed * dt;
  fish.y += uy * speed * dt;
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
