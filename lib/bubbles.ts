import { FISH_RADIUS, type FishState } from "@/lib/fish";

/**
 * Openness step per intensity level: ~0.05 → 1, ~0.15 → 2, ~0.3 → 3.
 * Spawning runs while openness is at or above the active floor.
 */
export const BUBBLE_ACTIVE_THRESHOLD = 0.05;

/** Openness step used to map score → intensity level. */
export const BUBBLE_OPENNESS_PER_COUNT = 0.1;

/** Cap intensity levels so a full open stays lightweight. */
export const BUBBLE_LEVEL_MAX = 12;

/** Bubbles spawned per second at level 1; higher levels multiply this. */
export const BUBBLE_SPAWN_RATE_PER_LEVEL = 4;

/** Base upward speed (px/s); canvas +y is down so rise uses negative vy. */
export const BUBBLE_RISE_SPEED = 90;

/** Horizontal drift amplitude (px/s) applied at spawn and each frame. */
export const BUBBLE_DRIFT = 28;

export type Bubble = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
};

/** Fractional spawn accumulator for continuous emission. */
export type BubbleEmitter = {
  accumulator: number;
};

export function createBubbleEmitter(): BubbleEmitter {
  return { accumulator: 0 };
}

function randRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

/**
 * Intensity level from jawOpen (~1 per 0.1 openness). 0 below 0.05.
 */
export function bubbleLevelForOpenness(openness: number): number {
  if (openness < BUBBLE_ACTIVE_THRESHOLD) {
    return 0;
  }
  const raw = Math.round(openness / BUBBLE_OPENNESS_PER_COUNT);
  return Math.min(BUBBLE_LEVEL_MAX, Math.max(1, raw));
}

function spawnOneBubble(bubbles: Bubble[], fish: FishState): void {
  const len = Math.hypot(fish.headingX, fish.headingY) || 1;
  const ux = fish.headingX / len;
  const uy = fish.headingY / len;
  const mouthX = fish.x + ux * FISH_RADIUS;
  const mouthY = fish.y + uy * FISH_RADIUS;

  bubbles.push({
    x: mouthX + randRange(-4, 4),
    y: mouthY + randRange(-3, 3),
    vx: randRange(-BUBBLE_DRIFT, BUBBLE_DRIFT),
    vy: -BUBBLE_RISE_SPEED * randRange(0.7, 1.3),
    r: randRange(2.5, 5.5),
  });
}

/**
 * Continuously spawn while openness >= 0.05. Rate scales with openness level.
 * Stops (and resets the accumulator) when openness drops below 0.05.
 */
export function emitBubblesContinuous(
  bubbles: Bubble[],
  fish: FishState,
  openness: number,
  dt: number,
  emitter: BubbleEmitter,
): void {
  const level = bubbleLevelForOpenness(openness);
  if (level <= 0 || dt <= 0) {
    emitter.accumulator = 0;
    return;
  }

  emitter.accumulator += level * BUBBLE_SPAWN_RATE_PER_LEVEL * dt;
  // Cap so a long frame hitch cannot dump a huge pile at once.
  const maxThisFrame = level * 3;
  let spawned = 0;
  while (emitter.accumulator >= 1 && spawned < maxThisFrame) {
    spawnOneBubble(bubbles, fish);
    emitter.accumulator -= 1;
    spawned += 1;
  }
  if (emitter.accumulator >= 1) {
    emitter.accumulator = emitter.accumulator % 1;
  }
}

/**
 * Rise, add light random drift, remove when fully off-screen.
 */
export function updateBubbles(
  bubbles: Bubble[],
  dt: number,
  width: number,
  height: number,
): void {
  if (dt <= 0 || bubbles.length === 0) {
    return;
  }

  for (let i = bubbles.length - 1; i >= 0; i--) {
    const b = bubbles[i];
    b.vx += randRange(-BUBBLE_DRIFT, BUBBLE_DRIFT) * dt;
    b.x += b.vx * dt;
    b.y += b.vy * dt;

    if (
      b.y + b.r < 0 ||
      b.y - b.r > height ||
      b.x + b.r < 0 ||
      b.x - b.r > width
    ) {
      bubbles.splice(i, 1);
    }
  }
}

export function drawBubbles(
  ctx: CanvasRenderingContext2D,
  bubbles: Bubble[],
): void {
  if (bubbles.length === 0) {
    return;
  }

  ctx.save();
  for (const b of bubbles) {
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(220, 245, 255, 0.35)";
    ctx.fill();
    ctx.strokeStyle = "rgba(200, 235, 255, 0.7)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.restore();
}
