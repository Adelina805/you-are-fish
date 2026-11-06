import { FISH_RADIUS, type FishState } from "@/lib/fish";

/** Pellet radius in canvas pixels (matches unscaled fish on phones). */
export const FOOD_RADIUS = FISH_RADIUS * 0.3;

/** Keep new pellets clear of the fish body and the previous bite. */
const FOOD_SPAWN_CLEARANCE = FISH_RADIUS * 2.5;

/** Extra bottom inset so pellets clear the tracking / Recalibrate chrome. */
const FOOD_BOTTOM_UI_INSET = 72;

/** Extra right inset so pellets clear the bottom-right tracking button. */
const FOOD_RIGHT_UI_INSET = 64;

const SPAWN_ATTEMPTS = 24;

export type Food = {
  x: number;
  y: number;
};

/** Optional exclusion for the top-right debug HUD while tracking info is open. */
export type FoodSpawnAvoid = {
  /** Bottom edge of the debug panel (canvas y), from drawDebugOverlay. */
  hudBottom: number;
  /** Left edge of the panel band to avoid (canvas x). */
  left: number;
};

export type FoodSpawnBounds = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

/**
 * Reachable play rect for a pellet center: fish can clamp onto it, and chrome
 * (tracking button, Recalibrate, optional debug HUD) stays clear.
 */
export function foodSpawnBounds(
  width: number,
  height: number,
  avoid: FoodSpawnAvoid | null = null,
): FoodSpawnBounds | null {
  const edge = FISH_RADIUS + FOOD_RADIUS;
  let minX = edge;
  let maxX = width - edge - FOOD_RIGHT_UI_INSET;
  let minY = edge;
  let maxY = height - edge - FOOD_BOTTOM_UI_INSET;

  if (avoid && avoid.hudBottom > 0) {
    // Prefer the lower play area when the HUD would leave a usable strip;
    // otherwise shrink the right side so pellets stay left of the panel.
    const belowHud = avoid.hudBottom + FOOD_RADIUS;
    if (belowHud < maxY - edge) {
      minY = Math.max(minY, belowHud);
    } else {
      maxX = Math.min(maxX, avoid.left - FOOD_RADIUS);
    }
  }

  if (maxX < minX || maxY < minY) {
    return null;
  }
  return { minX, maxX, minY, maxY };
}

export function isFoodInBounds(food: Food, bounds: FoodSpawnBounds): boolean {
  return (
    food.x >= bounds.minX &&
    food.x <= bounds.maxX &&
    food.y >= bounds.minY &&
    food.y <= bounds.maxY
  );
}

function randIn(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function tooClose(
  x: number,
  y: number,
  ax: number,
  ay: number,
  minDist: number,
): boolean {
  return Math.hypot(x - ax, y - ay) < minDist;
}

function farthestCorner(
  bounds: FoodSpawnBounds,
  fish: FishState,
): Food {
  const corners: Food[] = [
    { x: bounds.minX, y: bounds.minY },
    { x: bounds.maxX, y: bounds.minY },
    { x: bounds.minX, y: bounds.maxY },
    { x: bounds.maxX, y: bounds.maxY },
  ];
  let best = corners[0]!;
  let bestDist = -1;
  for (const c of corners) {
    const d = Math.hypot(c.x - fish.x, c.y - fish.y);
    if (d > bestDist) {
      bestDist = d;
      best = c;
    }
  }
  return best;
}

/**
 * Place one pellet in the safe rect, away from the fish and the previous bite.
 * Falls back to the farthest safe-rect corner on tiny viewports.
 */
export function spawnFood(
  width: number,
  height: number,
  fish: FishState,
  previous: Food | null = null,
  avoid: FoodSpawnAvoid | null = null,
): Food | null {
  const bounds = foodSpawnBounds(width, height, avoid);
  if (!bounds) {
    return null;
  }

  for (let i = 0; i < SPAWN_ATTEMPTS; i += 1) {
    const x = randIn(bounds.minX, bounds.maxX);
    const y = randIn(bounds.minY, bounds.maxY);
    if (tooClose(x, y, fish.x, fish.y, FOOD_SPAWN_CLEARANCE)) {
      continue;
    }
    if (
      previous &&
      tooClose(x, y, previous.x, previous.y, FOOD_SPAWN_CLEARANCE)
    ) {
      continue;
    }
    return { x, y };
  }

  return farthestCorner(bounds, fish);
}

/** Circle overlap using the same body radius as fish clamping / on-screen checks. */
export function foodCollected(fish: FishState, food: Food): boolean {
  return Math.hypot(fish.x - food.x, fish.y - food.y) <= FISH_RADIUS + FOOD_RADIUS;
}

export function drawFood(
  ctx: CanvasRenderingContext2D,
  food: Food,
): void {
  const r = FOOD_RADIUS;
  ctx.save();
  ctx.beginPath();
  ctx.arc(food.x, food.y, r, 0, Math.PI * 2);
  ctx.fillStyle = "#f4c430";
  ctx.fill();
  ctx.strokeStyle = "#e8a010";
  ctx.lineWidth = Math.max(1.5, r * 0.2);
  ctx.stroke();
  // Soft highlight so it reads as a pellet, not a flat disc.
  ctx.beginPath();
  ctx.arc(food.x - r * 0.25, food.y - r * 0.25, r * 0.28, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
  ctx.fill();
  ctx.restore();
}
