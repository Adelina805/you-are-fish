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

export const FISH_RADIUS = 36;

/** Target width of the head cutout on the fish (oval bbox maps to this). */
export const FISH_HEAD_WIDTH = FISH_RADIUS * 1.1;

/** Live face crop + oval mask to place on the fish face (ephemeral). */
export type FishFaceSource = {
  source: CanvasImageSource;
  sx: number;
  sy: number;
  sw: number;
  sh: number;
  /** Face-oval polygon in crop-local pixels; empty → no head sticker. */
  oval: { x: number; y: number }[];
};

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

export type FishFleeSide = "left" | "right";

/** Nearest horizontal screen edge from the fish’s current x. */
export function nearestFleeSide(fish: FishState, width: number): FishFleeSide {
  return fish.x < width / 2 ? "left" : "right";
}

/** True when the fish body is mostly past the flee-side edge. */
export function isFishMostlyOffScreen(
  fish: FishState,
  width: number,
  side: FishFleeSide,
): boolean {
  if (side === "left") {
    return fish.x < -FISH_RADIUS * 0.25;
  }
  return fish.x > width + FISH_RADIUS * 0.25;
}

/** True when the fish center is back inside the normal clamp region. */
export function isFishOnScreen(fish: FishState, width: number, height: number): boolean {
  const r = FISH_RADIUS;
  return fish.x >= r && fish.x <= width - r && fish.y >= r && fish.y <= height - r;
}

/**
 * Drive the fish quickly off the chosen horizontal side (no edge clamp).
 * Vertical motion is damped so exit is mostly horizontal.
 */
export function updateFishFlee(
  fish: FishState,
  side: FishFleeSide,
  dt: number,
  height: number,
): void {
  if (dt <= 0) {
    return;
  }

  const dirX = side === "left" ? -1 : 1;
  // Slightly above normal max so the exit reads as a quick dash.
  const fleeSpeed = FISH_MAX_SPEED * 1.35;
  fish.vx = dirX * fleeSpeed;
  fish.vy *= Math.max(0, 1 - FISH_DRAG * 2 * dt);
  fish.headingX = dirX;
  fish.headingY = 0;
  fish.x += fish.vx * dt;
  fish.y += fish.vy * dt;

  // Keep y loosely in view while fleeing so return y stays sensible.
  const r = FISH_RADIUS;
  if (fish.y < r) {
    fish.y = r;
    fish.vy = 0;
  } else if (fish.y > height - r) {
    fish.y = height - r;
    fish.vy = 0;
  }
}

/**
 * Place the fish just off-screen on `side`, facing inward, ready to swim back.
 */
export function placeFishForReturn(
  fish: FishState,
  side: FishFleeSide,
  width: number,
  height: number,
): void {
  const r = FISH_RADIUS;
  fish.y = Math.min(Math.max(fish.y, r), height - r);
  if (side === "left") {
    fish.x = -r * 1.5;
    fish.headingX = 1;
  } else {
    fish.x = width + r * 1.5;
    fish.headingX = -1;
  }
  fish.headingY = 0;
  fish.vx = fish.headingX * FISH_MAX_SPEED;
  fish.vy = 0;
}

/**
 * Swim back onto the screen from the flee side (no edge clamp until on-screen).
 */
export function updateFishReturn(
  fish: FishState,
  side: FishFleeSide,
  dt: number,
  width: number,
  height: number,
): void {
  if (dt <= 0) {
    return;
  }

  const dirX = side === "left" ? 1 : -1;
  const targetX = side === "left" ? FISH_RADIUS * 2.5 : width - FISH_RADIUS * 2.5;
  const targetY = Math.min(Math.max(fish.y, FISH_RADIUS), height - FISH_RADIUS);

  const dx = targetX - fish.x;
  const dy = targetY - fish.y;
  const dist = Math.hypot(dx, dy) || 1;
  const ux = dx / dist;
  const uy = dy / dist;

  // Bias strongly horizontal so return matches the flee axis.
  const hx = ux * 0.85 + dirX * 0.15;
  const hy = uy * 0.85;
  const hLen = Math.hypot(hx, hy) || 1;

  const speed = FISH_MAX_SPEED * 1.1;
  fish.vx = (hx / hLen) * speed;
  fish.vy = (hy / hLen) * speed;
  fish.headingX = fish.vx / speed;
  fish.headingY = fish.vy / speed;
  fish.x += fish.vx * dt;
  fish.y += fish.vy * dt;
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
  face: FishFaceSource | null = null,
): void {
  const { x, y, headingX, headingY } = fish;
  const len = Math.hypot(headingX, headingY) || 1;
  const ux = headingX / len;
  const uy = headingY / len;
  const px = -uy;
  const py = ux;
  const r = FISH_RADIUS;
  const bodyRx = r * 1.15;
  const bodyRy = r * 0.75;
  const bodyAngle = Math.atan2(uy, ux);

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

  // Solid orange body.
  ctx.beginPath();
  ctx.ellipse(x, y, bodyRx, bodyRy, bodyAngle, 0, Math.PI * 2);
  ctx.fillStyle = "#f0a030";
  ctx.fill();

  const eyeX = x + ux * r * 0.5;
  const eyeY = y + uy * r * 0.5;
  const hasOval =
    face &&
    face.sw > 0 &&
    face.sh > 0 &&
    face.oval.length >= 3;

  if (hasOval && face) {
    let ominX = Infinity;
    let ominY = Infinity;
    let omaxX = -Infinity;
    let omaxY = -Infinity;
    for (const p of face.oval) {
      if (p.x < ominX) ominX = p.x;
      if (p.y < ominY) ominY = p.y;
      if (p.x > omaxX) omaxX = p.x;
      if (p.y > omaxY) omaxY = p.y;
    }
    const ovalW = omaxX - ominX;
    const ovalH = omaxY - ominY;
    if (ovalW > 1 && ovalH > 1) {
      const scale = FISH_HEAD_WIDTH / ovalW;
      const ovalCx = (ominX + omaxX) / 2;
      const ovalCy = (ominY + omaxY) / 2;

      ctx.save();
      ctx.beginPath();
      const first = face.oval[0];
      ctx.moveTo(
        eyeX + (first.x - ovalCx) * scale,
        eyeY + (first.y - ovalCy) * scale,
      );
      for (let i = 1; i < face.oval.length; i += 1) {
        const p = face.oval[i];
        ctx.lineTo(
          eyeX + (p.x - ovalCx) * scale,
          eyeY + (p.y - ovalCy) * scale,
        );
      }
      ctx.closePath();
      ctx.clip();

      ctx.drawImage(
        face.source,
        face.sx,
        face.sy,
        face.sw,
        face.sh,
        eyeX - ovalCx * scale,
        eyeY - ovalCy * scale,
        face.sw * scale,
        face.sh * scale,
      );
      ctx.restore();

      ctx.beginPath();
      ctx.moveTo(
        eyeX + (first.x - ovalCx) * scale,
        eyeY + (first.y - ovalCy) * scale,
      );
      for (let i = 1; i < face.oval.length; i += 1) {
        const p = face.oval[i];
        ctx.lineTo(
          eyeX + (p.x - ovalCx) * scale,
          eyeY + (p.y - ovalCy) * scale,
        );
      }
      ctx.closePath();
      ctx.strokeStyle = "#f0a030";
      ctx.lineWidth = Math.max(1.5, r * 0.05);
      ctx.stroke();
    } else {
      ctx.fillStyle = "#1a1a1a";
      ctx.beginPath();
      ctx.arc(eyeX, eyeY, r * 0.14, 0, Math.PI * 2);
      ctx.fill();
    }
  } else {
    // Eye centered on the forward face when no live head cutout.
    ctx.fillStyle = "#1a1a1a";
    ctx.beginPath();
    ctx.arc(eyeX, eyeY, r * 0.14, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}
