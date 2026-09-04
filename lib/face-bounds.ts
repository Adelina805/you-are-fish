/** Axis-aligned face region in canvas pixels. */
export type FaceBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
};

/** Landmark with normalized x/y in [0, 1] relative to the image. */
export type NormalizedPoint = {
  x: number;
  y: number;
};

const PADDING = 0.12;

/**
 * Derive a padded axis-aligned bounding box from face landmarks.
 * Returns null when landmarks are missing or degenerate.
 */
export function faceBoundsFromLandmarks(
  landmarks: NormalizedPoint[] | undefined | null,
  canvasWidth: number,
  canvasHeight: number,
): FaceBounds | null {
  if (!landmarks?.length || canvasWidth <= 0 || canvasHeight <= 0) {
    return null;
  }

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;

  for (const point of landmarks) {
    if (typeof point.x !== "number" || typeof point.y !== "number") {
      continue;
    }
    if (point.x < minX) minX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.x > maxX) maxX = point.x;
    if (point.y > maxY) maxY = point.y;
  }

  if (maxX <= minX || maxY <= minY) {
    return null;
  }

  const padX = (maxX - minX) * PADDING;
  const padY = (maxY - minY) * PADDING;
  minX = Math.max(0, minX - padX);
  minY = Math.max(0, minY - padY);
  maxX = Math.min(1, maxX + padX);
  maxY = Math.min(1, maxY + padY);

  const x = minX * canvasWidth;
  const y = minY * canvasHeight;
  const width = (maxX - minX) * canvasWidth;
  const height = (maxY - minY) * canvasHeight;

  if (width < 1 || height < 1) {
    return null;
  }

  return { x, y, width, height };
}
