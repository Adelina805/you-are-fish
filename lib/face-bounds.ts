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

export type Point2 = {
  x: number;
  y: number;
};

/** MediaPipe-style edge between landmark indices. */
export type LandmarkConnection = {
  start: number;
  end: number;
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

/**
 * Walk face-oval connections into a closed ring of landmark indices.
 */
export function orderOvalIndices(
  connections: LandmarkConnection[] | undefined | null,
): number[] {
  if (!connections?.length) {
    return [];
  }

  const next = new Map<number, number>();
  for (const edge of connections) {
    next.set(edge.start, edge.end);
  }

  const start = connections[0].start;
  const ordered = [start];
  let current = next.get(start);
  while (current !== undefined && current !== start) {
    ordered.push(current);
    current = next.get(current);
    if (ordered.length > connections.length + 1) {
      break;
    }
  }
  return ordered;
}

/**
 * Face-oval polygon in crop-local pixels (relative to bounds / crop canvas).
 * Accounts for rounded crop canvas size vs float bounds.
 */
export function faceOvalPointsInCrop(
  landmarks: NormalizedPoint[] | undefined | null,
  bounds: FaceBounds,
  ovalConnections: LandmarkConnection[] | undefined | null,
  canvasWidth: number,
  canvasHeight: number,
  cropWidth: number,
  cropHeight: number,
): Point2[] {
  const indices = orderOvalIndices(ovalConnections);
  if (!landmarks?.length || indices.length < 3 || bounds.width <= 0 || bounds.height <= 0) {
    return [];
  }

  const scaleX = cropWidth / bounds.width;
  const scaleY = cropHeight / bounds.height;
  const points: Point2[] = [];

  for (const index of indices) {
    const landmark = landmarks[index];
    if (!landmark || typeof landmark.x !== "number" || typeof landmark.y !== "number") {
      continue;
    }
    const canvasX = landmark.x * canvasWidth;
    const canvasY = landmark.y * canvasHeight;
    points.push({
      x: (canvasX - bounds.x) * scaleX,
      y: (canvasY - bounds.y) * scaleY,
    });
  }

  return points.length >= 3 ? points : [];
}
