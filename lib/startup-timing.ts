/**
 * Startup performance marks for Enable Camera → Ready.
 * Marks are always recorded; the summary table logs only when DEBUG is true.
 */

const MARK_PREFIX = "yaf:";

export type StartupMark =
  | "enable_click"
  | "gum_done"
  | "vision_import_done"
  | "wasm_fileset_done"
  | "landmarker_ready"
  | "first_detect_done"
  | "calibration_ready"
  | "yolo_import"
  | "yolo_ready";

const MEASURE_PAIRS: Array<[string, StartupMark, StartupMark]> = [
  ["gum", "enable_click", "gum_done"],
  ["vision_import", "gum_done", "vision_import_done"],
  ["wasm_fileset", "vision_import_done", "wasm_fileset_done"],
  ["landmarker_create", "wasm_fileset_done", "landmarker_ready"],
  ["first_detect", "landmarker_ready", "first_detect_done"],
  ["enable_to_ready", "enable_click", "calibration_ready"],
  ["yolo_load", "yolo_import", "yolo_ready"],
];

let landmarkerDelegate: "GPU" | "CPU" | null = null;
let loggedCriticalPathForGeneration = -1;
let loggedYoloForGeneration = -1;

export function markStartup(name: StartupMark): void {
  try {
    performance.mark(`${MARK_PREFIX}${name}`);
  } catch {
    // performance.mark may throw if the name collides in some browsers
  }
}

export function setLandmarkerDelegate(delegate: "GPU" | "CPU"): void {
  landmarkerDelegate = delegate;
}

function measurePair(label: string, start: StartupMark, end: StartupMark): number | null {
  const startName = `${MARK_PREFIX}${start}`;
  const endName = `${MARK_PREFIX}${end}`;
  try {
    const measureName = `${MARK_PREFIX}measure:${label}`;
    performance.clearMeasures(measureName);
    performance.measure(measureName, startName, endName);
    const entries = performance.getEntriesByName(measureName);
    const last = entries[entries.length - 1];
    return last ? Math.round(last.duration) : null;
  } catch {
    return null;
  }
}

/** Log Enable Camera → Ready critical-path stages (once per start generation). */
export function logStartupTiming(debug: boolean, generation: number): void {
  if (!debug || generation === loggedCriticalPathForGeneration) {
    return;
  }
  loggedCriticalPathForGeneration = generation;

  const rows: Array<{ stage: string; ms: number }> = [];
  for (const [label, start, end] of MEASURE_PAIRS) {
    if (label === "yolo_load") {
      continue;
    }
    const ms = measurePair(label, start, end);
    if (ms !== null) {
      rows.push({ stage: label, ms });
    }
  }

  if (rows.length === 0) {
    return;
  }

  console.info(
    `[you-are-fish] startup timing${landmarkerDelegate ? ` (delegate=${landmarkerDelegate})` : ""}`,
    Object.fromEntries(rows.map((row) => [row.stage, `${row.ms}ms`])),
  );
}

/** Log YOLO worker load once it finishes (off the Ready critical path). */
export function logYoloTiming(debug: boolean, generation: number): void {
  if (!debug || generation === loggedYoloForGeneration) {
    return;
  }
  const ms = measurePair("yolo_load", "yolo_import", "yolo_ready");
  if (ms === null) {
    return;
  }
  loggedYoloForGeneration = generation;
  console.info("[you-are-fish] yolo load timing", { yolo_load: `${ms}ms` });
}

/** Clear marks from a previous Enable Camera attempt in this tab. */
export function resetStartupTiming(): void {
  landmarkerDelegate = null;
  loggedCriticalPathForGeneration = -1;
  loggedYoloForGeneration = -1;
  try {
    for (const name of [
      "enable_click",
      "gum_done",
      "vision_import_done",
      "wasm_fileset_done",
      "landmarker_ready",
      "first_detect_done",
      "calibration_ready",
      "yolo_import",
      "yolo_ready",
    ] as const) {
      performance.clearMarks(`${MARK_PREFIX}${name}`);
    }
    for (const [label] of MEASURE_PAIRS) {
      performance.clearMeasures(`${MARK_PREFIX}measure:${label}`);
    }
  } catch {
    // ignore
  }
}
