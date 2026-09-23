import * as ort from "onnxruntime-web";

import { CELL_PHONE_CLASS_ID, cocoLabel } from "@/lib/object-detection/coco-labels";
import { nonMaxSuppression } from "@/lib/object-detection/nms";

/** Served from Next `public/`. */
export const DEFAULT_PHONE_MODEL_URL = "/models/yolov8n.onnx";

export type Detection = {
  box: { x: number; y: number; width: number; height: number };
  classId: number;
  className: string;
  score: number;
};

export type PhoneDetectorConfig = {
  /** Min class score to keep a proposal (after decode, before NMS). */
  scoreThreshold: number;
  /** IoU threshold for greedy NMS when `nmsEnabled` is true. */
  iouThreshold: number;
  /** When false, `detections` matches score-filtered `candidates`. */
  nmsEnabled: boolean;
  /** Square YOLO input size (Ultralytics default 640). */
  inputSize: number;
  modelUrl: string;
  maxDetections: number;
};

export type PhoneDetectResult = {
  /** After phone-class + score filter, before NMS. */
  candidates: Detection[];
  /** After NMS (or copy of candidates if NMS disabled). */
  detections: Detection[];
};

export const DEFAULT_PHONE_DETECTOR_CONFIG: PhoneDetectorConfig = {
  scoreThreshold: 0.45,
  iouThreshold: 0.45,
  nmsEnabled: true,
  inputSize: 640,
  modelUrl: DEFAULT_PHONE_MODEL_URL,
  maxDetections: 20,
};

type LetterboxMeta = {
  scale: number;
  padX: number;
  padY: number;
  srcWidth: number;
  srcHeight: number;
};

let ortWasmConfigured = false;

function configureOrtWasm(): void {
  if (ortWasmConfigured) {
    return;
  }
  ortWasmConfigured = true;
  // Avoid SharedArrayBuffer / COOP-COEP requirements in Next/Vercel.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
}

/**
 * Client-side YOLOv8n COCO detector focused on the cell-phone class.
 * Preprocess is synchronous so callers can snapshot camera-only pixels
 * before yielding to async `session.run`.
 */
export class PhoneDetector {
  private readonly session: ort.InferenceSession;
  private readonly config: PhoneDetectorConfig;
  private readonly letterboxCanvas: HTMLCanvasElement;
  private readonly floatBuffer: Float32Array;
  private readonly inputName: string;

  private constructor(
    session: ort.InferenceSession,
    config: PhoneDetectorConfig,
    inputName: string,
  ) {
    this.session = session;
    this.config = config;
    this.inputName = inputName;
    this.letterboxCanvas = document.createElement("canvas");
    this.letterboxCanvas.width = config.inputSize;
    this.letterboxCanvas.height = config.inputSize;
    this.floatBuffer = new Float32Array(3 * config.inputSize * config.inputSize);
  }

  static async create(
    overrides: Partial<PhoneDetectorConfig> = {},
  ): Promise<PhoneDetector> {
    configureOrtWasm();
    const config: PhoneDetectorConfig = {
      ...DEFAULT_PHONE_DETECTOR_CONFIG,
      ...overrides,
    };

    const session = await ort.InferenceSession.create(config.modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });

    const inputName = session.inputNames[0];
    if (!inputName) {
      throw new Error("YOLOv8 ONNX model has no inputs");
    }

    return new PhoneDetector(session, config, inputName);
  }

  getConfig(): Readonly<PhoneDetectorConfig> {
    return this.config;
  }

  /**
   * Runs detection on a canvas that must currently show only the camera frame.
   * Pixel read + letterbox are sync; ONNX inference is async.
   */
  async detect(source: HTMLCanvasElement): Promise<PhoneDetectResult> {
    const { tensor, meta } = this.preprocess(source);
    const feeds: Record<string, ort.Tensor> = { [this.inputName]: tensor };
    const results = await this.session.run(feeds);
    const output = results[this.session.outputNames[0]];
    if (!output) {
      throw new Error("YOLOv8 ONNX model returned no output");
    }

    const candidates = this.decodePhoneCandidates(
      output.data as Float32Array,
      output.dims,
      meta,
    );
    const detections = nonMaxSuppression(candidates, {
      iouThreshold: this.config.iouThreshold,
      enabled: this.config.nmsEnabled,
      maxDetections: this.config.maxDetections,
    });

    return { candidates, detections };
  }

  async close(): Promise<void> {
    await this.session.release();
  }

  /**
   * Letterbox `source` into a 640×640 RGB float tensor in [0, 1], CHW layout.
   * Must stay synchronous (no await) so the rAF loop can finish reading
   * camera-only pixels before drawing overlays.
   */
  private preprocess(source: HTMLCanvasElement): {
    tensor: ort.Tensor;
    meta: LetterboxMeta;
  } {
    const inputSize = this.config.inputSize;
    const srcWidth = source.width;
    const srcHeight = source.height;
    if (srcWidth <= 0 || srcHeight <= 0) {
      throw new Error("Phone detector source canvas has zero size");
    }

    const scale = Math.min(inputSize / srcWidth, inputSize / srcHeight);
    const drawWidth = srcWidth * scale;
    const drawHeight = srcHeight * scale;
    const padX = (inputSize - drawWidth) / 2;
    const padY = (inputSize - drawHeight) / 2;

    const ctx = this.letterboxCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("Could not get 2d context for letterbox canvas");
    }

    ctx.fillStyle = "rgb(114, 114, 114)";
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(source, padX, padY, drawWidth, drawHeight);

    const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
    const { data } = imageData;
    const buf = this.floatBuffer;
    const plane = inputSize * inputSize;
    for (let i = 0; i < plane; i += 1) {
      const px = i * 4;
      buf[i] = data[px]! / 255;
      buf[plane + i] = data[px + 1]! / 255;
      buf[2 * plane + i] = data[px + 2]! / 255;
    }

    const tensor = new ort.Tensor("float32", buf, [1, 3, inputSize, inputSize]);
    return {
      tensor,
      meta: { scale, padX, padY, srcWidth, srcHeight },
    };
  }

  /**
   * Decode YOLOv8 raw output `[1, 84, N]` → phone-class boxes above score threshold.
   * Channels 0–3 are cx, cy, w, h in letterbox pixels; 4–83 are COCO class scores.
   */
  private decodePhoneCandidates(
    data: Float32Array,
    dims: readonly number[],
    meta: LetterboxMeta,
  ): Detection[] {
    const numPreds = dims.length === 3 ? dims[2]! : dims[1] === 84 ? dims[2]! : 0;
    if (!numPreds) {
      throw new Error(`Unexpected YOLO output shape: [${dims.join(", ")}]`);
    }

    const scoreThreshold = this.config.scoreThreshold;
    const classChannel = 4 + CELL_PHONE_CLASS_ID;
    const candidates: Detection[] = [];

    for (let i = 0; i < numPreds; i += 1) {
      const score = data[classChannel * numPreds + i]!;
      if (score < scoreThreshold) {
        continue;
      }

      const cx = data[0 * numPreds + i]!;
      const cy = data[1 * numPreds + i]!;
      const w = data[2 * numPreds + i]!;
      const h = data[3 * numPreds + i]!;

      const box = letterboxBoxToSource(cx, cy, w, h, meta);
      if (box.width <= 1 || box.height <= 1) {
        continue;
      }

      candidates.push({
        box,
        classId: CELL_PHONE_CLASS_ID,
        className: cocoLabel(CELL_PHONE_CLASS_ID),
        score,
      });
    }

    return candidates;
  }
}

function letterboxBoxToSource(
  cx: number,
  cy: number,
  w: number,
  h: number,
  meta: LetterboxMeta,
): Detection["box"] {
  const x1 = (cx - w / 2 - meta.padX) / meta.scale;
  const y1 = (cy - h / 2 - meta.padY) / meta.scale;
  const x2 = (cx + w / 2 - meta.padX) / meta.scale;
  const y2 = (cy + h / 2 - meta.padY) / meta.scale;

  const left = Math.max(0, Math.min(meta.srcWidth, x1));
  const top = Math.max(0, Math.min(meta.srcHeight, y1));
  const right = Math.max(0, Math.min(meta.srcWidth, x2));
  const bottom = Math.max(0, Math.min(meta.srcHeight, y2));

  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}
