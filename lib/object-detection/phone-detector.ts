import type {
  Detection,
  LetterboxMeta,
} from "@/lib/object-detection/phone-decode";
import type {
  WorkerInMessage,
  WorkerOutMessage,
  WorkerPhoneConfig,
} from "@/lib/object-detection/phone-worker-messages";

/** Served from Next `public/`. */
export const DEFAULT_PHONE_MODEL_URL = "/models/yolov8n.onnx";

export type { Detection };

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

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
};

/**
 * Main-thread facade for YOLOv8n phone detection.
 * Letterbox snapshot stays sync (camera-only pixels); ONNX runs in a worker.
 */
export class PhoneDetector {
  private readonly worker: Worker;
  private readonly config: PhoneDetectorConfig;
  private readonly letterboxCanvas: HTMLCanvasElement;
  private nextRequestId = 1;
  private readonly pending = new Map<number, PendingRequest>();
  private closed = false;

  private constructor(worker: Worker, config: PhoneDetectorConfig) {
    this.worker = worker;
    this.config = config;
    this.letterboxCanvas = document.createElement("canvas");
    this.letterboxCanvas.width = config.inputSize;
    this.letterboxCanvas.height = config.inputSize;

    this.worker.onmessage = (event: MessageEvent<WorkerOutMessage>) => {
      this.onWorkerMessage(event.data);
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || "Phone detector worker error");
      for (const [, pending] of this.pending) {
        pending.reject(error);
      }
      this.pending.clear();
    };
  }

  static async create(
    overrides: Partial<PhoneDetectorConfig> = {},
  ): Promise<PhoneDetector> {
    const config: PhoneDetectorConfig = {
      ...DEFAULT_PHONE_DETECTOR_CONFIG,
      ...overrides,
    };

    const worker = new Worker(
      new URL("./phone-detector.worker.ts", import.meta.url),
      { type: "module" },
    );

    const detector = new PhoneDetector(worker, config);
    try {
      await detector.postAndWait({
        type: "init",
        requestId: detector.allocRequestId(),
        config: toWorkerConfig(config),
      });
    } catch (caught) {
      worker.terminate();
      throw caught;
    }

    return detector;
  }

  getConfig(): Readonly<PhoneDetectorConfig> {
    return this.config;
  }

  /**
   * Snapshot camera-only pixels into the letterbox canvas (sync), then run
   * inference in the worker. Caller must invoke while the source is still
   * camera-only (before blue wash / overlays).
   */
  async detect(source: HTMLCanvasElement): Promise<PhoneDetectResult> {
    if (this.closed) {
      throw new Error("Phone detector is closed");
    }

    const meta = this.letterboxSnapshot(source);
    const bitmap = await createImageBitmap(this.letterboxCanvas);
    const requestId = this.allocRequestId();

    try {
      const result = await this.postAndWait<PhoneDetectResult>({
        type: "detect",
        requestId,
        bitmap,
        meta,
      }, [bitmap]);
      return result;
    } catch (caught) {
      // If postMessage failed before transfer, close the bitmap ourselves.
      try {
        bitmap.close();
      } catch {
        /* already transferred / closed */
      }
      throw caught;
    }
  }

  async close(): Promise<void> {
    if (this.closed) {
      return;
    }
    this.closed = true;
    try {
      await this.postAndWait({
        type: "close",
        requestId: this.allocRequestId(),
      });
    } catch {
      /* terminate regardless */
    } finally {
      this.worker.terminate();
      for (const [, pending] of this.pending) {
        pending.reject(new Error("Phone detector closed"));
      }
      this.pending.clear();
    }
  }

  /**
   * Draw `source` into the reusable 640×640 letterbox canvas.
   * Must stay synchronous (no await) so the rAF loop can finish reading
   * camera-only pixels before drawing overlays.
   */
  private letterboxSnapshot(source: HTMLCanvasElement): LetterboxMeta {
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

    const ctx = this.letterboxCanvas.getContext("2d");
    if (!ctx) {
      throw new Error("Could not get 2d context for letterbox canvas");
    }

    ctx.fillStyle = "rgb(114, 114, 114)";
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(source, padX, padY, drawWidth, drawHeight);

    return { scale, padX, padY, srcWidth, srcHeight };
  }

  private allocRequestId(): number {
    const id = this.nextRequestId;
    this.nextRequestId += 1;
    return id;
  }

  private postAndWait<T = unknown>(
    message: WorkerInMessage,
    transfer?: Transferable[],
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.pending.set(message.requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      try {
        if (transfer && transfer.length > 0) {
          this.worker.postMessage(message, transfer);
        } else {
          this.worker.postMessage(message);
        }
      } catch (caught) {
        this.pending.delete(message.requestId);
        reject(caught);
      }
    });
  }

  private onWorkerMessage(message: WorkerOutMessage): void {
    const pending = this.pending.get(message.requestId);
    if (!pending) {
      return;
    }
    this.pending.delete(message.requestId);

    if (message.type === "error") {
      pending.reject(new Error(message.message));
      return;
    }
    if (message.type === "result") {
      pending.resolve({
        candidates: message.candidates,
        detections: message.detections,
      } satisfies PhoneDetectResult);
      return;
    }
    // ready (init / close ack)
    pending.resolve(undefined);
  }
}

function toWorkerConfig(config: PhoneDetectorConfig): WorkerPhoneConfig {
  return {
    scoreThreshold: config.scoreThreshold,
    iouThreshold: config.iouThreshold,
    nmsEnabled: config.nmsEnabled,
    inputSize: config.inputSize,
    modelUrl: config.modelUrl,
    maxDetections: config.maxDetections,
  };
}
