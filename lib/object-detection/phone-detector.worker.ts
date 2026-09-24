/// <reference lib="webworker" />

import * as ort from "onnxruntime-web/wasm";

import {
  decodeAndNmsPhoneDetections,
  packRgbFloatChw,
} from "@/lib/object-detection/phone-decode";
import type {
  WorkerInMessage,
  WorkerOutMessage,
  WorkerPhoneConfig,
  WorkerDetectMessage,
  WorkerInitMessage,
  WorkerCloseMessage,
  WorkerReadyMessage,
  WorkerResultMessage,
  WorkerErrorMessage,
} from "@/lib/object-detection/phone-worker-messages";

declare const self: DedicatedWorkerGlobalScope;

let session: ort.InferenceSession | null = null;
let config: WorkerPhoneConfig | null = null;
let inputName = "";
let floatBuffer: Float32Array | null = null;
let letterboxCanvas: OffscreenCanvas | null = null;
let ortConfigured = false;

function configureOrtWasm(): void {
  if (ortConfigured) {
    return;
  }
  ortConfigured = true;
  // Avoid SharedArrayBuffer / COOP-COEP requirements in Next/Vercel.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
}

function resolveModelUrl(modelUrl: string): string {
  if (/^https?:\/\//i.test(modelUrl)) {
    return modelUrl;
  }
  return new URL(modelUrl, self.location.origin).href;
}

async function handleInit(message: WorkerInitMessage): Promise<void> {
  configureOrtWasm();
  if (session) {
    await session.release().catch(() => {
      /* ignore */
    });
    session = null;
  }

  config = message.config;
  const created = await ort.InferenceSession.create(
    resolveModelUrl(config.modelUrl),
    {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    },
  );

  const name = created.inputNames[0];
  if (!name) {
    await created.release().catch(() => {
      /* ignore */
    });
    throw new Error("YOLOv8 ONNX model has no inputs");
  }

  session = created;
  inputName = name;
  const plane = config.inputSize * config.inputSize;
  floatBuffer = new Float32Array(3 * plane);
  letterboxCanvas = new OffscreenCanvas(config.inputSize, config.inputSize);

  const out: WorkerReadyMessage = { type: "ready", requestId: message.requestId };
  self.postMessage(out);
}

async function handleDetect(message: WorkerDetectMessage): Promise<void> {
  if (!session || !config || !floatBuffer || !letterboxCanvas) {
    message.bitmap.close();
    throw new Error("Phone detector worker is not initialized");
  }

  const inputSize = config.inputSize;
  const ctx = letterboxCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    message.bitmap.close();
    throw new Error("Could not get 2d context in phone detector worker");
  }

  try {
    ctx.drawImage(message.bitmap, 0, 0, inputSize, inputSize);
  } finally {
    message.bitmap.close();
  }

  const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
  packRgbFloatChw(imageData.data, inputSize, floatBuffer);

  const tensor = new ort.Tensor("float32", floatBuffer, [
    1,
    3,
    inputSize,
    inputSize,
  ]);
  const feeds: Record<string, ort.Tensor> = { [inputName]: tensor };
  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  if (!output) {
    throw new Error("YOLOv8 ONNX model returned no output");
  }

  const { candidates, detections } = decodeAndNmsPhoneDetections(
    output.data as Float32Array,
    output.dims,
    message.meta,
    {
      scoreThreshold: config.scoreThreshold,
      iouThreshold: config.iouThreshold,
      nmsEnabled: config.nmsEnabled,
      maxDetections: config.maxDetections,
    },
  );

  const out: WorkerResultMessage = {
    type: "result",
    requestId: message.requestId,
    candidates,
    detections,
  };
  self.postMessage(out);
}

async function handleClose(message: WorkerCloseMessage): Promise<void> {
  if (session) {
    await session.release().catch(() => {
      /* ignore */
    });
    session = null;
  }
  config = null;
  floatBuffer = null;
  letterboxCanvas = null;
  inputName = "";
  const out: WorkerReadyMessage = { type: "ready", requestId: message.requestId };
  self.postMessage(out);
}

self.onmessage = (event: MessageEvent<WorkerInMessage>) => {
  const message = event.data;
  void (async () => {
    try {
      if (message.type === "init") {
        await handleInit(message);
        return;
      }
      if (message.type === "detect") {
        await handleDetect(message);
        return;
      }
      if (message.type === "close") {
        await handleClose(message);
      }
    } catch (caught) {
      const errOut: WorkerErrorMessage = {
        type: "error",
        requestId: message.requestId,
        message:
          caught instanceof Error ? caught.message : "Phone detector worker failed",
      };
      self.postMessage(errOut);
    }
  })();
};
