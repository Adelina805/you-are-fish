import type { Detection, LetterboxMeta } from "@/lib/object-detection/phone-decode";

export type WorkerPhoneConfig = {
  scoreThreshold: number;
  iouThreshold: number;
  nmsEnabled: boolean;
  inputSize: number;
  modelUrl: string;
  maxDetections: number;
};

export type WorkerInitMessage = {
  type: "init";
  requestId: number;
  config: WorkerPhoneConfig;
};

export type WorkerDetectMessage = {
  type: "detect";
  requestId: number;
  bitmap: ImageBitmap;
  meta: LetterboxMeta;
};

export type WorkerCloseMessage = {
  type: "close";
  requestId: number;
};

export type WorkerInMessage =
  | WorkerInitMessage
  | WorkerDetectMessage
  | WorkerCloseMessage;

export type WorkerReadyMessage = {
  type: "ready";
  requestId: number;
};

export type WorkerResultMessage = {
  type: "result";
  requestId: number;
  candidates: Detection[];
  detections: Detection[];
};

export type WorkerErrorMessage = {
  type: "error";
  requestId: number;
  message: string;
};

export type WorkerOutMessage =
  | WorkerReadyMessage
  | WorkerResultMessage
  | WorkerErrorMessage;
