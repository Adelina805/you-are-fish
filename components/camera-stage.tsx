"use client";

import type {
  DrawingUtils as DrawingUtilsType,
  FaceLandmarker as FaceLandmarkerType,
} from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  createBubbleEmitter,
  drawBubbles,
  emitBubblesContinuous,
  updateBubbles,
  type Bubble,
  type BubbleEmitter,
} from "@/lib/bubbles";
import { NeutralPoseCalibrator } from "@/lib/calibration";
import { classifyDirection } from "@/lib/direction";
import { createFish, drawFish, updateFish, type FishState } from "@/lib/fish";
import { estimateHeadPose } from "@/lib/head-pose";
import { MouthTracker } from "@/lib/mouth";
import {
  drawCalibrationPrompt,
  drawDebugOverlay,
  drawDirectionLabel,
  drawPoseSignalViz,
} from "@/lib/overlay";
import { PoseHistory } from "@/lib/pose-history";
import { PoseSmoother } from "@/lib/smoothing";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const CAMERA_ENABLED_KEY = "you-are-fish:camera-enabled";
const DEBUG = true;

// MediaPipe's WASM binds console.error at init and writes INFO/WARNING logs to
// stderr. Next.js treats those as overlay errors, so drop the known noise first.
const MEDIAPIPE_WASM_LOG_NOISE = [
  "Created TensorFlow Lite XNNPACK delegate for CPU",
  "Sets FaceBlendshapesGraph acceleration to xnnpack by default",
  "OpenGL error checking is disabled",
];

function isMediapipeWasmLogNoise(args: unknown[]): boolean {
  return args.some(
    (arg) =>
      typeof arg === "string" &&
      MEDIAPIPE_WASM_LOG_NOISE.some((noise) => arg.includes(noise)),
  );
}

let mediapipeWasmLogsSilenced = false;

function silenceMediapipeWasmLogs(): void {
  if (mediapipeWasmLogsSilenced) {
    return;
  }
  mediapipeWasmLogsSilenced = true;

  const originalError = console.error.bind(console);
  console.error = (...args: Parameters<typeof console.error>) => {
    if (isMediapipeWasmLogNoise(args)) {
      return;
    }
    originalError(...args);
  };
}

function setCameraEnabledFlag(enabled: boolean): void {
  try {
    if (enabled) {
      localStorage.setItem(CAMERA_ENABLED_KEY, "true");
    } else {
      localStorage.removeItem(CAMERA_ENABLED_KEY);
    }
  } catch {
    // localStorage may be unavailable in private browsing or restricted contexts
  }
}

function hasCameraEnabledFlag(): boolean {
  try {
    return localStorage.getItem(CAMERA_ENABLED_KEY) === "true";
  } catch {
    return false;
  }
}

async function shouldAutoStartCamera(): Promise<boolean> {
  try {
    const result = await navigator.permissions.query({ name: "camera" as PermissionName });
    if (result.state === "granted") {
      return true;
    }
    if (result.state === "denied") {
      return false;
    }
  } catch {
    // Safari and some browsers do not support permissions.query for camera
  }
  return hasCameraEnabledFlag();
}

type Status = "idle" | "loading" | "running" | "error";

type MeshStyle = {
  connections: NonNullable<Parameters<DrawingUtilsType["drawConnectors"]>[1]>;
  color: string;
  lineWidth: number;
};

type SessionMode = "stopped" | "preview" | "running";

type Session = {
  landmarker: FaceLandmarkerType | null;
  drawing: DrawingUtilsType | null;
  DrawingUtils: (typeof import("@mediapipe/tasks-vision"))["DrawingUtils"] | null;
  meshStyles: MeshStyle[];
  calibrator: NeutralPoseCalibrator;
  smoother: PoseSmoother;
  history: PoseHistory;
  mouth: MouthTracker;
  fish: FishState | null;
  bubbles: Bubble[];
  bubbleEmitter: BubbleEmitter;
  lastTimestamp: number;
  lastFrameTime: number | null;
  fps: number;
  mode: SessionMode;
  frameId: number;
};

function createSession(): Session {
  return {
    landmarker: null,
    drawing: null,
    DrawingUtils: null,
    meshStyles: [],
    calibrator: new NeutralPoseCalibrator(),
    smoother: new PoseSmoother(),
    history: new PoseHistory(),
    mouth: new MouthTracker(),
    fish: null,
    bubbles: [],
    bubbleEmitter: createBubbleEmitter(),
    lastTimestamp: -1,
    lastFrameTime: null,
    fps: 0,
    mode: "stopped",
    frameId: 0,
  };
}

function fitCanvasToDisplay(canvas: HTMLCanvasElement): boolean {
  const width = Math.max(1, Math.round(canvas.clientWidth));
  const height = Math.max(1, Math.round(canvas.clientHeight));
  const resized = canvas.width !== width || canvas.height !== height;
  if (resized) {
    canvas.width = width;
    canvas.height = height;
  }
  return resized;
}

function drawMirroredVideo(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
): boolean {
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;
  if (videoWidth === 0 || videoHeight === 0) {
    return false;
  }

  const width = canvas.width;
  const height = canvas.height;
  const scale = Math.max(width / videoWidth, height / videoHeight);
  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;

  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
  ctx.restore();
  return true;
}

async function loadVision(session: Session): Promise<void> {
  silenceMediapipeWasmLogs();
  const vision = await import("@mediapipe/tasks-vision");
  const fileset = await vision.FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    runningMode: "VIDEO" as const,
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
    outputFaceBlendshapes: true,
  };

  try {
    session.landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    });
  } catch {
    session.landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
      ...options,
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    });
  }

  session.DrawingUtils = vision.DrawingUtils;
  const FaceLandmarker = vision.FaceLandmarker;
  const contours =
    FaceLandmarker.FACE_LANDMARKS_CONTOURS ??
    [
      ...(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL ?? []),
      ...(FaceLandmarker.FACE_LANDMARKS_LIPS ?? []),
      ...(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE ?? []),
      ...(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE ?? []),
      ...(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW ?? []),
      ...(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW ?? []),
    ];

  session.meshStyles = [
    {
      connections: FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      color: "#C0C0C070",
      lineWidth: 1,
    },
    { connections: contours, color: "#E0E0E0", lineWidth: 1.5 },
    {
      connections: FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
      color: "#30FF30",
      lineWidth: 1.5,
    },
    {
      connections: FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
      color: "#30FF30",
      lineWidth: 1.5,
    },
  ];
}

export default function CameraStage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<Session>(createSession());
  const showTestingUiRef = useRef(true);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<string | null>(null);
  const [calibrating, setCalibrating] = useState(false);
  const [showTestingUi, setShowTestingUi] = useState(true);

  const stop = useCallback(() => {
    const session = sessionRef.current;
    session.mode = "stopped";
    if (session.frameId) {
      cancelAnimationFrame(session.frameId);
      session.frameId = 0;
    }
    const video = videoRef.current;
    const stream = video?.srcObject;
    if (stream instanceof MediaStream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }
    if (video) {
      video.srcObject = null;
    }
    session.landmarker?.close();
    session.landmarker = null;
  }, []);

  const previewLoop = useCallback(() => {
    const session = sessionRef.current;
    if (session.mode !== "preview") {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      session.frameId = requestAnimationFrame(previewLoop);
      return;
    }

    fitCanvasToDisplay(canvas);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      session.frameId = requestAnimationFrame(previewLoop);
      return;
    }

    drawMirroredVideo(ctx, canvas, video);
    session.frameId = requestAnimationFrame(previewLoop);
  }, []);

  const loop = useCallback(() => {
    const session = sessionRef.current;
    if (session.mode !== "running") {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = session.landmarker;
    if (!video || !canvas || !landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      session.frameId = requestAnimationFrame(loop);
      return;
    }

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      session.frameId = requestAnimationFrame(loop);
      return;
    }

    if (fitCanvasToDisplay(canvas)) {
      session.drawing = null;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      session.frameId = requestAnimationFrame(loop);
      return;
    }

    const now = performance.now();
    let dt = 0;
    if (session.lastFrameTime !== null) {
      dt = (now - session.lastFrameTime) / 1000;
      if (dt > 0) {
        session.fps = Math.round(1 / dt);
      }
      // Cap so a long tab-hide pause does not teleport the fish.
      dt = Math.min(dt, 0.05);
    }
    session.lastFrameTime = now;

    drawMirroredVideo(ctx, canvas, video);

    const width = canvas.width;
    const height = canvas.height;

    let timestamp = now;
    if (timestamp <= session.lastTimestamp) {
      timestamp = session.lastTimestamp + 1;
    }
    session.lastTimestamp = timestamp;

    const result = landmarker.detectForVideo(canvas, timestamp);
    const faceLandmarks = result.faceLandmarks[0];
    const faceDetected = Boolean(faceLandmarks);
    const matrix = result.facialTransformationMatrixes[0];
    const pose = estimateHeadPose(matrix?.data);
    const mouthStatus = session.mouth.update(
      result.faceBlendshapes[0],
      faceDetected,
    );

    const wasComplete = session.calibrator.isComplete;
    const wasActive = session.calibrator.isActive;
    if (session.calibrator.isActive) {
      session.calibrator.update(pose, now / 1000);
    }
    if (wasActive && !session.calibrator.isActive) {
      setCalibrating(false);
    }
    if (!wasComplete && session.calibrator.isComplete) {
      session.history.clear();
      session.smoother.reset();
    }

    const calibratedPose = pose ? session.calibrator.apply(pose) : null;
    const smoothedPose = session.calibrator.isComplete
      ? session.smoother.update(calibratedPose)
      : null;
    const direction =
      session.calibrator.isComplete && faceDetected ? classifyDirection(smoothedPose) : null;
    const vizPose = session.calibrator.isComplete ? calibratedPose : pose;
    if (vizPose) {
      session.history.append(vizPose);
    }

    if (showTestingUiRef.current && faceLandmarks && session.DrawingUtils) {
      if (!session.drawing) {
        session.drawing = new session.DrawingUtils(ctx);
      }
      for (const style of session.meshStyles) {
        session.drawing.drawConnectors(faceLandmarks, style.connections, {
          color: style.color,
          lineWidth: style.lineWidth,
        });
      }
    }

    if (!session.fish) {
      session.fish = createFish(width, height);
    }
    if (session.calibrator.isComplete) {
      updateFish(session.fish, direction, dt, width, height);
    } else {
      updateFish(session.fish, null, 0, width, height);
    }
    if (session.fish) {
      emitBubblesContinuous(
        session.bubbles,
        session.fish,
        mouthStatus.openness,
        dt,
        session.bubbleEmitter,
      );
    }
    updateBubbles(session.bubbles, dt, width, height);
    drawFish(ctx, session.fish);
    drawBubbles(ctx, session.bubbles);

    drawCalibrationPrompt(ctx, width, height, session.calibrator);
    if (showTestingUiRef.current) {
      drawDirectionLabel(
        ctx,
        width,
        height,
        direction,
        session.calibrator.isComplete && !session.calibrator.isActive,
      );
      const hudBottom = drawDebugOverlay(
        ctx,
        width,
        session.fps,
        faceDetected,
        pose,
        calibratedPose,
        smoothedPose,
        session.calibrator,
        mouthStatus,
      );
      if (DEBUG) {
        drawPoseSignalViz(ctx, width, vizPose, session.history, hudBottom);
      }
    }

    session.frameId = requestAnimationFrame(loop);
  }, []);

  const start = useCallback(
    async (options?: { isCancelled?: () => boolean }) => {
      const isCancelled = () => options?.isCancelled?.() ?? false;

      setError(null);
      setStatus("loading");
      sessionRef.current = createSession();
      const session = sessionRef.current;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (isCancelled()) {
          for (const track of stream.getTracks()) {
            track.stop();
          }
          return;
        }

        const video = videoRef.current;
        if (!video) {
          throw new Error("Video element missing");
        }
        video.srcObject = stream;
        await video.play();

        session.mode = "preview";
        session.frameId = requestAnimationFrame(previewLoop);

        await loadVision(session);
        if (isCancelled()) {
          stop();
          return;
        }

        setCameraEnabledFlag(true);
        session.mode = "running";
        setStatus("running");
        session.frameId = requestAnimationFrame(loop);
      } catch (caught) {
        if (isCancelled()) {
          return;
        }
        stop();
        if (caught instanceof DOMException && caught.name === "NotAllowedError") {
          setCameraEnabledFlag(false);
        }
        const message =
          caught instanceof Error ? caught.message : "Could not start the camera.";
        setError(message);
        setStatus("error");
      }
    },
    [loop, previewLoop, stop],
  );

  const onCalibrate = useCallback(() => {
    const session = sessionRef.current;
    session.calibrator.start();
    session.history.clear();
    session.smoother.reset();
    setCalibrating(true);
  }, []);

  const onToggleTestingUi = useCallback(() => {
    setShowTestingUi((prev) => {
      const next = !prev;
      showTestingUiRef.current = next;
      return next;
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      if (await shouldAutoStartCamera()) {
        await start({ isCancelled: () => cancelled });
        return;
      }
      if (!cancelled) {
        setStatus("idle");
      }
    })();

    return () => {
      cancelled = true;
      stop();
    };
  }, [start, stop]);

  return (
    <div className="fixed inset-0 overflow-hidden bg-black">
      <video ref={videoRef} className="hidden" playsInline muted autoPlay />
      <canvas ref={canvasRef} className="absolute inset-0 block h-full w-full" />

      {status === "idle" || status === "error" ? (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-4 bg-[#061018] px-6 text-center">
          <h1 className="font-serif text-4xl tracking-wide text-[#d7ecf5] sm:text-5xl md:text-6xl">
            You Are Fish
          </h1>
          <p className="max-w-md text-[#9ec3d4]">
            This live page uses your webcam in the browser. Nothing is uploaded;
            head pose stays on this device.
          </p>
          {error ? <p className="max-w-md text-sm text-red-300">{error}</p> : null}
          <button
            type="button"
            onClick={() => start()}
            className="rounded-full bg-[#3778dc] px-6 py-3 text-white transition hover:bg-[#4b8aee]"
          >
            Enable camera
          </button>
        </div>
      ) : null}

      {status === "running" ? (
        <button
          type="button"
          onClick={onToggleTestingUi}
          className="absolute top-3 left-3 z-10 rounded-md border-2 border-[#5aa0ff] bg-[#3778dc] px-3 py-1.5 text-sm text-white sm:top-4 sm:left-4 sm:px-5 sm:py-2 sm:text-lg"
        >
          {showTestingUi ? "Hide UI" : "Show UI"}
        </button>
      ) : null}

      {status === "running" && !calibrating ? (
        <button
          type="button"
          onClick={onCalibrate}
          className="absolute top-3 left-1/2 z-10 -translate-x-1/2 rounded-md border-2 border-[#5aa0ff] bg-[#3778dc] px-3 py-1.5 text-sm text-white sm:top-4 sm:px-5 sm:py-2 sm:text-lg"
        >
          Calibrate
        </button>
      ) : null}
    </div>
  );
}
