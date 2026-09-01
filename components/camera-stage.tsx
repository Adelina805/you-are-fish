"use client";

import type {
  DrawingUtils as DrawingUtilsType,
  FaceLandmarker as FaceLandmarkerType,
} from "@mediapipe/tasks-vision";
import { useCallback, useEffect, useRef, useState } from "react";

import { NeutralPoseCalibrator } from "@/lib/calibration";
import { classifyDirection } from "@/lib/direction";
import { estimateHeadPose } from "@/lib/head-pose";
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
const DEBUG = true;

type Status = "idle" | "loading" | "running" | "error";

type MeshStyle = {
  connections: NonNullable<Parameters<DrawingUtilsType["drawConnectors"]>[1]>;
  color: string;
  lineWidth: number;
};

type Session = {
  landmarker: FaceLandmarkerType | null;
  drawing: DrawingUtilsType | null;
  DrawingUtils: (typeof import("@mediapipe/tasks-vision"))["DrawingUtils"] | null;
  meshStyles: MeshStyle[];
  calibrator: NeutralPoseCalibrator;
  smoother: PoseSmoother;
  history: PoseHistory;
  lastTimestamp: number;
  lastFrameTime: number | null;
  fps: number;
  running: boolean;
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
    lastTimestamp: -1,
    lastFrameTime: null,
    fps: 0,
    running: false,
    frameId: 0,
  };
}

async function loadVision(session: Session): Promise<void> {
  const vision = await import("@mediapipe/tasks-vision");
  const fileset = await vision.FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    runningMode: "VIDEO" as const,
    numFaces: 1,
    outputFacialTransformationMatrixes: true,
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
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [calibrating, setCalibrating] = useState(false);

  const stop = useCallback(() => {
    const session = sessionRef.current;
    session.running = false;
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

  const loop = useCallback(() => {
    const session = sessionRef.current;
    if (!session.running) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = session.landmarker;
    if (!video || !canvas || !landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      session.frameId = requestAnimationFrame(loop);
      return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width === 0 || height === 0) {
      session.frameId = requestAnimationFrame(loop);
      return;
    }

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      session.drawing = null;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      session.frameId = requestAnimationFrame(loop);
      return;
    }

    const now = performance.now();
    if (session.lastFrameTime !== null) {
      const delta = (now - session.lastFrameTime) / 1000;
      if (delta > 0) {
        session.fps = Math.round(1 / delta);
      }
    }
    session.lastFrameTime = now;

    ctx.save();
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height);
    ctx.restore();

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

    if (faceLandmarks && session.DrawingUtils) {
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

    drawCalibrationPrompt(ctx, width, height, session.calibrator);
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
    );
    if (DEBUG) {
      drawPoseSignalViz(ctx, width, vizPose, session.history, hudBottom);
    }

    session.frameId = requestAnimationFrame(loop);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setStatus("loading");
    sessionRef.current = createSession();
    const session = sessionRef.current;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const video = videoRef.current;
      if (!video) {
        throw new Error("Video element missing");
      }
      video.srcObject = stream;
      await video.play();

      await loadVision(session);
      session.running = true;
      setStatus("running");
      session.frameId = requestAnimationFrame(loop);
    } catch (caught) {
      stop();
      const message =
        caught instanceof Error ? caught.message : "Could not start the camera.";
      setError(message);
      setStatus("error");
    }
  }, [loop, stop]);

  const onCalibrate = useCallback(() => {
    const session = sessionRef.current;
    session.calibrator.start();
    session.history.clear();
    session.smoother.reset();
    setCalibrating(true);
  }, []);

  useEffect(() => {
    return () => stop();
  }, [stop]);

  return (
    <div className="relative flex h-dvh w-full items-center justify-center overflow-hidden bg-[#061018]">
      <video ref={videoRef} className="hidden" playsInline muted autoPlay />
      <canvas ref={canvasRef} className="h-full w-full object-contain" />

      {status !== "running" ? (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-4 px-6 text-center">
          <h1 className="font-serif text-5xl tracking-wide text-[#d7ecf5] sm:text-6xl">
            You Are Fish
          </h1>
          <p className="max-w-md text-[#9ec3d4]">
            This live page uses your webcam in the browser. Nothing is uploaded;
            head pose stays on this device.
          </p>
          {error ? <p className="max-w-md text-sm text-red-300">{error}</p> : null}
          <button
            type="button"
            onClick={start}
            disabled={status === "loading"}
            className="rounded-full bg-[#3778dc] px-6 py-3 text-white transition hover:bg-[#4b8aee] disabled:opacity-60"
          >
            {status === "loading" ? "Starting camera…" : "Enable camera"}
          </button>
        </div>
      ) : null}

      {status === "running" && !calibrating ? (
        <button
          type="button"
          onClick={onCalibrate}
          className="absolute top-4 left-1/2 z-10 -translate-x-1/2 rounded-md border-2 border-[#5aa0ff] bg-[#3778dc] px-5 py-2 text-lg text-white"
        >
          Calibrate
        </button>
      ) : null}
    </div>
  );
}
