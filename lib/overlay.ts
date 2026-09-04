import type { NeutralPoseCalibrator } from "@/lib/calibration";
import type { Direction } from "@/lib/direction";
import type { HeadPose } from "@/lib/head-pose";
import type { PoseHistory } from "@/lib/pose-history";

const DEBUG = true;

const HUD_MARGIN = 16;
const HUD_LINE_HEIGHT = 28;
const HUD_PAD = 10;

const YAW_RANGE_DEG = 45;
const PITCH_RANGE_DEG = 45;
const YAW_BAR_LEN = 260;
const PITCH_BAR_LEN = 180;
const PANEL_MARGIN = 20;
const PANEL_PAD = 14;

/**
 * Overlay sizes were authored for a 1280×720 canvas. After matching the canvas
 * to the CSS viewport, those constants are 1:1 with screen pixels — huge in
 * mobile / DevTools device mode (~390px). Scale down on small viewports, but
 * not all the way to 720p-fit (that would make HUD text unreadably small).
 */
export function getUiScale(width: number, height: number): number {
  if (width <= 0 || height <= 0) {
    return 1;
  }
  const fit = Math.min(width / 1280, height / 720);
  if (fit >= 1) {
    return 1;
  }
  return Math.max(fit, Math.min(width / 480, height / 700, 0.9));
}

function beginScaledUi(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
): { width: number; height: number; scale: number } {
  const scale = getUiScale(width, height);
  ctx.save();
  ctx.scale(scale, scale);
  return { width: width / scale, height: height / scale, scale };
}

function formatAngle(degrees: number | null | undefined): string {
  if (degrees === null || degrees === undefined) {
    return "n/a";
  }
  const sign = degrees >= 0 ? "+" : "";
  return `${sign}${degrees.toFixed(1)}`;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  fill: string,
): void {
  ctx.fillStyle = fill;
  ctx.fillRect(x, y, width, height);
}

export function drawDebugOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  fps: number,
  faceDetected: boolean,
  pose: HeadPose | null,
  calibratedPose: HeadPose | null,
  smoothedPose: HeadPose | null,
  calibrator: NeutralPoseCalibrator,
): number {
  const rawYaw = pose?.yawDeg ?? null;
  const rawPitch = pose?.pitchDeg ?? null;
  const calYaw = calibratedPose?.yawDeg ?? null;
  const calPitch = calibratedPose?.pitchDeg ?? null;

  let lines: string[];
  if (calibrator.isActive) {
    lines = [
      "Calibrating...",
      `Yaw   raw ${formatAngle(rawYaw)}`,
      `Pitch raw ${formatAngle(rawPitch)}`,
      `Roll  ${formatAngle(pose?.rollDeg ?? null)}`,
    ];
  } else if (calibrator.isComplete) {
    lines = [
      `Yaw   raw ${formatAngle(rawYaw)}  cal ${formatAngle(calYaw)}`,
      `Pitch raw ${formatAngle(rawPitch)}  cal ${formatAngle(calPitch)}`,
      `Sm    yaw ${formatAngle(smoothedPose?.yawDeg ?? null)}  pitch ${formatAngle(smoothedPose?.pitchDeg ?? null)}`,
      `Neutral yaw ${formatAngle(calibrator.baselineYaw)}  pitch ${formatAngle(calibrator.baselinePitch)}`,
      `Roll  ${formatAngle(pose?.rollDeg ?? null)}`,
    ];
  } else {
    lines = [
      "Not calibrated",
      `Yaw   raw ${formatAngle(rawYaw)}`,
      `Pitch raw ${formatAngle(rawPitch)}`,
      `Roll  ${formatAngle(pose?.rollDeg ?? null)}`,
    ];
  }

  if (DEBUG) {
    lines = [
      `FPS   ${fps}`,
      `Face  ${faceDetected ? "yes" : "no"}`,
      `Res   ${width}x${ctx.canvas.height}`,
      ...lines,
    ];
  }

  const space = beginScaledUi(ctx, width, ctx.canvas.height);
  ctx.font = "16px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "alphabetic";
  const textWidths = lines.map((line) => ctx.measureText(line).width);
  const panelWidth = Math.min(
    Math.max(...textWidths) + HUD_PAD * 2,
    space.width - HUD_MARGIN * 2,
  );
  const panelHeight = HUD_PAD + lines.length * HUD_LINE_HEIGHT + HUD_PAD;
  const panelRight = space.width - HUD_MARGIN;
  const panelLeft = panelRight - panelWidth;
  const panelTop = HUD_MARGIN;

  ctx.fillStyle = "rgba(20, 20, 20, 0.55)";
  ctx.fillRect(panelLeft, panelTop, panelWidth, panelHeight);

  ctx.save();
  ctx.beginPath();
  ctx.rect(panelLeft, panelTop, panelWidth, panelHeight);
  ctx.clip();
  lines.forEach((line, index) => {
    const x = panelRight - HUD_PAD - textWidths[index];
    const y = panelTop + HUD_PAD + (index + 1) * HUD_LINE_HEIGHT - 8;
    ctx.fillStyle = "#000000";
    ctx.fillText(line, x + 1, y + 1);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(line, x, y);
  });
  ctx.restore();

  ctx.restore();
  return (panelTop + panelHeight + HUD_MARGIN) * space.scale;
}

export function drawCalibrationPrompt(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  calibrator: NeutralPoseCalibrator,
): void {
  if (!calibrator.isActive) {
    return;
  }

  const space = beginScaledUi(ctx, width, height);
  const prompt = "Look comfortably straight at the screen";
  const hint = "Hold still...";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "22px ui-sans-serif, system-ui, sans-serif";
  const promptWidth = ctx.measureText(prompt).width;
  ctx.font = "16px ui-sans-serif, system-ui, sans-serif";
  const hintWidth = ctx.measureText(hint).width;
  const panelWidth = Math.min(
    Math.max(promptWidth, hintWidth) + 40,
    space.width - HUD_MARGIN * 2,
  );
  const panelHeight = 84;
  const left = (space.width - panelWidth) / 2;
  const top = space.height / 2 - panelHeight / 2;

  roundRect(ctx, left, top, panelWidth, panelHeight, "rgba(20, 20, 20, 0.65)");
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, panelWidth, panelHeight);
  ctx.clip();
  ctx.fillStyle = "#ffffff";
  ctx.font = "22px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(prompt, space.width / 2, space.height / 2 - 10);
  ctx.fillStyle = "#c8c8c8";
  ctx.font = "16px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(hint, space.width / 2, space.height / 2 + 18);
  ctx.restore();
  ctx.textAlign = "start";
  ctx.restore();
}

export function drawDirectionLabel(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  direction: Direction | null,
  show: boolean,
): void {
  if (!show) {
    return;
  }

  const space = beginScaledUi(ctx, width, height);
  const label = direction ?? "—";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "bold 72px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = "#000000";
  ctx.fillText(label, space.width / 2 + 2, space.height / 2 + 2);
  ctx.fillStyle = direction ? "#ffffff" : "#787878";
  ctx.fillText(label, space.width / 2, space.height / 2);
  ctx.textAlign = "start";
  ctx.restore();
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function yawToX(yawDeg: number, centerX: number, halfLen: number): number {
  const ratio = clamp(yawDeg / YAW_RANGE_DEG, -1, 1);
  return Math.round(centerX - ratio * halfLen);
}

function pitchToY(pitchDeg: number, centerY: number, halfLen: number): number {
  const ratio = clamp(pitchDeg / PITCH_RANGE_DEG, -1, 1);
  return Math.round(centerY - ratio * halfLen);
}

export function drawPoseSignalViz(
  ctx: CanvasRenderingContext2D,
  width: number,
  pose: HeadPose | null,
  history: PoseHistory,
  topOffset: number,
): void {
  const height = ctx.canvas.height;
  const space = beginScaledUi(ctx, width, height);
  const top = topOffset / space.scale;
  const yawHalf = YAW_BAR_LEN / 2;
  const pitchHalf = PITCH_BAR_LEN / 2;
  const cornerX = space.width - PANEL_MARGIN - yawHalf;
  const cornerY = PANEL_MARGIN + top + pitchHalf;
  const yawLeft = cornerX - yawHalf;
  const yawRight = cornerX + yawHalf;
  const pitchTop = cornerY - pitchHalf;
  const pitchBottom = cornerY + pitchHalf;

  ctx.fillStyle = "rgba(20, 20, 20, 0.55)";
  ctx.fillRect(
    yawLeft - PANEL_PAD,
    pitchTop - PANEL_PAD,
    yawRight - yawLeft + PANEL_PAD * 2,
    pitchBottom - pitchTop + PANEL_PAD * 2,
  );

  ctx.strokeStyle = "rgb(140, 140, 140)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(yawLeft, cornerY);
  ctx.lineTo(yawRight, cornerY);
  ctx.moveTo(cornerX, pitchTop);
  ctx.lineTo(cornerX, pitchBottom);
  ctx.stroke();

  ctx.strokeStyle = "rgb(220, 220, 220)";
  ctx.beginPath();
  ctx.moveTo(cornerX - 8, cornerY);
  ctx.lineTo(cornerX + 8, cornerY);
  ctx.moveTo(cornerX, cornerY - 8);
  ctx.lineTo(cornerX, cornerY + 8);
  ctx.stroke();

  ctx.strokeStyle = "rgb(140, 140, 140)";
  for (const tick of [-1, 1]) {
    const tickX = yawToX(tick * YAW_RANGE_DEG, cornerX, yawHalf);
    const tickY = pitchToY(tick * PITCH_RANGE_DEG, cornerY, pitchHalf);
    ctx.beginPath();
    ctx.moveTo(tickX, cornerY - 6);
    ctx.lineTo(tickX, cornerY + 6);
    ctx.moveTo(cornerX - 6, tickY);
    ctx.lineTo(cornerX + 6, tickY);
    ctx.stroke();
  }

  const yawSamples = history.yawSamples();
  const pitchSamples = history.pitchSamples();
  const sampleCount = Math.min(yawSamples.length, pitchSamples.length);
  for (let index = 0; index < sampleCount; index += 1) {
    const fade = (index + 1) / sampleCount;
    const shade = 0.25 + 0.75 * fade;
    ctx.fillStyle = `rgb(${Math.round(80 * shade)}, ${Math.round(200 * shade)}, ${Math.round(180 * shade)})`;
    ctx.beginPath();
    ctx.arc(yawToX(yawSamples[index], cornerX, yawHalf), cornerY, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cornerX, pitchToY(pitchSamples[index], cornerY, pitchHalf), 3, 0, Math.PI * 2);
    ctx.fill();
  }

  if (pose) {
    ctx.fillStyle = "rgb(255, 220, 80)";
    ctx.beginPath();
    ctx.arc(yawToX(pose.yawDeg, cornerX, yawHalf), cornerY, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cornerX, pitchToY(pose.pitchDeg, cornerY, pitchHalf), 8, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "rgb(200, 200, 200)";
  ctx.font = "14px ui-sans-serif, system-ui, sans-serif";
  ctx.textBaseline = "top";
  ctx.fillText("yaw", yawLeft, cornerY + 10);
  ctx.fillText("pitch", cornerX + 10, pitchTop);
  ctx.fillText("0", cornerX + 8, cornerY + 8);
  ctx.restore();
}
