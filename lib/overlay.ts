import type { NeutralPoseCalibrator } from "@/lib/calibration";
import {
  PITCH_DEAD_ZONE_DEG,
  YAW_DEAD_ZONE_DEG,
  type LookDirection,
} from "@/lib/direction";
import type { HeadPose } from "@/lib/head-pose";
import {
  MOUTH_CLOSE_THRESHOLD,
  MOUTH_OPEN_THRESHOLD,
  type MouthStatus,
} from "@/lib/mouth";
import type { PoseHistory } from "@/lib/pose-history";

const HUD_MARGIN = 16;
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

/** Shared screen-pixel label size for debug HUD + pose graph. */
function debugLabelFontSize(fit: number): number {
  return Math.max(11, Math.round(14 * Math.max(fit, 0.75)));
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

export function drawDebugOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  fps: number,
  faceDetected: boolean,
  pose: HeadPose | null,
  calibratedPose: HeadPose | null,
  smoothedPose: HeadPose | null,
  calibrator: NeutralPoseCalibrator,
  mouth: MouthStatus | null = null,
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

  const mouthLine = mouth
    ? `Mouth ${mouth.openness.toFixed(2)}  ${mouth.state}`
    : "Mouth n/a";
  const thresholdsLine = `Thr  yaw±${YAW_DEAD_ZONE_DEG}° pitch±${PITCH_DEAD_ZONE_DEG}° mouth ${MOUTH_OPEN_THRESHOLD}/${MOUTH_CLOSE_THRESHOLD}`;
  lines = [
    `FPS   ${fps}`,
    `Face  ${faceDetected ? "yes" : "no"}`,
    `Res   ${width}x${ctx.canvas.height}`,
    mouthLine,
    thresholdsLine,
    ...lines,
  ];

  const height = ctx.canvas.height;
  const fit = Math.min(1, width / 1280, height / 720);
  const fontSize = debugLabelFontSize(fit);
  const pad = Math.max(6, Math.round(HUD_PAD * Math.max(fit, 0.55)));
  const margin = Math.max(8, Math.round(HUD_MARGIN * Math.max(fit, 0.55)));
  const lineHeight = Math.round(fontSize * 1.45);

  ctx.save();
  ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  ctx.textBaseline = "alphabetic";
  const textWidths = lines.map((line) => ctx.measureText(line).width);
  const panelWidth = Math.min(
    Math.max(...textWidths) + pad * 2,
    width - margin * 2,
  );
  const panelHeight = pad + lines.length * lineHeight + pad;
  const panelRight = width - margin;
  const panelLeft = panelRight - panelWidth;
  const panelTop = margin;

  ctx.fillStyle = "rgba(20, 20, 20, 0.55)";
  ctx.fillRect(panelLeft, panelTop, panelWidth, panelHeight);

  ctx.save();
  ctx.beginPath();
  ctx.rect(panelLeft, panelTop, panelWidth, panelHeight);
  ctx.clip();
  lines.forEach((line, index) => {
    const x = panelRight - pad - textWidths[index];
    const y = panelTop + pad + (index + 1) * lineHeight - Math.round(fontSize * 0.28);
    ctx.fillStyle = "#000000";
    ctx.fillText(line, x + 1, y + 1);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(line, x, y);
  });
  ctx.restore();
  ctx.restore();

  return panelTop + panelHeight + margin;
}

function formatSigned(value: number, digits = 2): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

function drawLookArrow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  x: number,
  y: number,
  length: number,
): void {
  // Screen y grows downward; LookDirection.y is look-up, so flip for canvas.
  const tipX = cx + x * length;
  const tipY = cy - y * length;
  const angle = Math.atan2(tipY - cy, tipX - cx);
  const headLen = Math.min(28, length * 0.35);
  const headAngle = Math.PI / 6;

  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(
    tipX - headLen * Math.cos(angle - headAngle),
    tipY - headLen * Math.sin(angle - headAngle),
  );
  ctx.lineTo(
    tipX - headLen * Math.cos(angle + headAngle),
    tipY - headLen * Math.sin(angle + headAngle),
  );
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#000000";
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fill();
}

export function drawDirectionLabel(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  direction: LookDirection | null,
  show: boolean,
): void {
  if (!show) {
    return;
  }

  const space = beginScaledUi(ctx, width, height);
  const cx = space.width / 2;
  const cy = space.height / 2;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  if (!direction) {
    ctx.font = "bold 72px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "#000000";
    ctx.fillText("—", cx + 2, cy + 2);
    ctx.fillStyle = "#787878";
    ctx.fillText("—", cx, cy);
  } else if (direction.magnitude === 0) {
    ctx.font = "bold 72px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "#000000";
    ctx.fillText("CENTER", cx + 2, cy + 2);
    ctx.fillStyle = "#ffffff";
    ctx.fillText("CENTER", cx, cy);
  } else {
    const arrowLen = 110;
    drawLookArrow(ctx, cx, cy - 28, direction.x, direction.y, arrowLen);

    const coords = `${formatSigned(direction.x)}, ${formatSigned(direction.y)}`;
    const angle =
      direction.angleDeg === null
        ? ""
        : `${direction.angleDeg >= 0 ? "+" : ""}${direction.angleDeg.toFixed(0)}°`;

    ctx.font = "bold 36px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#000000";
    ctx.fillText(coords, cx + 2, cy + 72 + 2);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(coords, cx, cy + 72);

    ctx.font = "22px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "#000000";
    ctx.fillText(angle, cx + 1, cy + 108 + 1);
    ctx.fillStyle = "#c8c8c8";
    ctx.fillText(angle, cx, cy + 108);
  }

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

function poseVizFit(width: number, height: number, topOffset: number): number {
  const panelW = YAW_BAR_LEN + PANEL_PAD * 2;
  const panelH = PITCH_BAR_LEN + PANEL_PAD * 2;
  const maxW = Math.min(panelW, width * 0.32);
  const availH = Math.max(64, height - topOffset - PANEL_MARGIN * 2);
  const maxH = Math.min(panelH, height * 0.3, availH);
  return Math.min(1, maxW / panelW, maxH / panelH);
}

export function drawPoseSignalViz(
  ctx: CanvasRenderingContext2D,
  width: number,
  pose: HeadPose | null,
  history: PoseHistory,
  topOffset: number,
): void {
  const height = ctx.canvas.height;
  // Size in screen pixels. HUD text keeps a higher scale floor for readability;
  // this graph was a 288×208 widget and stayed huge on phones under that floor.
  const fit = poseVizFit(width, height, topOffset);
  const yawHalf = (YAW_BAR_LEN * fit) / 2;
  const pitchHalf = (PITCH_BAR_LEN * fit) / 2;
  const pad = Math.max(8, PANEL_PAD * fit);
  const margin = Math.max(10, PANEL_MARGIN * fit);
  const tick = Math.max(4, 6 * fit);
  const zeroArm = Math.max(5, 8 * fit);
  const historyR = Math.max(2, 3 * fit);
  const markerR = Math.max(4, 8 * fit);
  const fontSize = debugLabelFontSize(fit);

  const cornerX = width - margin - yawHalf;
  const cornerY = topOffset + margin + pitchHalf;
  const yawLeft = cornerX - yawHalf;
  const yawRight = cornerX + yawHalf;
  const pitchTop = cornerY - pitchHalf;
  const pitchBottom = cornerY + pitchHalf;

  ctx.save();
  ctx.fillStyle = "rgba(20, 20, 20, 0.55)";
  ctx.fillRect(
    yawLeft - pad,
    pitchTop - pad,
    yawRight - yawLeft + pad * 2,
    pitchBottom - pitchTop + pad * 2,
  );

  ctx.strokeStyle = "rgb(140, 140, 140)";
  ctx.lineWidth = Math.max(1.25, 2 * fit);
  ctx.beginPath();
  ctx.moveTo(yawLeft, cornerY);
  ctx.lineTo(yawRight, cornerY);
  ctx.moveTo(cornerX, pitchTop);
  ctx.lineTo(cornerX, pitchBottom);
  ctx.stroke();

  ctx.strokeStyle = "rgb(220, 220, 220)";
  ctx.beginPath();
  ctx.moveTo(cornerX - zeroArm, cornerY);
  ctx.lineTo(cornerX + zeroArm, cornerY);
  ctx.moveTo(cornerX, cornerY - zeroArm);
  ctx.lineTo(cornerX, cornerY + zeroArm);
  ctx.stroke();

  ctx.strokeStyle = "rgb(140, 140, 140)";
  for (const tickSign of [-1, 1]) {
    const tickX = yawToX(tickSign * YAW_RANGE_DEG, cornerX, yawHalf);
    const tickY = pitchToY(tickSign * PITCH_RANGE_DEG, cornerY, pitchHalf);
    ctx.beginPath();
    ctx.moveTo(tickX, cornerY - tick);
    ctx.lineTo(tickX, cornerY + tick);
    ctx.moveTo(cornerX - tick, tickY);
    ctx.lineTo(cornerX + tick, tickY);
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
    ctx.arc(yawToX(yawSamples[index], cornerX, yawHalf), cornerY, historyR, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cornerX, pitchToY(pitchSamples[index], cornerY, pitchHalf), historyR, 0, Math.PI * 2);
    ctx.fill();
  }

  if (pose) {
    ctx.fillStyle = "rgb(255, 220, 80)";
    ctx.beginPath();
    ctx.arc(yawToX(pose.yawDeg, cornerX, yawHalf), cornerY, markerR, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cornerX, pitchToY(pose.pitchDeg, cornerY, pitchHalf), markerR, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "rgb(200, 200, 200)";
  ctx.font = `${fontSize}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  ctx.fillText("yaw", yawLeft, cornerY + Math.max(6, 10 * fit));
  ctx.fillText("pitch", cornerX + Math.max(6, 10 * fit), pitchTop);
  ctx.fillText("0", cornerX + Math.max(5, 8 * fit), cornerY + Math.max(5, 8 * fit));
  ctx.restore();
}
