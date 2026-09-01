import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from calibration import NeutralPoseCalibrator
from direction import Direction, classify_direction
from head_pose import HeadPose, estimate_head_pose
from pose_viz import PoseHistory, draw_pose_signal_viz
from smoothing import PoseSmoother

DEBUG = True

WINDOW_NAME = "You Are Fish"
CAMERA_INDEX = 0
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def draw_face_landmarks(frame, face_landmarks) -> None:
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
    )
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
    )
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
    )


def get_landmark_confidence(result) -> float | None:
    if not result or not result.face_landmarks:
        return None

    presence_values = [
        landmark.presence
        for landmark in result.face_landmarks[0]
        if landmark.presence is not None
    ]
    if not presence_values:
        return None

    return sum(presence_values) / len(presence_values)


def _format_angle(degrees: float | None) -> str:
    if degrees is None:
        return "n/a"
    return f"{degrees:+.1f}"


def get_head_pose(result) -> HeadPose | None:
    if not result or not result.facial_transformation_matrixes:
        return None
    return estimate_head_pose(result.facial_transformation_matrixes[0])


DEBUG_OVERLAY_MARGIN = 16
DEBUG_OVERLAY_FONT_SCALE = 0.75
DEBUG_OVERLAY_THICKNESS = 2
DEBUG_OVERLAY_LINE_HEIGHT = 28
DEBUG_OVERLAY_BACKING_ALPHA = 0.55


def debug_overlay_height(line_count: int) -> int:
    return DEBUG_OVERLAY_MARGIN + line_count * DEBUG_OVERLAY_LINE_HEIGHT + DEBUG_OVERLAY_MARGIN


def draw_debug_overlay(
    frame,
    fps: int,
    face_detected: bool,
    confidence: float | None,
    width: int,
    height: int,
    pose: HeadPose | None,
    calibrated_pose: HeadPose | None,
    smoothed_pose: HeadPose | None,
    calibrator: NeutralPoseCalibrator,
) -> int:
    raw_yaw = None if pose is None else pose.yaw_deg
    raw_pitch = None if pose is None else pose.pitch_deg
    cal_yaw = None if calibrated_pose is None else calibrated_pose.yaw_deg
    cal_pitch = None if calibrated_pose is None else calibrated_pose.pitch_deg

    if calibrator.is_active:
        lines = [
            "Calibrating...",
            f"Yaw   raw {_format_angle(raw_yaw)}",
            f"Pitch raw {_format_angle(raw_pitch)}",
            f"Roll  {_format_angle(None if pose is None else pose.roll_deg)}",
        ]
    elif calibrator.is_complete:
        sm_yaw = None if smoothed_pose is None else smoothed_pose.yaw_deg
        sm_pitch = None if smoothed_pose is None else smoothed_pose.pitch_deg
        lines = [
            f"Yaw   raw {_format_angle(raw_yaw)}  cal {_format_angle(cal_yaw)}",
            f"Pitch raw {_format_angle(raw_pitch)}  cal {_format_angle(cal_pitch)}",
            f"Sm    yaw {_format_angle(sm_yaw)}  pitch {_format_angle(sm_pitch)}",
            (
                "Neutral yaw "
                f"{_format_angle(calibrator.baseline_yaw)}  "
                f"pitch {_format_angle(calibrator.baseline_pitch)}"
            ),
            f"Roll  {_format_angle(None if pose is None else pose.roll_deg)}",
        ]
    else:
        lines = [
            "Not calibrated",
            f"Yaw   raw {_format_angle(raw_yaw)}",
            f"Pitch raw {_format_angle(raw_pitch)}",
            f"Roll  {_format_angle(None if pose is None else pose.roll_deg)}",
        ]

    if DEBUG:
        if confidence is None:
            confidence_text = "n/a"
        else:
            confidence_text = f"{confidence:.2f}"
        lines = [
            f"FPS   {fps}",
            f"Face  {'yes' if face_detected else 'no'}",
            f"Conf  {confidence_text}",
            f"Res   {width}x{height}",
            *lines,
        ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = DEBUG_OVERLAY_FONT_SCALE
    thickness = DEBUG_OVERLAY_THICKNESS
    line_height = DEBUG_OVERLAY_LINE_HEIGHT
    margin = DEBUG_OVERLAY_MARGIN
    pad = 10

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_width = max(size[0] for size in text_sizes) + pad * 2
    panel_height = debug_overlay_height(len(lines)) - DEBUG_OVERLAY_MARGIN + pad
    panel_right = width - margin
    panel_left = panel_right - panel_width
    panel_top = margin
    panel_bottom = panel_top + panel_height

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.addWeighted(
        overlay,
        DEBUG_OVERLAY_BACKING_ALPHA,
        frame,
        1.0 - DEBUG_OVERLAY_BACKING_ALPHA,
        0,
        frame,
    )

    for index, line in enumerate(lines):
        text_width = text_sizes[index][0]
        x = panel_right - pad - text_width
        baseline_y = panel_top + pad + (index + 1) * line_height - 6
        cv2.putText(
            frame,
            line,
            (x + 1, baseline_y + 1),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return panel_bottom + margin


CALIBRATION_PROMPT = "Look comfortably straight at the screen"
CALIBRATION_HINT = "Hold still..."
CALIBRATION_PROMPT_FONT_SCALE = 0.9
CALIBRATION_HINT_FONT_SCALE = 0.65
CALIBRATION_FONT = cv2.FONT_HERSHEY_SIMPLEX
CALIBRATION_THICKNESS = 2


def draw_calibration_prompt(frame, calibrator: NeutralPoseCalibrator) -> None:
    if not calibrator.is_active:
        return

    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width // 2
    center_y = frame_height // 2

    prompt_size, _ = cv2.getTextSize(
        CALIBRATION_PROMPT,
        CALIBRATION_FONT,
        CALIBRATION_PROMPT_FONT_SCALE,
        CALIBRATION_THICKNESS,
    )
    hint_size, _ = cv2.getTextSize(
        CALIBRATION_HINT,
        CALIBRATION_FONT,
        CALIBRATION_HINT_FONT_SCALE,
        CALIBRATION_THICKNESS,
    )

    prompt_x = center_x - prompt_size[0] // 2
    prompt_y = center_y
    hint_x = center_x - hint_size[0] // 2
    hint_y = prompt_y + 36

    overlay = frame.copy()
    panel_left = min(prompt_x, hint_x) - 20
    panel_right = max(prompt_x + prompt_size[0], hint_x + hint_size[0]) + 20
    panel_top = prompt_y - prompt_size[1] - 20
    panel_bottom = hint_y + 20
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (20, 20, 20),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        CALIBRATION_PROMPT,
        (prompt_x, prompt_y),
        CALIBRATION_FONT,
        CALIBRATION_PROMPT_FONT_SCALE,
        (255, 255, 255),
        CALIBRATION_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        CALIBRATION_HINT,
        (hint_x, hint_y),
        CALIBRATION_FONT,
        CALIBRATION_HINT_FONT_SCALE,
        (200, 200, 200),
        CALIBRATION_THICKNESS,
        cv2.LINE_AA,
    )


CALIBRATE_BUTTON_LABEL = "Calibrate"
CALIBRATE_BUTTON_MARGIN = 16
CALIBRATE_BUTTON_PAD_X = 18
CALIBRATE_BUTTON_PAD_Y = 10
CALIBRATE_BUTTON_FONT_SCALE = 0.75
CALIBRATE_BUTTON_THICKNESS = 2
CALIBRATE_BUTTON_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CALIBRATE_BUTTON_FILL = (55, 120, 220)
_CALIBRATE_BUTTON_BORDER = (90, 160, 255)
_CALIBRATE_BUTTON_TEXT = (255, 255, 255)


def draw_calibrate_button(frame, calibrator: NeutralPoseCalibrator) -> tuple[int, int, int, int] | None:
    if calibrator.is_active:
        return None

    text_size, baseline = cv2.getTextSize(
        CALIBRATE_BUTTON_LABEL,
        CALIBRATE_BUTTON_FONT,
        CALIBRATE_BUTTON_FONT_SCALE,
        CALIBRATE_BUTTON_THICKNESS,
    )
    button_width = text_size[0] + CALIBRATE_BUTTON_PAD_X * 2
    button_height = text_size[1] + baseline + CALIBRATE_BUTTON_PAD_Y * 2
    left = (frame.shape[1] - button_width) // 2
    top = CALIBRATE_BUTTON_MARGIN
    right = left + button_width
    bottom = top + button_height

    cv2.rectangle(frame, (left, top), (right, bottom), _CALIBRATE_BUTTON_FILL, thickness=-1)
    cv2.rectangle(frame, (left, top), (right, bottom), _CALIBRATE_BUTTON_BORDER, thickness=2)

    text_x = left + CALIBRATE_BUTTON_PAD_X
    text_y = bottom - CALIBRATE_BUTTON_PAD_Y - baseline
    cv2.putText(
        frame,
        CALIBRATE_BUTTON_LABEL,
        (text_x, text_y),
        CALIBRATE_BUTTON_FONT,
        CALIBRATE_BUTTON_FONT_SCALE,
        _CALIBRATE_BUTTON_TEXT,
        CALIBRATE_BUTTON_THICKNESS,
        cv2.LINE_AA,
    )
    return left, top, right, bottom


DIRECTION_FONT = cv2.FONT_HERSHEY_SIMPLEX
DIRECTION_FONT_SCALE = 2.0
DIRECTION_THICKNESS = 3
DIRECTION_COLOR = (255, 255, 255)
DIRECTION_SHADOW_COLOR = (0, 0, 0)
DIRECTION_INACTIVE_COLOR = (120, 120, 120)


def draw_direction_label(
    frame,
    direction: Direction | None,
    *,
    show: bool,
) -> None:
    if not show:
        return

    frame_height, frame_width = frame.shape[:2]
    label = "—" if direction is None else direction.value
    color = DIRECTION_INACTIVE_COLOR if direction is None else DIRECTION_COLOR

    text_size, baseline = cv2.getTextSize(
        label,
        DIRECTION_FONT,
        DIRECTION_FONT_SCALE,
        DIRECTION_THICKNESS,
    )
    x = (frame_width - text_size[0]) // 2
    y = frame_height // 2 + text_size[1] // 2

    cv2.putText(
        frame,
        label,
        (x + 2, y + 2),
        DIRECTION_FONT,
        DIRECTION_FONT_SCALE,
        DIRECTION_SHADOW_COLOR,
        DIRECTION_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (x, y),
        DIRECTION_FONT,
        DIRECTION_FONT_SCALE,
        color,
        DIRECTION_THICKNESS,
        cv2.LINE_AA,
    )


def main() -> None:
    model_path = ensure_model()

    latest_result = {"value": None}

    def on_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        del output_image, timestamp_ms
        latest_result["value"] = result

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        output_facial_transformation_matrixes=True,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

    start_time = time.monotonic()
    last_timestamp_ms = -1
    last_frame_time = None
    fps = 0
    pose_history = PoseHistory()
    calibrator = NeutralPoseCalibrator()
    pose_smoother = PoseSmoother()
    ui_state = {
        "button_rect": None,
        "calibrator": calibrator,
        "pose_history": pose_history,
        "pose_smoother": pose_smoother,
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        button_rect = ui_state["button_rect"]
        if button_rect is None:
            return

        left, top, right, bottom = button_rect
        if left <= x <= right and top <= y <= bottom:
            ui_state["calibrator"].start()
            ui_state["pose_history"].clear()
            ui_state["pose_smoother"].reset()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                frame_start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read frame from camera")
                frame = cv2.flip(frame, 1)

                if last_frame_time is not None:
                    frame_delta = frame_start - last_frame_time
                    if frame_delta > 0:
                        fps = int(round(1 / frame_delta))
                last_frame_time = frame_start

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                landmarker.detect_async(mp_image, timestamp_ms)

                result = latest_result["value"]
                face_detected = bool(result and result.face_landmarks)
                pose = get_head_pose(result)

                was_complete = calibrator.is_complete
                if calibrator.is_active:
                    calibrator.update(pose, frame_start)
                if not was_complete and calibrator.is_complete:
                    pose_history.clear()
                    pose_smoother.reset()

                calibrated_pose = calibrator.apply(pose) if pose is not None else None
                smoothed_pose = (
                    pose_smoother.update(calibrated_pose)
                    if calibrator.is_complete
                    else None
                )
                direction = (
                    classify_direction(smoothed_pose)
                    if calibrator.is_complete and face_detected
                    else None
                )
                viz_pose = calibrated_pose if calibrator.is_complete else pose
                if viz_pose is not None:
                    pose_history.append(viz_pose)
                if face_detected:
                    draw_face_landmarks(frame, result.face_landmarks[0])

                frame_height, frame_width = frame.shape[:2]
                draw_calibration_prompt(frame, calibrator)
                draw_direction_label(
                    frame,
                    direction,
                    show=calibrator.is_complete and not calibrator.is_active,
                )
                ui_state["button_rect"] = draw_calibrate_button(frame, calibrator)
                hud_bottom = draw_debug_overlay(
                    frame,
                    fps=fps,
                    face_detected=face_detected,
                    confidence=get_landmark_confidence(result),
                    width=frame_width,
                    height=frame_height,
                    pose=pose,
                    calibrated_pose=calibrated_pose,
                    smoothed_pose=smoothed_pose,
                    calibrator=calibrator,
                )
                if DEBUG:
                    draw_pose_signal_viz(frame, viz_pose, pose_history, top_offset=hud_bottom)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
