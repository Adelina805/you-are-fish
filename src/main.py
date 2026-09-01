import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from head_pose import HeadPose, estimate_head_pose

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


def draw_debug_overlay(
    frame,
    fps: int,
    face_detected: bool,
    confidence: float | None,
    width: int,
    height: int,
    pose: HeadPose | None,
) -> None:
    lines = [
        f"Yaw   {_format_angle(None if pose is None else pose.yaw_deg)}",
        f"Pitch {_format_angle(None if pose is None else pose.pitch_deg)}",
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
    font_scale = 0.5
    thickness = 1
    line_height = 18
    x = 8
    y = 18

    for index, line in enumerate(lines):
        baseline_y = y + index * line_height
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

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                frame_start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read frame from camera")

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
                if face_detected:
                    draw_face_landmarks(frame, result.face_landmarks[0])

                frame_height, frame_width = frame.shape[:2]
                draw_debug_overlay(
                    frame,
                    fps=fps,
                    face_detected=face_detected,
                    confidence=get_landmark_confidence(result),
                    width=frame_width,
                    height=frame_height,
                    pose=pose,
                )

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
