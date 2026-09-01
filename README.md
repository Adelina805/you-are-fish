# You Are Fish

Computer vision prototype for a webcam-based project.

## Milestone 4: Per-user calibration

On startup, click **Calibrate** when you are ready. The app then asks you to look comfortably straight at the screen, collects head-pose samples for about 1.5 seconds, and computes a neutral yaw/pitch baseline. Later measurements subtract that baseline so zero means your comfortable rest pose, not a fixed canonical face. Click **Calibrate** again any time to redo it.

The HUD shows:

- raw yaw/pitch
- calibrated yaw/pitch (after baseline subtraction)
- the calculated neutral baseline

With `DEBUG = True`, the pose signal viz uses calibrated values after calibration completes.

### Why calibrate per user?

MediaPipe's facial transformation matrix is a Procrustes fit to a canonical 3D face, so its Euler angles are relative to that model, not to "this person looking at this screen." A global zero of `(0, 0)` would assume identical camera geometry, seating posture, anatomy, and session setup for everyone. In practice, laptop webcams sit below eye level, people sit differently, and face shape varies. Per-user calibration makes zero mean "looking comfortably at the screen right now," which is what later directional controls will need.

## Milestone 3: Head orientation

Open the default webcam, detect one face, and estimate head orientation from MediaPipe Face Landmarker’s facial transformation matrix.

The matrix is a weighted Procrustes alignment of the canonical 3D face to the detected face (scale + rotation + translation). The app recovers the rotation with SVD and converts it to intrinsic Tait-Bryan YXZ angles. Values are shown in degrees; they are not classified into LEFT/RIGHT/UP/DOWN.

Sign convention (unmirrored webcam):

- Positive yaw: looking left
- Positive pitch: looking up
- Positive roll: tilting clockwise from the camera’s view

With `DEBUG = True`, a top-right debug panel shows raw yaw (horizontal bar) and pitch (vertical bar) with a zero crosshair, current marker, and recent unsmoothed history trail.

On first run, the face landmarker model is downloaded automatically to `models/face_landmarker.task`.

## Milestone 2: One-face Face Landmarker

Open the default webcam, detect one face with MediaPipe Tasks Face Landmarker, draw the face mesh over the live feed, and quit with `Q`.

On first run, the face landmarker model is downloaded automatically to `models/face_landmarker.task`.

## Milestone 1: Webcam feed

Open the default webcam, display the live feed, and quit with `Q`.

### Requirements

- Python 3.10–3.13
- `mediapipe==0.10.35` (pinned; MediaPipe 1.0.x currently crashes on macOS when initializing Face Landmarker)

### Setup

```bash
cd /Users/adelinamartinez/you-are-fish
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
python src/main.py
```

Press `Q` or `q` to exit.

### Deactivate

```bash
deactivate
```

To recreate the environment, delete `.venv` and repeat the setup steps.
