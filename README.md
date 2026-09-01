# You Are Fish

Computer vision prototype for a webcam-based project.

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
