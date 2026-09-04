# You Are Fish

Webcam computer-vision prototype. Live at [you-are-fish.vercel.app](https://you-are-fish.vercel.app). Click **Enable camera**, then **Calibrate**. Head pose is computed on your device; frames are not uploaded.

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Milestone 5: Directional classifier

After calibration, calibrated yaw/pitch are smoothed with an exponential moving average, then classified into a continuous 2D look vector in screen space. Inside the dead zone the UI shows `CENTER`; outside it shows a direction arrow plus unit coordinates `(x, y)` and angle. No fish yet.

The look vector lives in [`lib/direction.ts`](lib/direction.ts) as `LookDirection`:

- `x`, `y`: unit vector; `+x` = look toward screen right, `+y` = look up
- `angleDeg`: `atan2(y, x)` in degrees (`0°` = right, `90°` = up); `null` at center
- `magnitude`: `0` in the dead zone; `hypot(nx, ny) - 1` outside it (for later fish speed)

Thresholds:

- `YAW_DEAD_ZONE_DEG` (default `8`)
- `PITCH_DEAD_ZONE_DEG` (default `8`)

Smoothing strength is in [`lib/smoothing.ts`](lib/smoothing.ts) as `SMOOTHING_ALPHA` (default `0.25`).

### What is a dead zone?

A **dead zone** is an ellipse around the calibrated origin where the classifier reports `CENTER` even though yaw/pitch are not exactly zero. Pose is normalized as `nx = -yaw / yawDeadZone`, `ny = pitch / pitchDeadZone`. With default thresholds, if `hypot(nx, ny) <= 1`, you are considered looking at the screen. Outside the ellipse the unit vector `(nx, ny) / r` is the continuous look direction, so diagonals are first-class (no cardinal snap).

### Why it matters

A still head is never a perfect `(0, 0)`. MediaPipe jitter, breathing, micro-adjustments, and sitting slightly off your calibration pose all produce small angle changes. Without a dead zone, those motions would flicker between directions and would accidentally steer a fish later. The dead zone is the "I am looking at the screen" region; only a deliberate turn should leave it. An ellipse (not an axis-aligned box) means a 45° glance crosses the threshold at the same intent as a pure left/right glance.

### How to choose thresholds experimentally

The defaults are starting guesses, not a one-shot tune:

1. **Calibrate**, then sit still for 10–15 seconds. Watch the HUD `Sm yaw` / `Sm pitch` line. Note the noise envelope (e.g. yaw within ±2°, pitch within ±3°). The dead zone must be **larger** than that envelope, with a little margin.
2. Make the **smallest turn you want to count** as a look (any angle, including diagonals). Record typical peak smoothed angles. The threshold must sit **below** that gesture, or you will have to over-turn.
3. Set each axis between those two bounds: `noise_ceiling < dead_zone < intentional_gesture`. Yaw and pitch often differ, so keep separate constants.
4. Re-test: still → stays `CENTER`; slow glance → arrow tracks continuously; return to rest → `CENTER` again. If it flickers at the edge, raise the dead zone slightly. If you must crane your neck, lower it.
5. Repeat after changing chair, camera height, or distance.

## Milestone 4: Per-user calibration

Click **Calibrate** when you are ready. The app then asks you to look comfortably straight at the screen, collects head-pose samples for about 1.5 seconds, and computes a neutral yaw/pitch baseline. Later measurements subtract that baseline so zero means your comfortable rest pose, not a fixed canonical face. Click **Calibrate** again any time to redo it.

The HUD shows:

- raw yaw/pitch
- calibrated yaw/pitch (after baseline subtraction)
- the calculated neutral baseline

With `DEBUG = true`, the pose signal viz uses calibrated values after calibration completes.

### Why calibrate per user?

MediaPipe's facial transformation matrix is a Procrustes fit to a canonical 3D face, so its Euler angles are relative to that model, not to "this person looking at this screen." A global zero of `(0, 0)` would assume identical camera geometry, seating posture, anatomy, and session setup for everyone. In practice, laptop webcams sit below eye level, people sit differently, and face shape varies. Per-user calibration makes zero mean "looking comfortably at the screen right now," which is what later directional controls will need.

## Milestone 3: Head orientation

Open the webcam in the browser, detect one face, and estimate head orientation from MediaPipe Face Landmarker’s facial transformation matrix.

The matrix is a weighted Procrustes alignment of the canonical 3D face to the detected face (scale + rotation + translation). The app recovers the rotation with SVD and converts it to intrinsic Tait-Bryan YXZ angles. Values are shown in degrees; directional classification (Milestone 5) is applied only after calibration.

Sign convention (after correcting for the mirrored selfie canvas):

- Positive yaw: looking left
- Positive pitch: looking up
- Positive roll: tilting clockwise from the camera’s view

With `DEBUG = true`, a top-right debug panel shows raw yaw (horizontal bar) and pitch (vertical bar) with a zero crosshair, current marker, and recent unsmoothed history trail.

The Face Landmarker model is loaded in the browser from MediaPipe’s CDN.

## Milestone 2: One-face Face Landmarker

Open the webcam, detect one face with MediaPipe Tasks Face Landmarker, and draw the face mesh over the live feed.

## Milestone 1: Webcam feed

Open the webcam and display the live feed in the browser.

### LATER:

Facial expressions:
 mouth open → bubbles
 cheek puff → inflate
 eyebrows → ???

Hand gestures -> currents / whirlpools

Fish movement -> particle force field -> fluid-like ocean

Session behavior -> Fishsona parameters -> persistent aquarium

movement feel → one facial action → real fish/avatar → fluid response → guided game → Fishsona → persistence/database last.