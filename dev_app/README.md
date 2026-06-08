# Freeze Dev

Developer-only Android app for measuring MediaPipe pose keypoint update time on a gallery video.

## Build

```powershell
.\gradlew.bat :dev_app:assembleDebug
```

## Install

```powershell
adb install -r dev_app\build\outputs\apk\debug\dev_app-debug.apk
```

This installs separately from the main app because it uses package name:

```text
com.example.freeze.dev
```

The launcher name is:

```text
Freeze Dev
```

## Use

1. Open `Freeze Dev`.
2. Tap `Select Gallery Video`.
3. Choose a video from the phone gallery or file picker.
4. Tap `Run Benchmark`.
5. Wait for the result summary.

## Output

`avg_pose_update_ms` is the average time for one `PoseLandmarker.detectForVideo(...)` call.

`theoretical_pose_updates_per_sec` is calculated as:

```text
1000 / avg_pose_update_ms
```

The benchmark measures pose model inference time only. It does not include the main app's camera capture, OCR, game logic, Compose rendering, or screen recording overhead.
