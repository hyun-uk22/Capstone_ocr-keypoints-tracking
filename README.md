# YOLO Pose + PaddleOCR Red Light Green Light

Webcam-based capstone project for a "Red Light, Green Light" game.

The app runs a FastAPI server, serves `index.html`, receives webcam frames over a
WebSocket, tracks people with YOLO pose, reads player number cards with
PaddleOCR, and marks players eliminated when they move during red light.

## Main Flow

1. Browser opens `http://127.0.0.1:8000`.
2. `index.html` captures webcam frames and sends them to `/ws`.
3. `server.py` decodes each frame.
4. `PoseTracker` runs YOLO pose tracking with persistent track IDs.
5. `OCRManager` crops each detected person bbox and sends it to a separate OCR
   worker process.
6. The OCR worker preprocesses the crop, runs PaddleOCR, and accepts only valid
   three-digit number labels such as `001`, `025`, or `123`.
7. `GameState` updates green/red light state and eliminates players whose pose
   keypoints move too much during red light.
8. The server sends player boxes, keypoints, OCR text, alive state, movement, and
   game state back to the browser.

## Repository Layout

```text
yolokp/
  index.html                 Browser UI and webcam/WebSocket client
  server.py                  FastAPI app and main runtime entry point
  test_ocr.py                Standalone webcam OCR test
  requirements.txt           Python dependencies
  yolo26n-pose.pt            Default YOLO pose model
  src/
    game/
      state.py               Red/green light game state and movement checks
    ocr/
      processor.py           Async PaddleOCR worker and OCR filtering
    pose/
      tracker.py             Ultralytics YOLO pose tracking wrapper
```

## Requirements

- Python 3.10 is recommended.
- A webcam available to the browser.
- The default model file `yolo26n-pose.pt` in the project root.

Install dependencies:

```powershell
pip install -r requirements.txt
```

This project pins PaddleOCR 2.x because the current OCR path is built and tested
against PaddleOCR 2.7.3.

## Run

Start the server:

```powershell
python server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The server starts:

- YOLO pose model loading from `yolo26n-pose.pt`
- FastAPI/Uvicorn on `127.0.0.1:8000`
- A separate PaddleOCR worker process

## OCR Behavior

OCR is run asynchronously so the FastAPI event loop does not block.

Current OCR rules:

- Crop source: YOLO person bbox.
- Preprocessing variants: base image, contrast-enhanced image, binary image, and
  inverted binary image.
- PaddleOCR modes: normal detection/recognition and recognition-only fallback.
- Accepted text format: exactly three digits.
- Examples accepted: `001`, `025`, `123`.
- Examples rejected and retried: `7`, `17`, `1234`, random letters, low-confidence
  text.
- Common OCR confusions are normalized, for example `O01 -> 001` and `O2S -> 025`.

When OCR prints this:

```text
[OCR worker] ID 16: invalid format, retry (...)
```

it means PaddleOCR returned text, but the result did not match the required
three-digit format or confidence threshold. The manager will try OCR again on a
future frame.

## OCR Tuning

The OCR behavior can be adjusted with environment variables before starting the
server.

Minimum OCR confidence:

```powershell
$env:YOLOKP_OCR_MIN_SCORE="0.75"
python server.py
```

Expected digit count:

```powershell
$env:YOLOKP_OCR_EXPECTED_DIGITS="3"
python server.py
```

Save OCR debug images:

```powershell
$env:YOLOKP_OCR_DEBUG_DIR="D:\projects\capstone\yolokp\ocr_debug"
python server.py
```

Debug images show the actual crop variants sent to PaddleOCR and are useful when
OCR repeatedly reads only part of a number.

## Standalone OCR Test

Use `test_ocr.py` to test PaddleOCR without running the full game server:

```powershell
python test_ocr.py
```

Show a number card to the camera and press `SPACE`. The script displays OCR input
variants and prints recognition candidates.

## Runtime Notes

- If Ultralytics installs a missing package such as `lap` during runtime, restart
  the server once after installation.
- OCR runs in a child process. Stop the server with `CTRL+C` so the worker can be
  shut down cleanly.
- If OCR keeps rejecting results, enable `YOLOKP_OCR_DEBUG_DIR` and inspect the
  saved crops first. Most OCR failures come from a bbox where the number card is
  too small, blurred, tilted, partially covered, or outside the person bbox.

## License

MIT. See [LICENSE](LICENSE).
