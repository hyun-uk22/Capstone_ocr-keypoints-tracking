# Capstone OCR + Keypoints + Tracking

A capstone computer-vision project that combines three classic CV
pipelines in a single, easy-to-use framework:

| Module | Technique |
|--------|-----------|
| **OCR** | [EasyOCR](https://github.com/JaidedAI/EasyOCR) – detect and recognise text in any frame |
| **Keypoints** | OpenCV ORB / SIFT – detect and match local features |
| **Tracking** | OpenCV CSRT / KCF / MOSSE – track bounding boxes across frames |

All three modules are wired together in a single `Pipeline` class that
processes video frame-by-frame.

---

## Repository layout

```
Capstone_ocr-keypoints-tracking/
├── main.py              ← command-line demo
├── requirements.txt
├── src/
│   ├── ocr/
│   │   └── detector.py  ← OCRDetector
│   ├── keypoints/
│   │   └── detector.py  ← KeypointDetector
│   ├── tracking/
│   │   └── tracker.py   ← ObjectTracker
│   └── pipeline.py      ← Pipeline (combines all three)
└── tests/
    ├── test_ocr.py
    ├── test_keypoints.py
    ├── test_tracking.py
    └── test_pipeline.py
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note** – EasyOCR can use a CUDA GPU; pass `--gpu` to `main.py`
> to enable it.  CPU-only inference works out of the box.

### 2. Run the demo

Process a video file and write an annotated copy:

```bash
python main.py --input video.mp4 --output annotated.avi --show
```

Use the webcam instead:

```bash
python main.py --webcam --show
```

Disable individual modules:

```bash
python main.py --input video.mp4 --no-ocr --no-tracking --show
```

#### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | – | Input video file (mutually exclusive with `--webcam`) |
| `--webcam` | – | Use webcam as source |
| `--device INT` | `0` | Webcam device index |
| `--output FILE` | – | Write annotated video to this path |
| `--tracker TYPE` | `csrt` | Tracker algorithm: `csrt` / `kcf` / `mosse` |
| `--kp-method TYPE` | `orb` | Keypoint method: `orb` / `sift` |
| `--languages CODES` | `en` | Comma-separated EasyOCR language codes |
| `--gpu` | off | Enable GPU for EasyOCR |
| `--no-ocr` | off | Disable OCR |
| `--no-kp` | off | Disable keypoint detection |
| `--no-tracking` | off | Disable object tracking |
| `--show` | off | Display annotated frames in a window |

---

## Using the Python API

```python
import cv2
from src.ocr.detector import OCRDetector
from src.keypoints.detector import KeypointDetector
from src.tracking.tracker import ObjectTracker
from src.pipeline import Pipeline

# Build sub-modules
ocr     = OCRDetector(languages=["en"], min_confidence=0.6)
kp      = KeypointDetector(method="orb", max_features=500)
tracker = ObjectTracker(tracker_type="csrt")

# Create a unified pipeline
pipeline = Pipeline(
    ocr_detector=ocr,
    keypoint_detector=kp,
    object_tracker=tracker,
    draw_results=True,   # annotate frames in-place
)

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = pipeline.process(frame)
    print(f"OCR: {[r.text for r in result.ocr_results]}")
    print(f"Keypoints: {len(result.keypoint_result.keypoints)}")
    print(f"Tracks: {result.track_results}")

    cv2.imshow("demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## License

MIT – see [LICENSE](LICENSE).
