import os
import time
import multiprocessing as mp

import cv2


TORSO_KEYPOINTS = (5, 6, 11, 12)
OCR_DEBUG_DIR = os.environ.get("YOLOKP_OCR_DEBUG_DIR")
OCR_EXPECTED_DIGITS = int(os.environ.get("YOLOKP_OCR_EXPECTED_DIGITS", "3"))
OCR_MIN_SCORE = float(os.environ.get("YOLOKP_OCR_MIN_SCORE", "0.60"))


def _normalize_digit_text(text):
    if text is None:
        return ""

    text = str(text).strip()
    # Common OCR confusions for short number labels.
    table = str.maketrans({
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "I": "1", "l": "1", "|": "1", "!": "1",
        "Z": "2", "z": "2",
        "S": "5", "s": "5",
        "B": "8",
        "G": "6",
    })

    mapped = text.translate(table)
    normalized = "".join(ch for ch in mapped if ch.isdigit())
    if OCR_EXPECTED_DIGITS > 0 and len(normalized) != OCR_EXPECTED_DIGITS:
        return ""
    return normalized


def _select_digit_roi(img):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img

    # If the detected person box is portrait-shaped, crop to upper/middle torso.
    # If it is already wide/short, keep the full crop because YOLO probably saw
    # only the upper body or a close number card.
    if h > w * 1.15:
        x1, x2 = int(w * 0.08), int(w * 0.92)
        y1, y2 = int(h * 0.08), int(h * 0.68)
        return img[y1:y2, x1:x2]

    return img


def _prepare_ocr_image(img):
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side <= 0:
        return img

    target_side = 480
    if max_side < 260:
        scale = min(target_side / max_side, 2.0)
    elif max_side > target_side:
        scale = target_side / max_side
    else:
        scale = 1.0

    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interpolation)

    pad = max(18, int(min(resized.shape[:2]) * 0.08))
    return cv2.copyMakeBorder(
        resized,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )


def _to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _build_ocr_variants(img):
    base = _prepare_ocr_image(img)
    variants = [("base", base)]

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(("contrast", _to_bgr(clahe)))

    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("binary", _to_bgr(otsu)))
    variants.append(("binary_inv", _to_bgr(255 - otsu)))

    return variants


def _read_paddle_results(results):
    candidates = []
    if not results:
        return candidates

    def add_candidate(text, score):
        try:
            candidates.append((str(text), float(score)))
        except Exception:
            pass

    for item in results if isinstance(results, list) else [results]:
        if hasattr(item, "json"):
            try:
                item = item.json() if callable(item.json) else item.json
            except Exception:
                pass

        if isinstance(item, dict):
            data = item.get("res", item)
            texts = data.get("rec_texts", []) or data.get("texts", [])
            scores = data.get("rec_scores", []) or data.get("scores", [])
            for text, score in zip(texts, scores):
                add_candidate(text, score)
            continue

        lines = item
        if isinstance(lines, tuple) and len(lines) == 2:
            add_candidate(lines[0], lines[1])
            continue
        if isinstance(lines, list) and len(lines) == 1 and isinstance(lines[0], list):
            lines = lines[0]
        if isinstance(lines, list):
            for res in lines:
                if isinstance(res, tuple) and len(res) == 2:
                    add_candidate(res[0], res[1])
                    continue
                if isinstance(res, list) and len(res) == 2 and isinstance(res[0], str):
                    add_candidate(res[0], res[1])
                    continue
                if len(res) == 2 and isinstance(res[1], (list, tuple)):
                    text, score = res[1]
                    add_candidate(text, score)

    return candidates


def _run_ocr_attempts(ocr_engine, variants):
    all_candidates = []

    for name, image in variants:
        try:
            results = ocr_engine.ocr(image, cls=False)
            all_candidates.extend((name, text, score) for text, score in _read_paddle_results(results))
        except Exception:
            pass

        try:
            results = ocr_engine.ocr(image, det=False, cls=False)
            all_candidates.extend((f"{name}/rec", text, score) for text, score in _read_paddle_results(results))
        except Exception:
            pass

    return all_candidates


def _save_debug_images(track_id, variants):
    if not OCR_DEBUG_DIR:
        return

    try:
        os.makedirs(OCR_DEBUG_DIR, exist_ok=True)
        stamp = int(time.time() * 1000)
        for name, image in variants:
            path = os.path.join(OCR_DEBUG_DIR, f"{stamp}_id{track_id}_{name}.jpg")
            cv2.imwrite(path, image)
    except Exception:
        pass


def _build_paddle_ocr():
    import paddleocr
    from paddleocr import PaddleOCR

    version = getattr(paddleocr, "__version__", "0")
    location = getattr(paddleocr, "__file__", "unknown")
    print(f"[OCR worker] PaddleOCR {version} from {location}")

    try:
        major = int(version.split(".", 1)[0]) if version[:1].isdigit() else 0
    except Exception:
        major = 0

    if major >= 3:
        print("[OCR warning] PaddleOCR 3.x detected. Run `pip install -r requirements.txt` in the Python 3.10 env.")

    return PaddleOCR(
        lang="en",
        use_angle_cls=False,
        enable_mkldnn=False,
        show_log=False,
    )


def ocr_worker_process(in_q, out_q):
    """OCR worker process kept separate from the FastAPI event loop."""
    import sys

    try:
        import torch  # noqa: F401
    except Exception:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_enable_pir_api"] = "0"

    try:
        import paddle

        paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_enable_pir_api": False})
        ocr_engine = _build_paddle_ocr()
    except Exception as e:
        print(f"[OCR Error] library load failed: {e}", file=sys.stderr)
        return

    print("[OCR worker] engine ready")

    while True:
        track_id = None
        try:
            task = in_q.get()
            if task is None:
                break

            track_id, crop_img, req_time = task
            if crop_img is None or crop_img.size == 0:
                continue

            h, w = crop_img.shape[:2]
            roi = _select_digit_roi(crop_img)
            variants = _build_ocr_variants(roi)
            ocr_img = variants[0][1]
            oh, ow = ocr_img.shape[:2]
            print(f"[OCR worker] ID {track_id} crop received: {w}x{h}, OCR input: {ow}x{oh}, variants: {len(variants)}")
            _save_debug_images(track_id, variants)

            best_text, best_score = None, 0.0
            rejected_texts = []

            start_time = time.time()
            candidates = _run_ocr_attempts(ocr_engine, variants)

            for source, text, score in candidates:
                number_text = _normalize_digit_text(text)
                if number_text and score >= OCR_MIN_SCORE:
                    if score > best_score:
                        best_text = number_text
                        best_score = score
                elif text.strip():
                    rejected_texts.append(f"{source}:{text.strip()}:{score:.2f}")

            if best_text and best_score >= OCR_MIN_SCORE:
                score = float(min(best_score, 1.0))
                elapsed = time.time() - start_time
                print(f"[OCR success] ID {track_id}: {best_text} (score: {score:.2f}, {elapsed:.2f}s)")
                out_q.put((track_id, best_text, score, req_time))
            else:
                elapsed = time.time() - start_time
                sample = ", ".join(rejected_texts[:3])
                reason = f"invalid format, retry ({sample})" if rejected_texts else "no text"
                print(f"[OCR worker] ID {track_id}: {reason} ({elapsed:.2f}s, candidates: {len(candidates)})")
                out_q.put((track_id, None, 0.0, req_time))

        except Exception as e:
            print(f"[OCR process exception] {e}", file=sys.stderr)
            if track_id is not None:
                out_q.put((track_id, None, 0.0, 0))


class OCRManager:
    def __init__(self):
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.process = None
        self.cache = {}
        self.max_pending = 2

    def start(self):
        if self.process is None:
            self.process = mp.Process(
                target=ocr_worker_process,
                args=(self.in_q, self.out_q),
                daemon=True,
            )
            self.process.start()

    def stop(self):
        if self.process:
            self.in_q.put(None)
            self.process.join(timeout=3)
            self.process = None

    def process_results(self):
        while not self.out_q.empty():
            try:
                track_id, text, score, req_time = self.out_q.get_nowait()
                if text and score > 0.15:
                    self.cache[track_id] = {
                        "text": text,
                        "last_seen": time.time(),
                        "processing": False,
                    }
                else:
                    self.cache[track_id] = {
                        "text": "?",
                        "last_seen": time.time() - 28,
                        "processing": False,
                    }
            except Exception:
                break

    def _crop_from_keypoints(self, frame, box, keypoints):
        if keypoints is None:
            return None

        points = []
        for idx in TORSO_KEYPOINTS:
            if idx >= len(keypoints):
                continue
            kp = keypoints[idx]
            x, y = float(kp[0]), float(kp[1])
            conf = float(kp[2]) if len(kp) > 2 else 1.0
            if conf > 0.25 and x > 0 and y > 0:
                points.append((x, y))

        if len(points) < 2:
            return None

        h, w = frame.shape[:2]
        px = [p[0] for p in points]
        py = [p[1] for p in points]
        x1, x2 = min(px), max(px)
        y1, y2 = min(py), max(py)

        bw = max(1.0, float(box[2]) - float(box[0]))
        bh = max(1.0, float(box[3]) - float(box[1]))
        torso_w = max(x2 - x1, bw * 0.35)
        torso_h = max(y2 - y1, bh * 0.28)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        crop_w = torso_w * 1.8
        crop_h = torso_h * 1.8
        x1 = int(max(0, cx - crop_w / 2.0))
        x2 = int(min(w, cx + crop_w / 2.0))
        y1 = int(max(0, cy - crop_h * 0.55))
        y2 = int(min(h, cy + crop_h * 0.75))

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        return frame[y1:y2, x1:x2].copy()

    def get_text(self, frame, box, track_id, keypoints=None):
        now = time.time()
        self.process_results()

        cache_entry = self.cache.get(track_id)
        if cache_entry and (now - cache_entry["last_seen"] < 30):
            if (
                not cache_entry["processing"]
                and cache_entry["text"]
                and cache_entry["text"] != "?"
            ):
                return cache_entry["text"]
            if cache_entry["text"] == "?" and (now - cache_entry["last_seen"] < 2):
                return "?"

        if cache_entry and cache_entry.get("processing"):
            return cache_entry["text"]

        pending = sum(1 for entry in self.cache.values() if entry.get("processing"))
        if pending >= self.max_pending:
            return cache_entry["text"] if cache_entry else None

        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        crop = frame[y1:y2, x1:x2].copy()

        ch, cw = crop.shape[:2]
        print(f"[OCR manager] ID {track_id} queued crop: {cw}x{ch}")

        self.cache[track_id] = {
            "text": "recognizing...",
            "last_seen": now,
            "processing": True,
        }
        self.in_q.put((track_id, crop, now))
        return self.cache[track_id]["text"]

    def clean_cache(self):
        now = time.time()
        expired_ids = [
            k
            for k, v in self.cache.items()
            if now - v["last_seen"] > 60 and not v["processing"]
        ]
        for k in expired_ids:
            del self.cache[k]
