import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch  # noqa: F401
except Exception:
    pass

import cv2
import paddle
import paddleocr
from paddleocr import PaddleOCR

from src.ocr.processor import (
    OCR_EXPECTED_DIGITS,
    OCR_MIN_SCORE,
    _build_ocr_variants,
    _normalize_digit_text,
    _run_ocr_attempts,
    _select_digit_roi,
)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_enable_pir_api": False})


print("====================================")
print("OCR standalone test")
print("Show a number card to the camera and press SPACE.")
print("Press q to quit.")
print("====================================")
print(f"PaddleOCR {getattr(paddleocr, '__version__', 'unknown')} from {getattr(paddleocr, '__file__', 'unknown')}")

try:
    ocr_engine = PaddleOCR(lang="en", use_angle_cls=False, enable_mkldnn=False, show_log=False)
except Exception:
    ocr_engine = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)


def test_ocr(img):
    h, w = img.shape[:2]
    print(f"\n--- captured crop: {w}x{h} ---")
    print(f"expected digits: {OCR_EXPECTED_DIGITS}, min score: {OCR_MIN_SCORE:.2f}")

    variants = _build_ocr_variants(_select_digit_roi(img))
    ocr_img = variants[0][1]
    cv2.imshow("OCR Input", ocr_img)
    for name, variant in variants[1:]:
        cv2.imshow(f"OCR {name}", variant)

    print("running OCR...")
    candidates = _run_ocr_attempts(ocr_engine, variants)
    print(f"candidates: {candidates}")

    best_text, best_score = None, 0.0
    for source, text, score in candidates:
        only_numbers = _normalize_digit_text(text)
        if only_numbers and score >= OCR_MIN_SCORE:
            if score > best_score:
                best_text = only_numbers
                best_score = score
    if best_text:
        print(f"recognized: {best_text!r} (score: {best_score:.2f})")
    else:
        print("no text recognized")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("camera not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam (SPACE: OCR, q: quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half_size = 150
            y1, y2 = max(0, cy - half_size), min(h, cy + half_size)
            x1, x2 = max(0, cx - half_size), min(w, cx + half_size)
            test_ocr(frame[y1:y2, x1:x2])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
