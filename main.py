"""
Entry point for the OCR + Keypoints + Tracking capstone demo.

Usage
-----
Process a video file and write the annotated output::

    python main.py --input video.mp4 --output out.avi

Process a webcam stream (default device 0)::

    python main.py --webcam

Flags
-----
--input FILE        Path to input video file.
--output FILE       Path to write annotated video (optional).
--webcam            Use webcam as input source.
--device INT        Webcam device index (default 0).
--tracker TYPE      Tracker algorithm: csrt | kcf | mosse  (default csrt).
--kp-method TYPE    Keypoint method: orb | sift  (default orb).
--no-ocr            Disable OCR.
--no-kp             Disable keypoint detection.
--no-tracking       Disable object tracking.
--languages CODES   Comma-separated EasyOCR language codes (default en).
--gpu               Enable GPU for EasyOCR inference.
--show              Display annotated frames in a window.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from src.ocr.detector import OCRDetector
from src.keypoints.detector import KeypointDetector
from src.tracking.tracker import ObjectTracker
from src.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OCR + Keypoints + Tracking demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", metavar="FILE", help="Input video file")
    source.add_argument("--webcam", action="store_true", help="Use webcam")

    p.add_argument("--output", metavar="FILE", help="Output video file")
    p.add_argument("--device", type=int, default=0, help="Webcam device index")
    p.add_argument("--tracker", default="csrt", choices=["csrt", "kcf", "mosse"])
    p.add_argument("--kp-method", dest="kp_method", default="orb", choices=["orb", "sift"])
    p.add_argument("--no-ocr", dest="no_ocr", action="store_true")
    p.add_argument("--no-kp", dest="no_kp", action="store_true")
    p.add_argument("--no-tracking", dest="no_tracking", action="store_true")
    p.add_argument("--languages", default="en", help="Comma-separated EasyOCR language codes")
    p.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR")
    p.add_argument("--show", action="store_true", help="Display frames in a window")
    return p


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.webcam:
        cap = cv2.VideoCapture(args.device)
    else:
        cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video source '{args.input or args.device}'")
    return cap


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    ocr = (
        None
        if args.no_ocr
        else OCRDetector(languages=args.languages.split(","), gpu=args.gpu)
    )
    kp = None if args.no_kp else KeypointDetector(method=args.kp_method)
    tracker = None if args.no_tracking else ObjectTracker(tracker_type=args.tracker)
    return Pipeline(
        ocr_detector=ocr,
        keypoint_detector=kp,
        object_tracker=tracker,
        draw_results=True,
    )


def open_writer(
    cap: cv2.VideoCapture, output_path: str
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def main() -> None:
    args = build_parser().parse_args()
    cap = open_capture(args)
    pipeline = build_pipeline(args)

    writer: cv2.VideoWriter | None = None
    if args.output:
        writer = open_writer(cap, args.output)

    print("Processing – press 'q' to quit …")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process(frame)

        n_ocr = len(result.ocr_results)
        n_kp = len(result.keypoint_result.keypoints) if result.keypoint_result else 0
        n_tr = len(result.track_results)
        print(
            f"[frame {frame_idx:05d}]  OCR: {n_ocr}  KP: {n_kp}  tracks: {n_tr}",
            end="\r",
        )

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("Capstone Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    print()  # newline after the carriage-return progress line
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_idx} frame(s).")


if __name__ == "__main__":
    main()
