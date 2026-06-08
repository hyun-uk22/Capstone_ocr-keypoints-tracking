#!/usr/bin/env python3
"""Measure how often the current MediaPipe pose model returns keypoints on a video.

This script runs the same pose landmarker .task model used by the Android app
against a local video and reports the frame interval between successful pose
results.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DEFAULT_VIDEO = Path(
    "YTDown_YouTube_Media_Q7AIS0kqDB8_001_1080p (online-video-cutter.com).mp4"
)
DEFAULT_MODEL = Path("app/src/main/assets/pose_landmarker_heavy.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pose keypoint result frame intervals for a video."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--num-poses", type=int, default=5)
    parser.add_argument("--min-detection-confidence", type=float, default=0.32)
    parser.add_argument("--min-presence-confidence", type=float, default=0.30)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.30)
    parser.add_argument(
        "--resize-width",
        type=int,
        default=1280,
        help="Resize frames before inference. Use 0 to keep original video size.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many decoded video frames. 0 means full video.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run inference every N decoded frames.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N inference attempts. Use 0 to disable.",
    )
    return parser.parse_args()


def create_landmarker(args: argparse.Namespace) -> vision.PoseLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(args.model))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def resize_if_needed(frame, resize_width: int):
    if resize_width <= 0 or frame.shape[1] <= resize_width:
        return frame
    scale = resize_width / frame.shape[1]
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (resize_width, height), interpolation=cv2.INTER_AREA)


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * q), len(ordered) - 1)
    return float(ordered[index])


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    decoded_frames = 0
    attempted_frames = 0
    success_frames: list[int] = []
    inference_times_ms: list[float] = []

    started = time.perf_counter()
    with create_landmarker(args) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            decoded_frames += 1
            if args.max_frames and decoded_frames > args.max_frames:
                break
            if (decoded_frames - 1) % args.stride != 0:
                continue

            frame_bgr = resize_if_needed(frame_bgr, args.resize_width)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((decoded_frames - 1) * 1000 / fps) if fps > 0 else decoded_frames

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            inference_times_ms.append((time.perf_counter() - t0) * 1000)
            attempted_frames += 1

            if result.pose_landmarks:
                success_frames.append(decoded_frames)

            if args.progress_every > 0 and attempted_frames % args.progress_every == 0:
                print(
                    "progress: "
                    f"decoded={decoded_frames} "
                    f"attempts={attempted_frames} "
                    f"pose_results={len(success_frames)}",
                    flush=True,
                )

    cap.release()
    elapsed_s = time.perf_counter() - started

    intervals = [
        current - previous for previous, current in zip(success_frames, success_frames[1:])
    ]
    avg_interval = statistics.mean(intervals) if intervals else 0.0
    median_interval = statistics.median(intervals) if intervals else 0.0
    avg_inference_ms = statistics.mean(inference_times_ms) if inference_times_ms else 0.0
    effective_hz = fps / avg_interval if fps > 0 and avg_interval > 0 else 0.0
    attempted_hz = attempted_frames / elapsed_s if elapsed_s > 0 else 0.0

    print("Pose frame interval benchmark")
    print(f"video: {args.video}")
    print(f"model: {args.model}")
    print(f"video_size: {width}x{height}")
    print(f"video_fps: {fps:.3f}")
    print(f"video_total_frames_hint: {total_frames_hint}")
    print(f"decoded_frames: {decoded_frames}")
    print(f"inference_stride: {args.stride}")
    print(f"inference_attempts: {attempted_frames}")
    print(f"pose_success_frames: {len(success_frames)}")
    print(f"pose_success_rate: {len(success_frames) / attempted_frames:.3f}" if attempted_frames else "pose_success_rate: 0")
    print(f"avg_inference_ms: {avg_inference_ms:.2f}")
    print(f"processed_attempts_per_second: {attempted_hz:.2f}")

    if intervals:
        print(f"avg_frames_per_pose_result: {avg_interval:.2f}")
        print(f"median_frames_per_pose_result: {median_interval:.2f}")
        print(f"p90_frames_per_pose_result: {percentile(intervals, 0.90):.2f}")
        print(f"max_frames_between_pose_results: {max(intervals)}")
        print(f"effective_pose_result_hz_by_video_time: {effective_hz:.2f}")
    else:
        print("avg_frames_per_pose_result: n/a")
        print("effective_pose_result_hz_by_video_time: n/a")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
