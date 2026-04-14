"""
YOLO26 Pose Estimation 가중치별 비교 테스트
- Kinetics 비디오에 대해 yolo26 n/s/m/l/x 5개 가중치로 pose estimation 수행
- 각 가중치별 추론 속도, 키포인트 검출 결과를 비교
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

# 테스트할 가중치 목록 (최초 실행 시 자동 다운로드됨)
POSE_MODELS = [
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
]

# COCO 17 키포인트 이름
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def test_single_model(model_name: str, video_path: str, save_dir: Path) -> dict:
    """단일 가중치로 비디오 pose estimation 수행"""
    print(f"\n{'='*60}")
    print(f"모델: {model_name}")
    print(f"{'='*60}")

    model = YOLO(model_name)

    start = time.time()
    results = model(
        source=video_path,
        save=True,
        project=str(save_dir),
        name=Path(model_name).stem,
        stream=True,  # 메모리 절약을 위해 스트리밍
    )

    frame_count = 0
    total_persons = 0
    for r in results:
        frame_count += 1
        if r.keypoints is not None:
            total_persons += len(r.keypoints)

    elapsed = time.time() - start

    stats = {
        "model": model_name,
        "frames": frame_count,
        "total_persons_detected": total_persons,
        "total_time_sec": round(elapsed, 2),
        "fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
    }

    print(f"  프레임 수: {stats['frames']}")
    print(f"  검출 인원(누적): {stats['total_persons_detected']}")
    print(f"  총 소요 시간: {stats['total_time_sec']}s")
    print(f"  처리 FPS: {stats['fps']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Pose Estimation 가중치별 비교 테스트")
    parser.add_argument("--video", type=str, required=True, help="테스트할 비디오 파일 경로")
    parser.add_argument("--models", nargs="+", default=None,
                        help="테스트할 모델 (예: yolo26n-pose.pt yolo26s-pose.pt). 미지정 시 전체 테스트")
    parser.add_argument("--save-dir", type=str, default="results", help="결과 저장 디렉토리")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"오류: 비디오 파일을 찾을 수 없습니다 - {video_path}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    models = args.models if args.models else POSE_MODELS

    print(f"비디오: {video_path}")
    print(f"테스트 모델: {models}")
    print(f"결과 저장: {save_dir}")

    all_stats = []
    for model_name in models:
        stats = test_single_model(model_name, str(video_path), save_dir)
        all_stats.append(stats)

    # 결과 요약
    print(f"\n{'='*60}")
    print("결과 요약")
    print(f"{'='*60}")
    print(f"{'모델':<22} {'FPS':>8} {'검출 인원':>10} {'소요 시간':>10}")
    print("-" * 54)
    for s in all_stats:
        print(f"{s['model']:<22} {s['fps']:>8} {s['total_persons_detected']:>10} {s['total_time_sec']:>9}s")


if __name__ == "__main__":
    main()
