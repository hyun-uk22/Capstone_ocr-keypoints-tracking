"""
YOLO26 Pose Estimation 웹캠 실시간 모드
- 웹캠으로 실시간 키포인트 검출
- q 키로 종료
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Pose Estimation 웹캠 실시간 모드")
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt",
                        help="사용할 모델 (기본: yolo26n-pose.pt)")
    parser.add_argument("--camera", type=int, default=0,
                        help="카메라 번호 (기본: 0)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="검출 신뢰도 임계값 (기본: 0.5)")
    args = parser.parse_args()

    model = YOLO(args.model)

    print(f"모델: {args.model}")
    print(f"카메라: {args.camera}")
    print("종료하려면 q 키를 누르세요")

    results = model(
        source=args.camera,
        show=True,
        conf=args.conf,
        stream=True,
    )

    for _ in results:
        pass


if __name__ == "__main__":
    main()
