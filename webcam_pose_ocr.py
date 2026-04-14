"""
YOLO26 Pose Estimation + PaddleOCR 통합 웹캠 실시간 모드
- 웹캠으로 사람 키포인트 검출 + 트래킹
- 각 사람의 바운딩박스 내에서 OCR 수행 (별도 스레드)
- OCR 성공 시 해당 트래킹 대상에 텍스트(ID/이름) 부여
- q 키로 종료
"""

import argparse
import os
import sys
import time
import threading
from queue import Queue

# PaddlePaddle 호환 문제 우회
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR


class PersonTracker:
    """트래킹 ID별로 OCR 결과를 저장·관리 (스레드 안전)"""

    def __init__(self):
        self.labels = {}
        self.lock = threading.Lock()

    def update_label(self, track_id: int, text: str, confidence: float):
        with self.lock:
            existing = self.labels.get(track_id)
            if existing is None or confidence > existing["confidence"]:
                self.labels[track_id] = {
                    "label": text,
                    "confidence": confidence,
                    "last_seen": time.time(),
                }

    def get_label(self, track_id: int) -> str | None:
        with self.lock:
            entry = self.labels.get(track_id)
            if entry is None:
                return None
            if time.time() - entry["last_seen"] > 30:
                del self.labels[track_id]
                return None
            entry["last_seen"] = time.time()
            return entry["label"]

    def cleanup(self, active_ids: set):
        with self.lock:
            expired = [tid for tid in self.labels if tid not in active_ids
                       and time.time() - self.labels[tid]["last_seen"] > 30]
            for tid in expired:
                del self.labels[tid]


class OCRWorker:
    """OCR을 별도 스레드에서 비동기 실행"""

    def __init__(self, ocr_engine, person_tracker, debug=False):
        self.ocr_engine = ocr_engine
        self.person_tracker = person_tracker
        self.debug = debug
        self.queue = Queue(maxsize=1)  # 최신 요청만 유지
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            jobs = self.queue.get()
            if jobs is None:
                break
            for tid, crop in jobs:
                text, conf = self._run_ocr(crop)
                if text:
                    self.person_tracker.update_label(tid, text, conf)
                    print(f"  [OCR] ID:{tid} → \"{text}\" (신뢰도: {conf:.2f})")

    def _run_ocr(self, crop_img) -> tuple[str | None, float]:
        try:
            results = self.ocr_engine.predict(crop_img)
            if not results:
                if self.debug:
                    print("    [DEBUG] OCR 결과 없음")
                return None, 0.0

            best_text = None
            best_score = 0.0
            for res in results:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                if self.debug:
                    for text, score in zip(texts, scores):
                        print(f"    [DEBUG] OCR 원본: \"{text}\" (신뢰도: {score:.3f})")
                for text, score in zip(texts, scores):
                    text = text.strip()
                    if text and score > best_score:
                        best_text = text
                        best_score = score

            if best_text and best_score > 0.15:
                return best_text, best_score
            if self.debug and best_text:
                print(f"    [DEBUG] 신뢰도 미달: \"{best_text}\" ({best_score:.3f})")
            return None, 0.0
        except Exception as e:
            if self.debug:
                print(f"    [DEBUG] OCR 예외: {e}")
            return None, 0.0

    def submit(self, jobs: list[tuple[int, np.ndarray]]):
        """OCR 작업 제출 (이전 작업이 있으면 버리고 최신만 처리)"""
        if not self.queue.full():
            self.queue.put(jobs)


def crop_person_region(frame, box):
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return frame[y1:y2, x1:x2]


def draw_results(frame, boxes, track_ids, person_tracker):
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        tid = int(tid)

        label = person_tracker.get_label(tid)
        if label:
            display_text = f"{label} (ID:{tid})"
            color = (0, 255, 0)
        else:
            display_text = f"ID:{tid}"
            color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, display_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 Pose + PaddleOCR 실시간 웹캠")
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt",
                        help="YOLO pose 모델 (기본: yolo26n-pose.pt)")
    parser.add_argument("--camera", type=int, default=0,
                        help="카메라 번호 (기본: 0)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO 검출 신뢰도 (기본: 0.5)")
    parser.add_argument("--ocr-interval", type=float, default=5.0,
                        help="OCR 실행 간격 초 (기본: 5초)")
    parser.add_argument("--lang", type=str, default="korean",
                        help="OCR 언어 (기본: korean, 한/영/숫자 모두 인식)")
    parser.add_argument("--debug", action="store_true",
                        help="OCR 디버그 로그 출력")
    args = parser.parse_args()

    print("YOLO 모델 로딩...")
    yolo = YOLO(args.model)

    print("PaddleOCR 로딩...")
    ocr_engine = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    person_tracker = PersonTracker()
    ocr_worker = OCRWorker(ocr_engine, person_tracker, debug=args.debug)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"오류: 카메라 {args.camera}를 열 수 없습니다")
        sys.exit(1)

    print(f"모델: {args.model} | 카메라: {args.camera} | OCR 언어: {args.lang}")
    print("종료하려면 q 키를 누르세요")

    frame_count = 0
    fps_time = time.time()
    fps = 0
    last_ocr_time = 0.0
    prev_person_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO pose + 트래킹
        results = yolo.track(
            source=frame,
            persist=True,
            conf=args.conf,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []

        # OCR: 일정 시간마다 또는 검출 인원 변경 시 비동기 실행
        now = time.time()
        current_person_count = len(boxes)
        person_count_changed = current_person_count != prev_person_count
        time_triggered = (now - last_ocr_time) >= args.ocr_interval
        if (time_triggered or person_count_changed) and current_person_count > 0:
            last_ocr_time = now
            prev_person_count = current_person_count
            if args.debug:
                print(f"\n  [DEBUG] 프레임 #{frame_count} | 검출 인원: {len(boxes)}명")
            jobs = []
            for box, tid in zip(boxes, track_ids):
                tid = int(tid)
                if person_tracker.get_label(tid):
                    continue
                crop = crop_person_region(frame, box)
                if crop is None:
                    if args.debug:
                        print(f"    [DEBUG] ID:{tid} 크롭 실패")
                    continue
                if args.debug:
                    h, w = crop.shape[:2]
                    print(f"    [DEBUG] ID:{tid} 크롭 크기: {w}x{h}")
                jobs.append((tid, crop.copy()))
            if jobs:
                ocr_worker.submit(jobs)

        # 오래된 ID 정리
        active_ids = set(int(t) for t in track_ids) if len(track_ids) > 0 else set()
        person_tracker.cleanup(active_ids)

        # 시각화
        annotated = result.plot()
        annotated = draw_results(annotated, boxes, track_ids, person_tracker)

        # FPS 표시
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
        if frame_count >= 30:
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO26 Pose + OCR", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    main()
