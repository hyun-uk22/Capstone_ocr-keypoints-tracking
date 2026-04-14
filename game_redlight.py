"""
무궁화 꽃이 피었습니다 게임
- YOLO26 Pose Estimation + PaddleOCR 기반
- 키포인트 움직임 감지로 탈락 판정
- 키 조작: s=게임시작, r=리셋, q=종료
"""

import argparse
import os
import sys
import time
import threading
import random
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


# ── 게임 상태 ──
GAME_WAITING = "waiting"      # 시작 대기
GAME_GREEN = "green_light"    # 움직여도 됨 ("무궁화 꽃이~" 말하는 중)
GAME_RED = "red_light"        # 멈춰야 함 (말 끝남)
GAME_OVER = "game_over"       # 게임 종료

# COCO 17 키포인트 중 움직임 감지에 사용할 주요 관절
MOTION_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# 5,6=양어깨 7,8=양팔꿈치 9,10=양손목 11,12=양엉덩이 13,14=양무릎 15,16=양발목


class GameState:
    """게임 상태 관리"""

    def __init__(self, move_threshold=15.0, green_duration=(3.0, 6.0), red_duration=(2.0, 4.0)):
        self.state = GAME_WAITING
        self.move_threshold = move_threshold
        self.green_duration = green_duration  # (최소, 최대) 초
        self.red_duration = red_duration
        self.state_start_time = 0.0
        self.current_duration = 0.0

        # 플레이어 상태: {track_id: {"alive": bool, "prev_kpts": np.array}}
        self.players = {}
        self.lock = threading.Lock()

    def start(self):
        """게임 시작"""
        with self.lock:
            self.state = GAME_GREEN
            self.state_start_time = time.time()
            self.current_duration = random.uniform(*self.green_duration)
            # 모든 플레이어 생존 상태로 초기화
            for pid in self.players:
                self.players[pid]["alive"] = True
                self.players[pid]["prev_kpts"] = None
            print("\n[게임] 시작! 무궁화 꽃이 피었습니다~")

    def reset(self):
        """게임 리셋"""
        with self.lock:
            self.state = GAME_WAITING
            self.players.clear()
            print("\n[게임] 리셋 완료. s 키로 다시 시작하세요.")

    def update_state(self):
        """시간 경과에 따라 그린↔레드 전환"""
        with self.lock:
            if self.state not in (GAME_GREEN, GAME_RED):
                return

            elapsed = time.time() - self.state_start_time
            if elapsed >= self.current_duration:
                if self.state == GAME_GREEN:
                    # 그린 → 레드: "피었습니다!" → 멈춰!
                    self.state = GAME_RED
                    self.current_duration = random.uniform(*self.red_duration)
                    self.state_start_time = time.time()
                    # 현재 키포인트를 기준점으로 저장
                    for pid in self.players:
                        if self.players[pid]["alive"] and self.players[pid].get("current_kpts") is not None:
                            self.players[pid]["prev_kpts"] = self.players[pid]["current_kpts"].copy()
                    print("[게임] 레드 라이트! 움직이면 탈락!")

                elif self.state == GAME_RED:
                    # 레드 → 그린: 다시 움직여도 됨
                    self.state = GAME_GREEN
                    self.current_duration = random.uniform(*self.green_duration)
                    self.state_start_time = time.time()
                    # 기준점 초기화
                    for pid in self.players:
                        self.players[pid]["prev_kpts"] = None
                    print("[게임] 그린 라이트! 무궁화 꽃이 피었습니다~")

                    # 생존자 확인 → 게임 종료 체크
                    alive_count = sum(1 for p in self.players.values() if p["alive"])
                    if len(self.players) > 0 and alive_count == 0:
                        self.state = GAME_OVER
                        print("[게임] 게임 오버! 모든 플레이어 탈락!")

    def update_player_keypoints(self, track_id: int, keypoints: np.ndarray):
        """플레이어 키포인트 업데이트 및 움직임 감지"""
        with self.lock:
            if track_id not in self.players:
                self.players[track_id] = {
                    "alive": True if self.state == GAME_WAITING else True,
                    "prev_kpts": None,
                    "current_kpts": None,
                    "movement": 0.0,
                }

            player = self.players[track_id]
            if not player["alive"]:
                return

            player["current_kpts"] = keypoints

            # 레드 라이트에서만 움직임 감지
            if self.state == GAME_RED and player["prev_kpts"] is not None:
                movement = self._calc_movement(player["prev_kpts"], keypoints)
                player["movement"] = movement
                if movement > self.move_threshold:
                    player["alive"] = False
                    print(f"[게임] ID:{track_id} 탈락! (움직임: {movement:.1f})")

    def _calc_movement(self, prev_kpts: np.ndarray, curr_kpts: np.ndarray) -> float:
        """주요 키포인트 간 평균 이동거리 계산"""
        total = 0.0
        count = 0
        for idx in MOTION_KEYPOINTS:
            if idx >= len(prev_kpts) or idx >= len(curr_kpts):
                continue
            px, py = prev_kpts[idx][:2]
            cx, cy = curr_kpts[idx][:2]
            # 키포인트가 검출되지 않은 경우 (0,0) 무시
            if (px == 0 and py == 0) or (cx == 0 and cy == 0):
                continue
            dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            total += dist
            count += 1
        return total / count if count > 0 else 0.0

    def get_player_status(self, track_id: int) -> dict | None:
        with self.lock:
            return self.players.get(track_id)

    def get_alive_count(self) -> int:
        with self.lock:
            return sum(1 for p in self.players.values() if p["alive"])

    def get_total_count(self) -> int:
        with self.lock:
            return len(self.players)


class PersonTracker:
    """트래킹 ID별로 OCR 결과를 저장·관리"""

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
        self.queue = Queue(maxsize=1)
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
                return None, 0.0
            best_text = None
            best_score = 0.0
            for res in results:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                if self.debug:
                    for text, score in zip(texts, scores):
                        print(f"    [DEBUG] OCR: \"{text}\" ({score:.3f})")
                for text, score in zip(texts, scores):
                    text = text.strip()
                    if text and score > best_score:
                        best_text = text
                        best_score = score
            if best_text and best_score > 0.15:
                return best_text, best_score
            return None, 0.0
        except Exception as e:
            if self.debug:
                print(f"    [DEBUG] OCR 예외: {e}")
            return None, 0.0

    def submit(self, jobs):
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


def draw_game_ui(frame, game: GameState, boxes, track_ids, person_tracker):
    """게임 UI 그리기"""
    h, w = frame.shape[:2]

    # ── 상단 게임 상태 배너 ──
    if game.state == GAME_WAITING:
        banner_color = (128, 128, 128)
        banner_text = "[WAITING] Press 's' to START"
    elif game.state == GAME_GREEN:
        banner_color = (0, 180, 0)
        elapsed = time.time() - game.state_start_time
        remaining = max(0, game.current_duration - elapsed)
        banner_text = f"[GREEN LIGHT] Move! ({remaining:.1f}s)"
    elif game.state == GAME_RED:
        banner_color = (0, 0, 220)
        elapsed = time.time() - game.state_start_time
        remaining = max(0, game.current_duration - elapsed)
        banner_text = f"[RED LIGHT] FREEZE! ({remaining:.1f}s)"
    elif game.state == GAME_OVER:
        banner_color = (0, 0, 0)
        banner_text = "[GAME OVER] Press 'r' to RESET"

    cv2.rectangle(frame, (0, 0), (w, 50), banner_color, -1)
    cv2.putText(frame, banner_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # ── 생존자 수 표시 ──
    alive = game.get_alive_count()
    total = game.get_total_count()
    if total > 0:
        score_text = f"Alive: {alive}/{total}"
        cv2.putText(frame, score_text, (w - 200, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── 각 플레이어 표시 ──
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = map(int, box)
        tid = int(tid)

        player = game.get_player_status(tid)
        label = person_tracker.get_label(tid)
        name = label if label else f"ID:{tid}"

        if player and not player["alive"]:
            # 탈락: 빨강
            color = (0, 0, 255)
            status = "OUT"
        elif label:
            # 생존 + OCR 성공: 초록
            color = (0, 255, 0)
            status = "ALIVE"
        else:
            # 생존 + OCR 미확인: 주황
            color = (0, 165, 255)
            status = "ALIVE"

        # 바운딩박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 이름 + 상태
        display_text = f"{name} [{status}]"
        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, display_text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 움직임 수치 표시 (레드 라이트 중)
        if game.state == GAME_RED and player and player["alive"]:
            movement = player.get("movement", 0)
            mv_text = f"Move: {movement:.1f}/{game.move_threshold:.0f}"
            cv2.putText(frame, mv_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 탈락 시 X 표시
        if player and not player["alive"]:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.line(frame, (x2, y1), (x1, y2), (0, 0, 255), 3)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="무궁화 꽃이 피었습니다 - YOLO26 Pose + OCR 게임")
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--ocr-interval", type=float, default=5.0)
    parser.add_argument("--lang", type=str, default="korean")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--move-threshold", type=float, default=15.0,
                        help="움직임 감지 임계값 (기본: 15.0, 낮을수록 민감)")
    parser.add_argument("--green-min", type=float, default=3.0,
                        help="그린 라이트 최소 시간 (기본: 3초)")
    parser.add_argument("--green-max", type=float, default=6.0,
                        help="그린 라이트 최대 시간 (기본: 6초)")
    parser.add_argument("--red-min", type=float, default=2.0,
                        help="레드 라이트 최소 시간 (기본: 2초)")
    parser.add_argument("--red-max", type=float, default=4.0,
                        help="레드 라이트 최대 시간 (기본: 4초)")
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
    game = GameState(
        move_threshold=args.move_threshold,
        green_duration=(args.green_min, args.green_max),
        red_duration=(args.red_min, args.red_max),
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"오류: 카메라 {args.camera}를 열 수 없습니다")
        sys.exit(1)

    print(f"모델: {args.model} | 카메라: {args.camera}")
    print("=" * 50)
    print("  s = 게임 시작")
    print("  r = 게임 리셋")
    print("  q = 종료")
    print("=" * 50)

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
        keypoints = result.keypoints

        # 키포인트 업데이트 → 움직임 감지
        if keypoints is not None and len(track_ids) > 0:
            kpts_data = keypoints.data.cpu().numpy()
            for i, tid in enumerate(track_ids):
                tid = int(tid)
                if i < len(kpts_data):
                    game.update_player_keypoints(tid, kpts_data[i])

        # 게임 상태 전환 (그린↔레드)
        game.update_state()

        # OCR: 인원 변경 시 또는 주기적
        now = time.time()
        current_person_count = len(boxes)
        person_count_changed = current_person_count != prev_person_count
        time_triggered = (now - last_ocr_time) >= args.ocr_interval
        if (time_triggered or person_count_changed) and current_person_count > 0:
            last_ocr_time = now
            prev_person_count = current_person_count
            jobs = []
            for box, tid in zip(boxes, track_ids):
                tid = int(tid)
                if person_tracker.get_label(tid):
                    continue
                crop = crop_person_region(frame, box)
                if crop is None:
                    continue
                jobs.append((tid, crop.copy()))
            if jobs:
                ocr_worker.submit(jobs)

        # 오래된 ID 정리
        active_ids = set(int(t) for t in track_ids) if len(track_ids) > 0 else set()
        person_tracker.cleanup(active_ids)

        # 시각화
        annotated = result.plot()
        annotated = draw_game_ui(annotated, game, boxes, track_ids, person_tracker)

        # FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
        if frame_count >= 30:
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Red Light Green Light", annotated)

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # 아무 키나 눌렸으면 디버그 출력
            print(f"  [KEY] 입력값: {key} ('{chr(key) if 32 <= key < 127 else '?'}')")
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("s") or key == ord("S"):
            game.start()
        elif key == ord("r") or key == ord("R"):
            game.reset()

    cap.release()
    cv2.destroyAllWindows()
    print("종료")


if __name__ == "__main__":
    main()
