import time
import random
import numpy as np

# --- 게임 상태 상수 ---
GAME_WAITING = "waiting"
GAME_GREEN = "green_light"
GAME_RED = "red_light"
GAME_OVER = "game_over"

# 움직임 감지용 주요 관절 인덱스 (COCO 기준)
MOTION_KEYPOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

class GameState:
    def __init__(self, move_threshold=15.0):
        self.state = GAME_WAITING
        self.move_threshold = move_threshold
        self.green_duration = (3.0, 6.0)
        self.red_duration = (2.0, 4.0)
        self.state_start_time = 0.0
        self.current_duration = 0.0
        self.players = {}

    def start(self):
        self.state = GAME_GREEN
        self.state_start_time = time.time()
        self.current_duration = random.uniform(*self.green_duration)
        for pid in self.players:
            self.players[pid]["alive"] = True
            self.players[pid]["prev_kpts"] = None

    def reset(self):
        self.state = GAME_WAITING
        self.players.clear()

    def update_state(self):
        if self.state not in (GAME_GREEN, GAME_RED):
            return
        
        elapsed = time.time() - self.state_start_time
        if elapsed >= self.current_duration:
            if self.state == GAME_GREEN:
                self.state = GAME_RED
                self.current_duration = random.uniform(*self.red_duration)
                self.state_start_time = time.time()
                # 레드 라이트 시작 시점의 포즈를 기준점으로 저장
                for pid in self.players:
                    if self.players[pid]["alive"] and self.players[pid].get("current_kpts") is not None:
                        self.players[pid]["prev_kpts"] = self.players[pid]["current_kpts"].copy()
            elif self.state == GAME_RED:
                self.state = GAME_GREEN
                self.current_duration = random.uniform(*self.green_duration)
                self.state_start_time = time.time()
                # 초록불이 되면 다시 기준점 초기화
                for pid in self.players:
                    self.players[pid]["prev_kpts"] = None
                
                alive_count = sum(1 for p in self.players.values() if p["alive"])
                if len(self.players) > 0 and alive_count == 0:
                    self.state = GAME_OVER

    def update_player_keypoints(self, track_id, keypoints):
        if track_id not in self.players:
            self.players[track_id] = {
                "alive": True,
                "prev_kpts": None,
                "current_kpts": None,
                "movement": 0.0,
            }
        player = self.players[track_id]
        if not player["alive"]:
            return
        
        player["current_kpts"] = keypoints
        
        # 레드 라이트 상태에서만 움직임 감지 수행
        if self.state == GAME_RED and player["prev_kpts"] is not None:
            movement = self._calc_movement(player["prev_kpts"], keypoints)
            player["movement"] = float(movement)
            if movement > self.move_threshold:
                player["alive"] = False

    def _calc_movement(self, prev_kpts, curr_kpts):
        """두 프레임 간의 주요 관절 평균 변위량을 계산합니다."""
        total, count = 0.0, 0
        for idx in MOTION_KEYPOINTS:
            if idx >= len(prev_kpts) or idx >= len(curr_kpts):
                continue
            px, py = prev_kpts[idx][:2]
            cx, cy = curr_kpts[idx][:2]
            
            # 유효하지 않은 좌표(0,0) 제외
            if (px == 0 and py == 0) or (cx == 0 and cy == 0):
                continue
                
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            total += dist
            count += 1
        return total / count if count > 0 else 0.0

    def get_frontend_state(self):
        """프론트엔드 전송용 상태 데이터를 구성합니다."""
        elapsed = time.time() - self.state_start_time
        remaining = max(0, self.current_duration - elapsed)
        
        if self.state == GAME_WAITING:
            banner_text = "[대기 중] 게임 시작 버튼을 누르세요"
        elif self.state == GAME_GREEN:
            banner_text = f"[무궁화 꽃이 피었습니다] 자유롭게 움직이세요! ({remaining:.1f}초)"
        elif self.state == GAME_RED:
            banner_text = f"[레드 라이트] 멈춰! 움직이면 탈락! ({remaining:.1f}초)"
        else:
            banner_text = "[게임 종료] 리셋 버튼을 누르세요"

        alive = sum(1 for p in self.players.values() if p["alive"])
        total = len(self.players)
        
        return {
            "state": self.state,
            "banner": banner_text,
            "alive_count": alive,
            "total_count": total
        }
