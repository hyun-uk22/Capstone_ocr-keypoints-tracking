import os
import cv2
import numpy as np
import base64
import json
import time
import asyncio
import random
import multiprocessing as mp
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# 환경 변수 설정
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"

# --- 게임 상태 클래스 ---
GAME_WAITING = "waiting"
GAME_GREEN = "green_light"
GAME_RED = "red_light"
GAME_OVER = "game_over"
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
        if self.state not in (GAME_GREEN, GAME_RED): return
        
        elapsed = time.time() - self.state_start_time
        if elapsed >= self.current_duration:
            if self.state == GAME_GREEN:
                self.state = GAME_RED
                self.current_duration = random.uniform(*self.red_duration)
                self.state_start_time = time.time()
                for pid in self.players:
                    if self.players[pid]["alive"] and self.players[pid].get("current_kpts") is not None:
                        self.players[pid]["prev_kpts"] = self.players[pid]["current_kpts"].copy()
            elif self.state == GAME_RED:
                self.state = GAME_GREEN
                self.current_duration = random.uniform(*self.green_duration)
                self.state_start_time = time.time()
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
        if not player["alive"]: return
        
        player["current_kpts"] = keypoints
        if self.state == GAME_RED and player["prev_kpts"] is not None:
            movement = self._calc_movement(player["prev_kpts"], keypoints)
            player["movement"] = float(movement)
            if movement > self.move_threshold:
                player["alive"] = False

    def _calc_movement(self, prev_kpts, curr_kpts):
        total, count = 0.0, 0
        for idx in MOTION_KEYPOINTS:
            if idx >= len(prev_kpts) or idx >= len(curr_kpts): continue
            px, py = prev_kpts[idx][:2]
            cx, cy = curr_kpts[idx][:2]
            if (px == 0 and py == 0) or (cx == 0 and cy == 0): continue
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            total += dist
            count += 1
        return total / count if count > 0 else 0.0

    def get_frontend_state(self):
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

def ocr_worker_process(in_q, out_q):
    from paddleocr import PaddleOCR
    import os
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    
    print("\n[OCR 백그라운드 프로세스] 초기화 중...")
    ocr_engine = PaddleOCR(lang="korean", use_textline_orientation=True)
    print("[OCR 백그라운드 프로세스] 초기화 완료\n")
    
    while True:
        task = in_q.get()
        if task is None: break
        track_id, crop_img, req_time = task
        try:
            results = ocr_engine.predict(crop_img)
            best_text, best_score = None, 0.0
            if results:
                for res in results:
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])
                    for text, score in zip(texts, scores):
                        text = text.strip()
                        if text and score > best_score:
                            best_text = text
                            best_score = score
            if best_text and best_score > 0.15:
                out_q.put((track_id, best_text, best_score, req_time))
            else:
                out_q.put((track_id, None, 0.0, req_time))
        except:
            out_q.put((track_id, None, 0.0, req_time))

# --- 전역 변수 ---
model = None
ocr_in_q = None
ocr_out_q = None
ocr_process = None
ocr_cache = {}
game_state = GameState(move_threshold=15.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, ocr_in_q, ocr_out_q, ocr_process
    from ultralytics import YOLO
    print("YOLO 모델 로딩 중...")
    model = YOLO("yolo26n-pose.pt")
    
    ocr_in_q = mp.Queue()
    ocr_out_q = mp.Queue()
    ocr_process = mp.Process(target=ocr_worker_process, args=(ocr_in_q, ocr_out_q), daemon=True)
    ocr_process.start()
    
    yield
    
    if ocr_in_q: ocr_in_q.put(None)
    if ocr_process: ocr_process.join(timeout=3)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_ocr_queue():
    while ocr_out_q is not None and not ocr_out_q.empty():
        try:
            track_id, text, score, req_time = ocr_out_q.get_nowait()
            if text:
                ocr_cache[track_id] = {"text": text, "last_seen": time.time(), "processing": False}
            else:
                ocr_cache[track_id] = {"text": None, "last_seen": time.time() - 25, "processing": False}
        except:
            break

def get_ocr_text(frame, box, track_id):
    now = time.time()
    process_ocr_queue()
    cache_entry = ocr_cache.get(track_id)
    
    if cache_entry and (now - cache_entry["last_seen"] < 30):
        if not cache_entry["processing"] and cache_entry["text"]:
            cache_entry["last_seen"] = now
        return cache_entry["text"]
        
    if cache_entry and cache_entry.get("processing"):
        return cache_entry["text"]
        
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 - x1 < 10 or y2 - y1 < 10: return None
        
    crop = frame[y1:y2, x1:x2].copy()
    ocr_cache[track_id] = {"text": cache_entry["text"] if cache_entry else "인식 중...", "last_seen": now, "processing": True}
    
    if ocr_in_q:
        ocr_in_q.put((track_id, crop, now))
        
    return ocr_cache[track_id]["text"]

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # 클라이언트로부터 UI 명령어가 수신된 경우
            if data.startswith("{"):
                try:
                    cmd_data = json.loads(data)
                    cmd = cmd_data.get("command")
                    if cmd == "START":
                        game_state.start()
                    elif cmd == "RESET":
                        game_state.reset()
                except:
                    pass
                continue
            
            # 프레임 이미지 데이터 처리
            try:
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None: continue
            except: continue

            if model is None:
                await asyncio.sleep(0.1)
                continue

            results = model.track(source=frame, persist=True, conf=0.5, verbose=False)
            
            response_data = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy()
                    kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
                    
                    # 게임: 키포인트 업데이트 (움직임 계산)
                    if kpts is not None:
                        for i, tid in enumerate(track_ids):
                            game_state.update_player_keypoints(int(tid), kpts[i])
                    
                    # 게임: 상태 전환 검사
                    game_state.update_state()
                    
                    for i in range(len(track_ids)):
                        tid = int(track_ids[i])
                        box = boxes[i].tolist()
                        ocr_text = get_ocr_text(frame, box, tid)
                        
                        player = game_state.players.get(tid, {})
                        alive = player.get("alive", True)
                        movement = player.get("movement", 0.0)
                        
                        person_info = {
                            "id": tid,
                            "box": box,
                            "ocr": ocr_text,
                            "keypoints": kpts[i].tolist() if kpts is not None else [],
                            "alive": alive,
                            "movement": movement
                        }
                        response_data.append(person_info)

            await websocket.send_text(json.dumps({
                "status": "ok",
                "results": response_data,
                "game": game_state.get_frontend_state()
            }))

            now = time.time()
            expired_ids = [k for k, v in ocr_cache.items() if now - v["last_seen"] > 60 and not v["processing"]]
            for k in expired_ids: del ocr_cache[k]

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try: await websocket.close()
        except: pass

if __name__ == "__main__":
    mp.freeze_support() 
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
