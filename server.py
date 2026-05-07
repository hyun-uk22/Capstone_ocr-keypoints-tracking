import os
import cv2
import numpy as np
import base64
import json
import time
import asyncio
import multiprocessing as mp
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# 도메인 모듈 임포트
from src.game.state import GameState
from src.ocr.processor import OCRManager
from src.pose.tracker import PoseTracker

# --- 전역 객체 초기화 ---
game_state = GameState(move_threshold=15.0)
ocr_manager = OCRManager()
pose_tracker = PoseTracker("yolo26n-pose.pt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 자원 초기화
    pose_tracker.load_model()
    ocr_manager.start()
    
    yield
    
    # 서버 종료 시 자원 해제
    ocr_manager.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    is_processing = False
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # 1. UI 명령어 처리 (START/RESET)
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
            
            # 2. 프레임 이미지 데이터 처리
            try:
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except:
                continue

            # 3. 모델 추론 (YOLO-Pose + Tracking) - 이벤트 루프 블로킹 방지를 위해 비동기 스레드 위임
            results = await asyncio.to_thread(pose_tracker.track, frame, 0.5)
            
            response_data = []
            if results and len(results) > 0:
                result = results[0]
                
                # 트래킹 ID가 부여된 경우에만 처리
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy()
                    kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
                    
                    # 게임 상태 및 플레이어 데이터 업데이트
                    if kpts is not None:
                        for i, tid in enumerate(track_ids):
                            game_state.update_player_keypoints(int(tid), kpts[i])
                    
                    # 게임 전역 상태(초록불/빨간불) 전환 검사
                    game_state.update_state()
                    
                    # 각 플레이어별 정보 구성
                    for i in range(len(track_ids)):
                        tid = int(track_ids[i])
                        box = boxes[i].tolist()
                        
                        # OCR 인식 결과 가져오기 (비동기 큐/캐시 활용)
                        player_keypoints = kpts[i] if kpts is not None else None
                        ocr_text = ocr_manager.get_text(frame, box, tid, player_keypoints)
                        
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

            # 4. 결과 전송
            await websocket.send_text(json.dumps({
                "status": "ok",
                "results": response_data,
                "game": game_state.get_frontend_state()
            }))

            # 5. 오래된 OCR 캐시 정리
            ocr_manager.clean_cache()

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    # Windows 환경 멀티프로세싱 지원
    mp.freeze_support() 
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
