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

# 환경 변수 설정 (에러 및 지연 방지)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"

def ocr_worker_process(in_q, out_q):
    """
    이 함수는 완전히 별도의 프로세스(다른 CPU 코어)에서 실행됩니다.
    YOLO 메인 루프와 자원을 공유하지 않으므로 키포인트 추적을 절대 방해하지 않습니다.
    """
    from paddleocr import PaddleOCR
    import os
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["FLAGS_use_mkldnn"] = "0"
    
    print("\n[OCR 백그라운드 프로세스] 초기화 중...")
    ocr_engine = PaddleOCR(lang="korean", use_textline_orientation=True)
    print("[OCR 백그라운드 프로세스] 초기화 완료 및 대기 중...\n")
    
    while True:
        task = in_q.get()
        if task is None: # 종료 신호
            break
            
        track_id, crop_img, req_time = task
        try:
            results = ocr_engine.predict(crop_img)
            best_text = None
            best_score = 0.0
            
            if results:
                for res in results:
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])
                    for text, score in zip(texts, scores):
                        text = text.strip()
                        if text and score > best_score:
                            best_text = text
                            best_score = score
                            
            if best_text and best_score > 0.15: # 신뢰도 0.15 초과 시 성공
                out_q.put((track_id, best_text, best_score, req_time))
            else:
                out_q.put((track_id, None, 0.0, req_time))
        except Exception as e:
            print(f"[OCR 에러]: {e}")
            out_q.put((track_id, None, 0.0, req_time))

# --- 전역 변수 ---
model = None
ocr_in_q = None
ocr_out_q = None
ocr_process = None
ocr_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, ocr_in_q, ocr_out_q, ocr_process
    
    from ultralytics import YOLO
    print("YOLO 모델 로딩 중...")
    model = YOLO("yolo26n-pose.pt")
    
    print("OCR 워커 프로세스 생성 중...")
    ocr_in_q = mp.Queue()
    ocr_out_q = mp.Queue()
    # 데몬 프로세스로 생성하여 메인 서버 종료 시 함께 꺼지도록 설정
    ocr_process = mp.Process(target=ocr_worker_process, args=(ocr_in_q, ocr_out_q), daemon=True)
    ocr_process.start()
    
    yield  # 서버 실행 중 (메인 루프)
    
    print("서버 종료 중. 워커 정리...")
    if ocr_in_q:
        ocr_in_q.put(None)
    if ocr_process:
        ocr_process.join(timeout=3)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_ocr_queue():
    """백그라운드 프로세스에서 완료된 OCR 결과를 가져와 캐시에 반영 (블로킹 없음)"""
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
    """메인 프로세스에서 OCR 결과 반환 (즉시 리턴)"""
    now = time.time()
    
    # 큐 업데이트 (매 프레임마다 즉시 완료된 게 있는지 확인)
    process_ocr_queue()
    
    cache_entry = ocr_cache.get(track_id)
    
    # 1. 유효한 결과가 있는 경우
    if cache_entry and (now - cache_entry["last_seen"] < 30):
        if not cache_entry["processing"] and cache_entry["text"]:
            cache_entry["last_seen"] = now
        return cache_entry["text"]
        
    # 2. 백그라운드 코어에서 이미 분석 중인 경우
    if cache_entry and cache_entry.get("processing"):
        return cache_entry["text"]
        
    # 3. 새로운 사람 등장 -> 프로세스 큐에 작업 던지기
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
        
    crop = frame[y1:y2, x1:x2].copy()
    
    # 캐시를 '인식 중' 상태로 등록
    ocr_cache[track_id] = {
        "text": cache_entry["text"] if cache_entry else "인식 중...", 
        "last_seen": now, 
        "processing": True
    }
    
    if ocr_in_q:
        ocr_in_q.put((track_id, crop, now)) # 별도 프로세스로 전송
        
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
            
            try:
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None: continue
            except: continue

            if model is None:
                # 서버가 뜨고 YOLO 모델이 아직 로딩 전이면 무시
                await asyncio.sleep(0.1)
                continue

            # YOLO 추론은 메인 프로세스에서 방해 없이 단독으로 실행됨 (극강의 FPS 보장)
            results = model.track(source=frame, persist=True, conf=0.5, verbose=False)
            
            response_data = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy()
                    kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
                    
                    for i in range(len(track_ids)):
                        tid = int(track_ids[i])
                        box = boxes[i].tolist()
                        
                        # OCR 요청 던지기 (절대 블로킹 안됨)
                        ocr_text = get_ocr_text(frame, box, tid)
                        
                        person_info = {
                            "id": tid,
                            "box": box,
                            "ocr": ocr_text,
                            "keypoints": kpts[i].tolist() if kpts is not None else []
                        }
                        response_data.append(person_info)

            await websocket.send_text(json.dumps({
                "status": "ok",
                "results": response_data
            }))

            # 더 이상 추적 안되는 가비지 데이터 메모리 정리
            now = time.time()
            expired_ids = [k for k, v in ocr_cache.items() if now - v["last_seen"] > 60 and not v["processing"]]
            for k in expired_ids:
                del ocr_cache[k]

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try: await websocket.close()
        except: pass

if __name__ == "__main__":
    # 윈도우 환경에서 멀티프로세싱 사용 시 필수 코드
    mp.freeze_support() 
    import uvicorn
    # 문자열 "server:app" 형식으로 호출해야 멀티프로세싱 충돌이 없습니다.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
