from ultralytics import YOLO

class PoseTracker:
    def __init__(self, model_path="yolo26n-pose.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """YOLO 모델 로드"""
        print(f"YOLO 모델 로딩 중 ({self.model_path})...")
        self.model = YOLO(self.model_path)

    def track(self, frame, conf=0.5):
        """프레임에서 포즈 추적 수행"""
        if self.model is None:
            return None
        
        # persist=True를 통해 프레임 간 ID 유지
        results = self.model.track(
            source=frame, 
            persist=True, 
            conf=conf, 
            verbose=False
        )
        return results
