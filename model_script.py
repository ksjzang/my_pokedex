from ultralytics import YOLO
from PIL import Image

# YOLOv8 모델 로드
MODEL_PATH = "keremberke/yolov8m-pokemon-classification"  # Hugging Face 모델 경로
model = YOLO(MODEL_PATH)

def predict_pokemon(image_path):
    # 이미지를 열고 추론
    image = Image.open(image_path).convert("RGB")
    results = model.predict(image)

    # 가장 높은 점수의 라벨 가져오기
    if results:
        best_result = max(results, key=lambda x: x.confidence)
        return best_result.label
    else:
        return "Unknown"
