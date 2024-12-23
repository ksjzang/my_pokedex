import os
import requests
from ultralytics import YOLO
from PIL import Image

# 모델 URL 및 저장 경로 설정
MODEL_URL = "https://huggingface.co/keremberke/yolov8m-pokemon-classification/resolve/main/best.pt"
MODEL_PATH = "weights/best.pt"

# 모델 다운로드 함수
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully!")
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

# YOLOv8 모델 로드
download_model()
model = YOLO(MODEL_PATH)

def predict_pokemon(image_path):
    # 이미지를 열고 추론
    image = Image.open(image_path).convert("RGB")
    results = model(image)

    # 결과에서 가장 높은 점수의 라벨 가져오기
    if results and results[0].boxes:
        best_result = max(results[0].boxes, key=lambda x: x.conf)
        return best_result.cls  # 분류 라벨
    else:
        return "Unknown"
