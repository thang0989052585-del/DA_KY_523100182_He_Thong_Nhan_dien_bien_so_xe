"""
train_yolo.py — Huấn luyện mô hình YOLO phát hiện biển số xe
"""
import os
import shutil
import torch
from ultralytics import YOLO
import config

def train():
    print("="*60)
    print("  HUẤN LUYỆN YOLO PHÁT HIỆN BIỂN SỐ XE")
    print("="*60)

    model = YOLO("yolov8n.pt") 
    dataset_yaml = os.path.join(config.DATASET_DIR, "yolo_data", "dataset.yaml")

    print("[1/2] Bắt đầu huấn luyện...")
    device = "0" if torch.cuda.is_available() else "cpu"
    
    results = model.train(
        data=dataset_yaml,
        epochs=30,           
        imgsz=640,          
        batch=16,           
        device=device,
        project="runs/detect",
        name="yolo_plate_detector",
        exist_ok=True       
    )

    # Move cleanly
    final_path = os.path.join(config.SAVED_MODELS_DIR, "yolo_plate_detector.pt")
    shutil.copy("runs/detect/yolo_plate_detector/weights/best.pt", final_path)
    shutil.rmtree("runs")

    print("\n[2/2] HOÀN THÀNH HUẤN LUYỆN YOLO!")
    print(f"Mô hình tốt nhất được lưu gọn gàng tại: {final_path}")

if __name__ == "__main__":
    train()
