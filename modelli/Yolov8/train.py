from ultralytics import YOLO
import os

# Load a model
model = YOLO('pre_train/yolov8x-seg.pt')

# Ottiene il percorso del file yaml
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
yaml = os.path.join(current_directory, "../../dataset/yolo/data.yaml")

# Train the model
results = model.train(data=yaml, epochs=10, imgsz=640)