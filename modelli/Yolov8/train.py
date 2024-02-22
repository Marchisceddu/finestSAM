from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="data.yaml", epochs=10, imgsz=640)