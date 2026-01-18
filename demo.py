from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data/dataset.yaml", epochs=1)