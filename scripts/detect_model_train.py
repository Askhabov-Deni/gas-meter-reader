from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(
    data="database/model_detect_id/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="gas_meter_nano_v1",
    patience=20,          # early stopping
    augment=True,
    degrees=5,            # поворот 
    scale=0.3,
    hsv_v=0.4,            # вариации освещения
)