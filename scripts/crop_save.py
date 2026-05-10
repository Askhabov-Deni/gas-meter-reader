import cv2
import os
from ultralytics import YOLO


model = YOLO("runs/detect/gas_meter_display_v1/weights/best.pt")

input_folder = "database/model_detect/dataset/images/train"
output_folder = 'database/model_ocr/digits_images'
photos = [f for f in os.listdir(input_folder) if f.endswith(('.jpeg', '.jpg')) ]
os.makedirs(output_folder, exist_ok=True)

for photo in photos:
    path = os.path.join(input_folder, photo)

    results = model(path, conf=0.5)
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        
        # Самая уверенная рамка
        best_idx = boxes.conf.argmax()
        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = boxes.conf[best_idx].item()
        
        x1, y1, x2, y2 = map(int, box)
        
        print(f"Дисплей найден Уверенность: {conf:.3f}")
        print(f"Координаты: ({x1}, {y1})  ({x2}, {y2})")
        
        img = cv2.imread(path)
        img = img[y1:y2, x1:x2]

        # Сохраняем
        out_path = os.path.join(output_folder, photo)
        cv2.imwrite(out_path, img)
    else:
        print("Дисплей не найден!")