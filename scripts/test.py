import cv2
import os
from ultralytics import YOLO


model = YOLO("runs/detect/gas_meter_display_v1/weights/best.pt")

folder = 'database/test_photos'
photos = [f for f in os.listdir(folder) if f.endswith(('.jpeg', '.jpg')) ]


for photo in photos:
    path = os.path.join(folder, photo)

    results = model(path, conf=0.5)
    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        
        # Самая уверенная рамка
        best_idx = boxes.conf.argmax()
        box = boxes.xyxy[best_idx].cpu().numpy()
        conf = boxes.conf[best_idx].item()
        
        x1, y1, x2, y2 = map(int, box)
        
        print(f"Дисплей найден! Уверенность: {conf:.3f}")
        print(f"Координаты: ({x1}, {y1})  ({x2}, {y2})")
        
        img = cv2.imread(path)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow(path, img)
        cv2.waitKey(0)  # Ждет нажатия ЛЮБОЙ клавиши
        cv2.destroyAllWindows()  # Закрывает окно после нажатия
    else:
        print("Дисплей не найден!")