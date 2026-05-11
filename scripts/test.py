import cv2
import os
from ultralytics import YOLO


IMG_FOLDER_PATH = 'database/raw_photos' # путь к папке  с фотографиями для теста


model_nano = YOLO("runs/detect/gas_meter_nano_v1/weights/best.pt")
model_small = YOLO("runs/detect/gas_meter_small_v1/weights/best.pt")
model_medium = YOLO("runs/detect/gas_meter_medium_v1/weights/best.pt")


models = {
    "nano":  {"model": model_nano, "color": (0, 0, 255)},    # красный
    "small": {"model": model_small, "color": (255, 0, 0)},    # синий
    "medium":{"model": model_medium, "color": (0, 255, 0)},    # зелёный
}


def detect_display(model, path, conf=0.5): # функция для обнаружения дисплея на фотографии, возвращает координаты рамки и уверенность
    results = model(path, conf=conf)
    result = results[0]
    
    if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            # Самая уверенная рамка
            best_idx = boxes.conf.argmax()
            box = boxes.xyxy[best_idx].cpu().numpy()
            conf = boxes.conf[best_idx].item()
            
            x1, y1, x2, y2 = map(int, box)

            return (x1, y1, x2, y2), conf
    else:
        print("Дисплей не найден!")
        return None, None


photos = [f for f in os.listdir(IMG_FOLDER_PATH) if f.endswith(('.jpeg', '.jpg')) ]  # Получаем список всех фотографий в папке для теста

len_photos = len(photos)
print(f"Найдено {len_photos} фотографий для теста.")

for i in range(1000, len_photos):   # модели обочно обучаются на 1000 фотографиях, поэтому тестируем с 1000ой
    photo = photos[i]
    path = os.path.join(IMG_FOLDER_PATH, photo)
    img = cv2.imread(path)

    print(f"\nТестируем фотографию {i+1}/{len_photos}: {photo}")

    for model in models.values():
        coords, conf = detect_display(model["model"], path)
        
        if coords is not None:
            x1, y1, x2, y2 = coords
            print(f"Дисплей найден! Уверенность: {conf:.3f}")
            print(f"Координаты: ({x1}, {y1})  ({x2}, {y2})")
            cv2.rectangle(img, (x1, y1), (x2, y2), model["color"], 2)
        else:
            print("Дисплей не найден!")
        
        
        cv2.rectangle(img, (x1, y1), (x2, y2), model["color"], 1)
        

    cv2.rectangle(img, (5, 5), (250, 95), (10,10,10), -1)

    # Надписи
    cv2.putText(img, "red - nano",  (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (30, 30, 200), 2)
    cv2.putText(img, "blue - small", (15, 55), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (200, 30, 30), 2)
    cv2.putText(img, "green - medium", (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (30, 200, 30), 2)
    
    # Подсказка
    cv2.putText(img, "ESC - exit | SPACE - next", (5, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    img = cv2.resize(img, None, fx=1, fy=1)
    cv2.imshow(photo, img)
    
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        print("\nВыход по ESC.")
        break


    cv2.destroyAllWindows()