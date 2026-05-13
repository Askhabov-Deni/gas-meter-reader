import cv2
import os
from ultralytics import YOLO


model = YOLO("runs/detect/gas_meter_all_classes_s_v1/weights/best.pt")

INPUT_PATH = "database/raw_photos"
OUTPUT_PATH = 'out_put/crops'


# Создаём папку для каждого класса модели
class_names = model.names
for class_name in class_names.values():
    os.makedirs(os.path.join(OUTPUT_PATH, class_name), exist_ok=True)


photos = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
print(f"Найдено {len(photos)} фотографий.\n")

for photo in photos:
    path = os.path.join(INPUT_PATH, photo)

    results = model(path, conf=0.5)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print(f"[{photo}] ничего не найдено")
        continue

    img = cv2.imread(path)
    name_without_ext, ext = os.path.splitext(photo)

    # Счётчик для нумерации нескольких кропов одного класса на одном фото
    class_counter = {}

    for box, conf, cls in zip(
        result.boxes.xyxy,
        result.boxes.conf,
        result.boxes.cls,
    ):
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        conf_val = conf.item()
        cls_id = int(cls.item())
        cls_name = class_names.get(cls_id, str(cls_id))

        # Считаем сколько кропов этого класса уже сохранили для этого фото
        class_counter[cls_name] = class_counter.get(cls_name, 0) + 1
        crop_idx = class_counter[cls_name]

        crop = img[y1:y2, x1:x2]

        # имя файла: исходное_имя__класс_номер.jpg
        filename = f"{name_without_ext}__{cls_name}_{crop_idx}{ext}"
        out_path = os.path.join(OUTPUT_PATH, cls_name, filename)
        cv2.imwrite(out_path, crop)

        print(f"  [{photo}] {cls_name} #{crop_idx}  conf={conf_val:.3f}  coords=({x1},{y1},{x2},{y2})  -> {out_path}")