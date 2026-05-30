"""
OCR Pipeline для распознавания цифр со счётчиков газа.

Классы:
  - gas_meter  : барабанный счётчик, тёмный фон, 5 цифр
  - serial_id  : заводской номер, печатный шрифт
  - marker_id  : маркер от руки на корпусе

Зависимости:
  pip install transformers pillow opencv-python torch torchvision easyocr
"""

import re
import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# Препроцессинг
# ─────────────────────────────────────────────

def preprocess_gas_meter(img_bgr: np.ndarray) -> np.ndarray:
    """Инвертируем тёмный фон + CLAHE."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inv  = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(inv)
    _, thresh = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def preprocess_default(img_bgr: np.ndarray) -> np.ndarray:
    """Лёгкое усиление контраста."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def to_pil(img_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(img_rgb)


# ─────────────────────────────────────────────
# Ленивая загрузка моделей
# ─────────────────────────────────────────────

_easy_ocr           = None
_trocr_printed      = None
_trocr_handwritten  = None


def get_easyocr():
    global _easy_ocr
    if _easy_ocr is None:
        import easyocr
        _easy_ocr = easyocr.Reader(['en'], gpu=False)
    return _easy_ocr


def get_trocr(kind: str = 'printed'):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    global _trocr_printed, _trocr_handwritten

    if kind == 'printed' and _trocr_printed is None:
        name = 'microsoft/trocr-base-printed'
        _trocr_printed = (
            TrOCRProcessor.from_pretrained(name),
            VisionEncoderDecoderModel.from_pretrained(name),
        )
    if kind == 'handwritten' and _trocr_handwritten is None:
        name = 'microsoft/trocr-base-handwritten'
        _trocr_handwritten = (
            TrOCRProcessor.from_pretrained(name),
            VisionEncoderDecoderModel.from_pretrained(name),
        )
    return _trocr_printed if kind == 'printed' else _trocr_handwritten


# ─────────────────────────────────────────────
# Распознавание
# ─────────────────────────────────────────────

def _run_easyocr(img_rgb: np.ndarray) -> str:
    reader = get_easyocr()
    results = reader.readtext(
        img_rgb,
        detail=1,
        allowlist='0123456789',
        paragraph=False,
    )
    if not results:
        return ''
    # Берём только результаты с confidence > 0.3
    texts = [text for (_, text, conf) in results if conf > 0.3]
    return ''.join(texts)


def _run_trocr(pil_img: Image.Image, kind: str) -> str:
    import torch
    proc, model = get_trocr(kind)
    pixel_values = proc(pil_img.convert('RGB'), return_tensors='pt').pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values)
    return proc.batch_decode(ids, skip_special_tokens=True)[0]


def _digits_only(text: str) -> str:
    return re.sub(r'[^0-9]', '', text)


# ─────────────────────────────────────────────
# Постпроцессинг / валидация
# ─────────────────────────────────────────────

def postprocess(text: str, class_name: str) -> str:
    digits = _digits_only(text)

    if class_name == 'gas_meter':
        # Ровно 5 цифр
        if len(digits) > 5:
            digits = digits[:5]
        return digits

    if class_name == 'serial_id':
        return digits  # длина варьируется

    if class_name == 'marker_id':
        return digits

    return digits


# ─────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────

def read_meter_crop(image_path: str, class_name: str) -> dict:
    """
    Параметры
    ----------
    image_path : путь к кропу
    class_name : 'gas_meter' | 'serial_id' | 'marker_id'

    Возвращает
    ----------
    {
        'raw'   : сырой текст модели,
        'value' : только цифры после постпроцессинга,
        'model' : какая модель использовалась,
    }
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f'Не могу открыть: {image_path}')

    if class_name == 'gas_meter':
        img_rgb = preprocess_gas_meter(img_bgr)
        raw     = _run_easyocr(img_rgb)
        model   = 'EasyOCR + invert+CLAHE'

    elif class_name == 'serial_id':
        img_rgb = preprocess_default(img_bgr)
        raw     = _run_easyocr(img_rgb)
        model   = 'EasyOCR'
        # Fallback на TrOCR если EasyOCR вернул пустоту
        if not _digits_only(raw):
            raw   = _run_trocr(to_pil(img_rgb), 'printed')
            model = 'TrOCR-printed (fallback)'

    elif class_name == 'marker_id':
        img_rgb = preprocess_default(img_bgr)
        raw     = _run_trocr(to_pil(img_rgb), 'handwritten')
        model   = 'TrOCR-handwritten'
        # Fallback на EasyOCR если TrOCR вернул мусор
        if not _digits_only(raw):
            raw   = _run_easyocr(img_rgb)
            model = 'EasyOCR (fallback)'

    else:
        raise ValueError(f'Неизвестный класс: {class_name}')

    value = postprocess(raw, class_name)

    return {
        'raw'  : raw.strip(),
        'value': value,
        'model': model,
    }


# ─────────────────────────────────────────────
# CLI / быстрый тест
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    test_cases = [
        ('out_put/crops/gas_meter/1300000013__gas_meter_1.jpeg',    'gas_meter'),
        ('serial_number_1.jpg','serial_id'),
        ('out_put/crops/marker_id/1300000020__marker_id_1.jpeg',    'marker_id'),
    ]

    if len(sys.argv) == 3:
        test_cases = [(sys.argv[1], sys.argv[2])]

    for path, cls in test_cases:
        try:
            result = read_meter_crop(path, cls)
            print(f'[{cls}] {path}')
            print(f'  raw   : {result["raw"]}')
            print(f'  value : {result["value"]}')
            print(f'  model : {result["model"]}')
            print()
        except FileNotFoundError:
            print(f'[{cls}] {path} — файл не найден, пропускаем\n')