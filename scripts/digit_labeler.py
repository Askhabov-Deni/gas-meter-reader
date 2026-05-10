"""
Полуавтоматический инструмент разметки цифр на кропах дисплея газового счётчика.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import shutil
from pathlib import Path

# ─── НАСТРОЙКИ ───────────────────────────────────────────────────────────────
CROPS_DIR  = "database/model_ocr/digits_images"       # папка с кропами дисплеев
LABELS_DIR = "database/model_ocr/digits_labels"  # куда сохранять .txt
DONE_DIR   = None  # куда перемещать готовые (None — не перемещать)
CLASS_ID   = 0             # начальный класс для цифр (0=digit_0, 1=digit_1, ...)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────


def find_digit_boxes_by_projection(gray_crop):
    """
    Ищет позиции цифр через вертикальную проекцию пикселей.git --version
    Возвращает список (x1, x2) для каждой найденной группы.
    """
    _, binary = cv2.threshold(gray_crop, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(binary, axis=0).astype(float)

    # Сглаживание чтобы убрать мелкий шум
    kernel = np.ones(3) / 3
    proj = np.convolve(proj, kernel, mode="same")

    threshold = proj.max() * 0.05
    in_digit = False
    groups = []
    start = 0
    for x, val in enumerate(proj):
        if val > threshold and not in_digit:
            start = x
            in_digit = True
        elif val <= threshold and in_digit:
            groups.append((start, x))
            in_digit = False
    if in_digit:
        groups.append((start, len(proj) - 1))

    # Убираем слишком маленькие группы (шум)
    min_width = gray_crop.shape[1] * 0.04
    groups = [(x1, x2) for x1, x2 in groups if (x2 - x1) >= min_width]
    return groups


def uniform_split(width, n=5):
    """Равномерное деление ширины на n частей."""
    step = width / n
    return [(int(i * step), int((i + 1) * step)) for i in range(n)]


def boxes_to_yolo(boxes, img_w, img_h, digit_str):
    """
    Конвертирует список (x1, x2) + строку цифр в YOLO строки.
    Возвращает список строк для .txt файла.
    """
    lines = []
    for (x1, x2), digit_char in zip(boxes, digit_str):
        cx = (x1 + x2) / 2 / img_w
        cy = 0.5  # по высоте — центр
        bw = (x2 - x1) / img_w
        bh = 0.85  # занимаем 85% высоты кропа
        class_id = int(digit_char)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def draw_preview(pil_img, boxes, digit_str, img_w, img_h):
    """Рисует bbox-ы с подписями на PIL изображении."""
    draw = ImageDraw.Draw(pil_img)
    colors = ["#FF4444", "#FF8800", "#FFCC00", "#44BB44", "#4488FF"]
    for i, ((x1, x2), ch) in enumerate(zip(boxes, digit_str)):
        color = colors[i % len(colors)]
        draw.rectangle([x1, 2, x2, img_h - 2], outline=color, width=2)
        # Подпись
        draw.rectangle([x1, 2, x1 + 18, 20], fill=color)
        draw.text((x1 + 3, 3), ch, fill="white")
    return pil_img


class DigitLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gas Meter — Digit Labeler")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("900x650")
        self.root.resizable(True, True)

        # Папки
        for d in [CROPS_DIR, LABELS_DIR]:
            Path(d).mkdir(exist_ok=True)
        if DONE_DIR:
            Path(DONE_DIR).mkdir(exist_ok=True)

        self.image_paths = []
        self.current_idx = 0
        self.current_boxes = []
        self.current_img = None
        self.photo = None
        self.auto_mode = tk.BooleanVar(value=True)

        self._build_ui()
        self._load_image_list()
        if self.image_paths:
            self._show_image(0)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Верхняя панель
        top = tk.Frame(self.root, bg="#16213e", pady=8)
        top.pack(fill="x")

        tk.Label(top, text="⛽ Gas Meter Digit Labeler",
                 bg="#16213e", fg="#e0e0ff",
                 font=("Courier New", 14, "bold")).pack(side="left", padx=16)

        self.progress_var = tk.StringVar(value="0 / 0")
        tk.Label(top, textvariable=self.progress_var,
                 bg="#16213e", fg="#888aaa",
                 font=("Courier New", 11)).pack(side="left", padx=12)

        btn_open = tk.Button(top, text="📂 Папка с кропами",
                             command=self._choose_folder,
                             bg="#0f3460", fg="#e0e0ff",
                             relief="flat", padx=10, pady=4,
                             cursor="hand2")
        btn_open.pack(side="right", padx=12)

        # Центральная область
        center = tk.Frame(self.root, bg="#1a1a2e")
        center.pack(fill="both", expand=True, padx=16, pady=8)

        # Левая — изображение
        img_frame = tk.Frame(center, bg="#0d0d1a", bd=0)
        img_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(img_frame, bg="#0d0d1a",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)

        self.filename_label = tk.Label(img_frame, text="",
                                       bg="#0d0d1a", fg="#555577",
                                       font=("Courier New", 9))
        self.filename_label.pack(pady=2)

        # Правая — контролы
        ctrl = tk.Frame(center, bg="#1a1a2e", width=220)
        ctrl.pack(side="right", fill="y", padx=(12, 0))
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="ПОКАЗАНИЯ", bg="#1a1a2e", fg="#6666aa",
                 font=("Courier New", 9, "bold")).pack(anchor="w", pady=(8, 2))

        # Поле ввода
        entry_frame = tk.Frame(ctrl, bg="#0d0d1a",
                               highlightbackground="#334",
                               highlightthickness=1)
        entry_frame.pack(fill="x", pady=4)

        self.digit_var = tk.StringVar()
        self.digit_entry = tk.Entry(entry_frame,
                                    textvariable=self.digit_var,
                                    font=("Courier New", 28, "bold"),
                                    bg="#0d0d1a", fg="#00ff88",
                                    insertbackground="#00ff88",
                                    relief="flat", width=7,
                                    justify="center")
        self.digit_entry.pack(padx=8, pady=8)
        self.digit_entry.bind("<KeyRelease>", self._on_digit_change)
        self.digit_entry.bind("<Return>", lambda e: self._save_and_next())
        self.digit_entry.bind("<KP_Enter>", lambda e: self._save_and_next())
        self.digit_entry.focus()

        tk.Label(ctrl, text="5 цифр без дробной части",
                 bg="#1a1a2e", fg="#444466",
                 font=("Courier New", 8)).pack()

        # Режим автоматического bbox
        mode_frame = tk.Frame(ctrl, bg="#1a1a2e")
        mode_frame.pack(fill="x", pady=12)
        tk.Label(mode_frame, text="Режим bbox:", bg="#1a1a2e", fg="#6666aa",
                 font=("Courier New", 9)).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Авто (проекция)",
                       variable=self.auto_mode, value=True,
                       bg="#1a1a2e", fg="#aaaacc",
                       selectcolor="#0d0d1a",
                       font=("Courier New", 9),
                       command=self._recompute_boxes).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Равномерный",
                       variable=self.auto_mode, value=False,
                       bg="#1a1a2e", fg="#aaaacc",
                       selectcolor="#0d0d1a",
                       font=("Courier New", 9),
                       command=self._recompute_boxes).pack(anchor="w")

        # Статус bbox
        self.bbox_status = tk.Label(ctrl, text="",
                                    bg="#1a1a2e",
                                    font=("Courier New", 8),
                                    wraplength=200, justify="left")
        self.bbox_status.pack(anchor="w", pady=4)

        # Кнопки
        btn_cfg = dict(bg="#0f3460", fg="#e0e0ff", relief="flat",
                       pady=8, cursor="hand2",
                       font=("Courier New", 10, "bold"))

        self.btn_save = tk.Button(ctrl, text="✅  Сохранить  [Enter]",
                                  command=self._save_and_next,
                                  **btn_cfg)
        self.btn_save.pack(fill="x", pady=4)
        self.btn_save.configure(bg="#1a6640")

        tk.Button(ctrl, text="⏭  Пропустить  [→]",
                  command=self._skip,
                  **btn_cfg).pack(fill="x", pady=2)

        tk.Button(ctrl, text="⏮  Назад  [←]",
                  command=self._prev,
                  **btn_cfg).pack(fill="x", pady=2)

        tk.Button(ctrl, text="🗑  Удалить кроп",
                  command=self._delete_current,
                  bg="#4a1a1a", fg="#ffaaaa",
                  relief="flat", pady=6,
                  cursor="hand2",
                  font=("Courier New", 9)).pack(fill="x", pady=(12, 2))

        # Нижняя строка подсказок
        hints = tk.Label(self.root,
                         text="Enter — сохранить    ← → — навигация    Только цифры 0-9",
                         bg="#1a1a2e", fg="#333355",
                         font=("Courier New", 8))
        hints.pack(side="bottom", pady=4)

        # Клавиши
        self.root.bind("<Right>", lambda e: self._skip())
        self.root.bind("<Left>", lambda e: self._prev())

    #  Загрузка файлов 

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с кропами")
        if folder:
            global CROPS_DIR
            CROPS_DIR = folder
            self._load_image_list()
            if self.image_paths:
                self._show_image(0)

    def _load_image_list(self):
        p = Path(CROPS_DIR)
        if not p.exists():
            self.image_paths = []
            return
        # Исключаем уже размеченные
        already_done = {
            Path(f).stem
            for f in Path(LABELS_DIR).glob("*.txt")
        }
        self.image_paths = sorted([
            str(f) for f in p.iterdir()
            if f.suffix.lower() in EXTENSIONS
            and f.stem not in already_done
        ])
        self.current_idx = 0
        self._update_progress()

    def _update_progress(self):
        total_done = len(list(Path(LABELS_DIR).glob("*.txt")))
        remaining = len(self.image_paths)
        self.progress_var.set(
            f"Осталось: {remaining}  |  Готово: {total_done}"
        )

    #  Отображение

    def _show_image(self, idx):
        if not self.image_paths:
            self._show_empty()
            return

        self.current_idx = max(0, min(idx, len(self.image_paths) - 1))
        path = self.image_paths[self.current_idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            self._skip()
            return

        self.current_img = img_bgr
        self.digit_var.set("")
        self._recompute_boxes()
        self.filename_label.config(text=Path(path).name)
        self.digit_entry.focus()

    def _recompute_boxes(self):
        if self.current_img is None:
            return
        img = self.current_img
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.auto_mode.get():
            boxes = find_digit_boxes_by_projection(gray)
            if len(boxes) == 5:
                self.current_boxes = boxes
                self.bbox_status.config(
                    text=f"✅ Авто: найдено 5 групп",
                    fg="#44ff88")
            else:
                self.current_boxes = uniform_split(w)
                self.bbox_status.config(
                    text=f"⚠️ Авто: {len(boxes)} групп → равномерный",
                    fg="#ffaa44")
        else:
            self.current_boxes = uniform_split(w)
            self.bbox_status.config(
                text="Равномерное деление",
                fg="#8888aa")

        self._render_canvas()

    def _render_canvas(self):
        if self.current_img is None:
            return

        digit_str = self.digit_var.get().strip()
        img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        h, w = self.current_img.shape[:2]

        # Масштабирование под canvas
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 200)
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((nw, nh), Image.LANCZOS)

        # Масштабированные boxes
        scaled_boxes = [(int(x1 * scale), int(x2 * scale))
                        for x1, x2 in self.current_boxes]

        if len(digit_str) == 5 and digit_str.isdigit():
            pil_img = draw_preview(pil_img, scaled_boxes, digit_str, nw, nh)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2
        self.canvas.create_image(x_off, y_off, anchor="nw", image=self.photo)

    def _show_empty(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            200, 150,
            text="Нет изображений\nВыберите папку с кропами",
            fill="#555577",
            font=("Courier New", 12),
            justify="center"
        )
        self.progress_var.set("Готово! Все размечено 🎉")

    # ── Действия ─────────────────────────────────────────────────────────────

    def _on_digit_change(self, event=None):
        val = self.digit_var.get()
        # Оставляем только цифры, максимум 5
        clean = "".join(c for c in val if c.isdigit())[:5]
        if clean != val:
            self.digit_var.set(clean)
            self.digit_entry.icursor(len(clean))
        self._render_canvas()

    def _save_and_next(self):
        digit_str = self.digit_var.get().strip()
        if len(digit_str) != 5 or not digit_str.isdigit():
            self.digit_entry.config(bg="#3a0d0d")
            self.root.after(300,
                lambda: self.digit_entry.config(bg="#0d0d1a"))
            return

        path = self.image_paths[self.current_idx]
        stem = Path(path).stem
        h, w = self.current_img.shape[:2]

        lines = boxes_to_yolo(self.current_boxes, w, h, digit_str)

        label_path = Path(LABELS_DIR) / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        if DONE_DIR:
            dst = Path(DONE_DIR) / Path(path).name
            shutil.move(path, dst)

        # Убираем из списка и показываем следующий
        self.image_paths.pop(self.current_idx)
        self._update_progress()

        if not self.image_paths:
            self._show_empty()
        else:
            next_idx = min(self.current_idx, len(self.image_paths) - 1)
            self._show_image(next_idx)

    def _skip(self):
        if not self.image_paths:
            return
        next_idx = (self.current_idx + 1) % len(self.image_paths)
        self._show_image(next_idx)

    def _prev(self):
        if not self.image_paths:
            return
        prev_idx = (self.current_idx - 1) % len(self.image_paths)
        self._show_image(prev_idx)

    def _delete_current(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        if messagebox.askyesno("Удалить?",
                               f"Удалить файл?\n{Path(path).name}"):
            os.remove(path)
            self.image_paths.pop(self.current_idx)
            self._update_progress()
            if not self.image_paths:
                self._show_empty()
            else:
                self._show_image(min(self.current_idx,
                                     len(self.image_paths) - 1))

    def _on_canvas_resize(self, event):
        self._render_canvas()


def main():
    root = tk.Tk()
    app = DigitLabelerApp(root)
    root.bind("<Configure>", lambda e: app._render_canvas()
              if e.widget == root else None)
    root.mainloop()


if __name__ == "__main__":
    main()