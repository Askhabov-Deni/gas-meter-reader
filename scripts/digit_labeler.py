"""
Полуавтоматический инструмент разметки цифр на кропах газового счётчика.
Поддерживает три класса: gas_meter, serial_id, marker_id

Режимы bbox:
  - Авто (проекция)  — для gas_meter
  - Равномерный      — запасной вариант
  - Клики            — для serial_id / marker_id: кликаешь границы между цифрами
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import os
from pathlib import Path

# ─── НАСТРОЙКИ ───────────────────────────────────────────────────────────────
BASE_DIR = "database/model_ocr"

CLASSES = {
    "gas_meter": {
        "label":      "Gas Meter",
        "color":      "#00ff88",
        "crops_dir":  f"{BASE_DIR}/gas_meter/images",
        "labels_dir": f"{BASE_DIR}/gas_meter/labels",
        "fixed_len":  5,
        "hint":       "5 цифр (показания дисплея)",
        "default_mode": "auto",
    },
    "serial_id": {
        "label":      "Serial ID",
        "color":      "#00cfff",
        "crops_dir":  f"{BASE_DIR}/serial_id/images",
        "labels_dir": f"{BASE_DIR}/serial_id/labels",
        "fixed_len":  None,
        "hint":       "Заводской номер — кликай границы между цифрами",
        "default_mode": "click",
    },
    "marker_id": {
        "label":      "Marker ID",
        "color":      "#ffaa00",
        "crops_dir":  f"{BASE_DIR}/marker_id/images",
        "labels_dir": f"{BASE_DIR}/marker_id/labels",
        "fixed_len":  None,
        "hint":       "Написано маркером — кликай границы между цифрами",
        "default_mode": "click",
    },
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────


def find_digit_boxes_by_projection(gray_crop):
    _, binary = cv2.threshold(gray_crop, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(binary, axis=0).astype(float)
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
    min_width = gray_crop.shape[1] * 0.04
    groups = [(x1, x2) for x1, x2 in groups if (x2 - x1) >= min_width]
    return groups


def uniform_split(width, n):
    step = width / n
    return [(int(i * step), int((i + 1) * step)) for i in range(n)]


def splits_to_boxes(splits, img_w):
    """Конвертирует список X-разделителей в список (x1, x2) боксов."""
    boundaries = [0] + sorted(splits) + [img_w]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def boxes_to_yolo(boxes, img_w, img_h, digit_str):
    lines = []
    for (x1, x2), digit_char in zip(boxes, digit_str):
        cx = (x1 + x2) / 2 / img_w
        cy = 0.5
        bw = (x2 - x1) / img_w
        bh = 0.85
        class_id = int(digit_char)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def draw_preview(pil_img, boxes, digit_str, accent_color):
    draw = ImageDraw.Draw(pil_img)
    colors = [accent_color, "#ff6688", "#ffcc00", "#44eeff", "#aa88ff",
              "#ff8844", "#88ff44", "#ff44cc", "#44ffcc", "#ffff44"]
    nw, nh = pil_img.size
    for i, ((x1, x2), ch) in enumerate(zip(boxes, digit_str)):
        color = colors[i % len(colors)]
        draw.rectangle([x1, 2, x2, nh - 2], outline=color, width=2)
        draw.rectangle([x1, 2, x1 + 16, 18], fill=color)
        draw.text((x1 + 2, 2), ch, fill="#000000")
    return pil_img


def draw_splits(pil_img, splits_scaled, accent_color):
    """Рисует вертикальные линии-разделители в режиме кликов."""
    draw = ImageDraw.Draw(pil_img)
    nw, nh = pil_img.size
    for x in splits_scaled:
        draw.line([(x, 0), (x, nh)], fill=accent_color, width=2)
        # Маленький кружок сверху для наглядности
        draw.ellipse([x - 5, 0, x + 5, 10], fill=accent_color)
    return pil_img


class DigitLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meter Digit Labeler")
        self.root.configure(bg="#0e0e14")
        self.root.geometry("1000x680")
        self.root.resizable(True, True)

        self.current_class_key = tk.StringVar(value="gas_meter")
        self.image_paths = []
        self.current_idx = 0
        self.current_boxes = []
        self.current_img = None
        self.photo = None

        # Режим: "auto" | "uniform" | "click"
        self.bbox_mode = tk.StringVar(value="auto")

        # Клики: список X-координат разделителей в координатах ИЗОБРАЖЕНИЯ
        self.click_splits = []

        # Параметры масштабирования (для перевода кликов canvas → image)
        self._img_scale = 1.0
        self._img_offset_x = 0
        self._img_offset_y = 0

        for cfg in CLASSES.values():
            Path(cfg["crops_dir"]).mkdir(parents=True, exist_ok=True)
            Path(cfg["labels_dir"]).mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._on_class_change()

    # ── Свойства ─────────────────────────────────────────────────────────────

    @property
    def cls(self):
        return CLASSES[self.current_class_key.get()]

    @property
    def accent(self):
        return self.cls["color"]

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Верхняя панель
        top = tk.Frame(self.root, bg="#080810", pady=6)
        top.pack(fill="x")

        tk.Label(top, text="METER LABELER",
                 bg="#080810", fg="#ffffff",
                 font=("Courier New", 13, "bold")).pack(side="left", padx=16)

        self.progress_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.progress_var,
                 bg="#080810", fg="#555577",
                 font=("Courier New", 10)).pack(side="left", padx=12)

        tk.Button(top, text="📂 Сменить папку",
                  command=self._choose_folder,
                  bg="#1a1a2e", fg="#aaaacc",
                  relief="flat", padx=10, pady=3,
                  cursor="hand2").pack(side="right", padx=12)

        # Вкладки классов
        tabs_frame = tk.Frame(self.root, bg="#0e0e14")
        tabs_frame.pack(fill="x")

        self.tab_buttons = {}
        for key, cfg in CLASSES.items():
            btn = tk.Button(
                tabs_frame, text=cfg["label"],
                command=lambda k=key: self._switch_class(k),
                bg="#14141e", fg="#555577",
                relief="flat", padx=18, pady=8,
                cursor="hand2",
                font=("Courier New", 10, "bold"),
                borderwidth=0,
            )
            btn.pack(side="left")
            self.tab_buttons[key] = btn

        tk.Frame(self.root, bg="#1a1a2e", height=1).pack(fill="x")

        # Центральная область
        center = tk.Frame(self.root, bg="#0e0e14")
        center.pack(fill="both", expand=True, padx=12, pady=8)

        # Левая — изображение
        img_frame = tk.Frame(center, bg="#080810")
        img_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(img_frame, bg="#080810", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        self.filename_label = tk.Label(img_frame, text="",
                                       bg="#080810", fg="#333355",
                                       font=("Courier New", 8))
        self.filename_label.pack(pady=2)

        # Правая — контролы
        ctrl = tk.Frame(center, bg="#0e0e14", width=230)
        ctrl.pack(side="right", fill="y", padx=(12, 0))
        ctrl.pack_propagate(False)

        self.hint_label = tk.Label(ctrl, text="",
                                   bg="#0e0e14", fg="#444466",
                                   font=("Courier New", 8),
                                   wraplength=210)
        self.hint_label.pack(anchor="w", pady=(10, 4))

        # Поле ввода
        entry_frame = tk.Frame(ctrl, bg="#080810",
                               highlightbackground="#222233",
                               highlightthickness=1)
        entry_frame.pack(fill="x", pady=4)

        self.digit_var = tk.StringVar()
        self.digit_entry = tk.Entry(
            entry_frame,
            textvariable=self.digit_var,
            font=("Courier New", 32, "bold"),
            bg="#080810", fg="#00ff88",
            insertbackground="#00ff88",
            relief="flat", width=8,
            justify="center",
        )
        self.digit_entry.pack(padx=8, pady=10)
        self.digit_entry.bind("<KeyRelease>", self._on_digit_change)
        self.digit_entry.bind("<Return>",   lambda e: self._save_and_next())
        self.digit_entry.bind("<KP_Enter>", lambda e: self._save_and_next())

        self.len_label = tk.Label(ctrl, text="",
                                  bg="#0e0e14", fg="#444466",
                                  font=("Courier New", 9))
        self.len_label.pack()

        # Режим bbox
        mode_frame = tk.Frame(ctrl, bg="#0e0e14")
        mode_frame.pack(fill="x", pady=10)
        tk.Label(mode_frame, text="BBOX РЕЖИМ",
                 bg="#0e0e14", fg="#333355",
                 font=("Courier New", 8, "bold")).pack(anchor="w")

        for val, txt in [("auto", "Авто (проекция)"),
                         ("uniform", "Равномерный"),
                         ("click", "Клики  [ЛКМ = граница, ПКМ = удалить]")]:
            tk.Radiobutton(mode_frame, text=txt,
                           variable=self.bbox_mode, value=val,
                           bg="#0e0e14", fg="#aaaacc",
                           selectcolor="#080810",
                           font=("Courier New", 9),
                           command=self._on_mode_change).pack(anchor="w")

        self.bbox_status = tk.Label(ctrl, text="",
                                    bg="#0e0e14", fg="#555577",
                                    font=("Courier New", 8),
                                    wraplength=210, justify="left")
        self.bbox_status.pack(anchor="w", pady=2)

        # Кнопка сброса кликов
        self.btn_clear_clicks = tk.Button(ctrl, text="🗑  Сбросить клики  [C]",
                                          command=self._clear_clicks,
                                          bg="#1a1408", fg="#ffaa44",
                                          relief="flat", pady=5,
                                          cursor="hand2",
                                          font=("Courier New", 9))
        self.btn_clear_clicks.pack(fill="x", pady=2)

        # Кнопки действий
        self.btn_save = tk.Button(ctrl, text="✅  Сохранить  [Enter]",
                                  command=self._save_and_next,
                                  bg="#0d3320", fg="#00ff88",
                                  relief="flat", pady=9,
                                  cursor="hand2",
                                  font=("Courier New", 10, "bold"))
        self.btn_save.pack(fill="x", pady=(10, 3))

        tk.Button(ctrl, text="⏭  Пропустить  [→]",
                  command=self._skip,
                  bg="#14141e", fg="#aaaacc",
                  relief="flat", pady=7,
                  cursor="hand2",
                  font=("Courier New", 9)).pack(fill="x", pady=2)

        tk.Button(ctrl, text="⏮  Назад  [←]",
                  command=self._prev,
                  bg="#14141e", fg="#aaaacc",
                  relief="flat", pady=7,
                  cursor="hand2",
                  font=("Courier New", 9)).pack(fill="x", pady=2)

        tk.Button(ctrl, text="🗑  Удалить кроп",
                  command=self._delete_current,
                  bg="#1a0808", fg="#ff6666",
                  relief="flat", pady=6,
                  cursor="hand2",
                  font=("Courier New", 9)).pack(fill="x", pady=(10, 2))

        tk.Label(self.root,
                 text="Enter — сохранить    ← → — навигация    Tab — класс    C — сбросить клики",
                 bg="#0e0e14", fg="#222233",
                 font=("Courier New", 8)).pack(side="bottom", pady=3)

        self.root.bind("<Right>", lambda e: self._skip())
        self.root.bind("<Left>",  lambda e: self._prev())
        self.root.bind("<Tab>",   lambda e: self._cycle_class())
        self.root.bind("c",       lambda e: self._clear_clicks())
        self.root.bind("C",       lambda e: self._clear_clicks())

        self._update_tab_styles()

    # ── Переключение классов ──────────────────────────────────────────────────

    def _switch_class(self, key):
        self.current_class_key.set(key)
        self._on_class_change()

    def _cycle_class(self):
        keys = list(CLASSES.keys())
        cur = self.current_class_key.get()
        self._switch_class(keys[(keys.index(cur) + 1) % len(keys)])

    def _on_class_change(self):
        self._update_tab_styles()
        self.hint_label.config(text=self.cls["hint"])
        self.digit_entry.config(fg=self.accent, insertbackground=self.accent)
        self.btn_save.config(fg=self.accent)
        # Ставим дефолтный режим для класса
        self.bbox_mode.set(self.cls["default_mode"])
        self._clear_clicks(render=False)
        self._load_image_list()
        if self.image_paths:
            self._show_image(0)
        else:
            self._show_empty()
        self.digit_entry.focus()

    def _update_tab_styles(self):
        cur = self.current_class_key.get()
        for key, btn in self.tab_buttons.items():
            btn.config(
                bg="#1a1a2e" if key == cur else "#0e0e14",
                fg=CLASSES[key]["color"] if key == cur else "#333355",
            )

    # ── Клики на canvas ───────────────────────────────────────────────────────

    def _on_canvas_click(self, event):
        if self.bbox_mode.get() != "click" or self.current_img is None:
            return
        # Переводим координаты canvas → изображение
        img_x = (event.x - self._img_offset_x) / self._img_scale
        h, w = self.current_img.shape[:2]
        img_x = max(1, min(int(img_x), w - 1))
        self.click_splits.append(img_x)
        self._update_boxes_from_clicks()
        self._render_canvas()

    def _on_canvas_right_click(self, event):
        if self.bbox_mode.get() != "click" or not self.click_splits:
            return
        # Удаляем ближайший разделитель к месту клика
        img_x = (event.x - self._img_offset_x) / self._img_scale
        closest = min(range(len(self.click_splits)),
                      key=lambda i: abs(self.click_splits[i] - img_x))
        self.click_splits.pop(closest)
        self._update_boxes_from_clicks()
        self._render_canvas()

    def _clear_clicks(self, render=True):
        self.click_splits = []
        self._update_boxes_from_clicks()
        if render:
            self._render_canvas()

    def _update_boxes_from_clicks(self):
        if self.current_img is None:
            return
        h, w = self.current_img.shape[:2]
        n_splits = len(self.click_splits)
        n_digits = n_splits + 1
        self.current_boxes = splits_to_boxes(self.click_splits, w)
        self.bbox_status.config(
            text=f"🖱 Разделителей: {n_splits}  →  {n_digits} цифр\n"
                 f"ЛКМ — добавить, ПКМ — удалить",
            fg="#ffaa44" if n_splits == 0 else "#44ff88",
        )

    # ── Режим bbox ────────────────────────────────────────────────────────────

    def _on_mode_change(self):
        self._clear_clicks(render=False)
        self._recompute_boxes()

    # ── Загрузка файлов ───────────────────────────────────────────────────────

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с кропами")
        if folder:
            CLASSES[self.current_class_key.get()]["crops_dir"] = folder
            self._load_image_list()
            if self.image_paths:
                self._show_image(0)

    def _load_image_list(self):
        crops_dir  = self.cls["crops_dir"]
        labels_dir = self.cls["labels_dir"]
        p = Path(crops_dir)
        if not p.exists():
            self.image_paths = []
            self._update_progress()
            return
        already_done = {Path(f).stem for f in Path(labels_dir).glob("*.txt")}
        self.image_paths = sorted([
            str(f) for f in p.iterdir()
            if f.suffix.lower() in EXTENSIONS
            and f.stem not in already_done
        ])
        self.current_idx = 0
        self._update_progress()

    def _update_progress(self):
        done = len(list(Path(self.cls["labels_dir"]).glob("*.txt")))
        self.progress_var.set(
            f"{self.cls['label']}   осталось: {len(self.image_paths)}  |  готово: {done}"
        )

    # ── Отображение ───────────────────────────────────────────────────────────

    def _show_image(self, idx):
        if not self.image_paths:
            self._show_empty()
            return
        self.current_idx = max(0, min(idx, len(self.image_paths) - 1))
        img_bgr = cv2.imread(self.image_paths[self.current_idx])
        if img_bgr is None:
            self._skip()
            return
        self.current_img = img_bgr
        self.digit_var.set("")
        self._clear_clicks(render=False)
        self._recompute_boxes()
        self.filename_label.config(text=Path(self.image_paths[self.current_idx]).name)
        self._update_len_label()
        self.digit_entry.focus()

    def _recompute_boxes(self):
        if self.current_img is None:
            return
        mode = self.bbox_mode.get()
        h, w = self.current_img.shape[:2]

        if mode == "click":
            self._update_boxes_from_clicks()
            self._render_canvas()
            return

        digit_str = self.digit_var.get().strip()
        n = len(digit_str) if digit_str else (self.cls["fixed_len"] or 5)

        if mode == "auto":
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            boxes = find_digit_boxes_by_projection(gray)
            if len(boxes) == n:
                self.current_boxes = boxes
                self.bbox_status.config(text=f"✅ Авто: {len(boxes)} групп", fg="#44ff88")
            else:
                self.current_boxes = uniform_split(w, n)
                self.bbox_status.config(
                    text=f"⚠️ Авто: {len(boxes)} → равномерный ({n})",
                    fg="#ffaa44")
        else:  # uniform
            self.current_boxes = uniform_split(w, n)
            self.bbox_status.config(text=f"Равномерно: {n} частей", fg="#888aaa")

        self._render_canvas()

    def _render_canvas(self):
        if self.current_img is None:
            return
        digit_str = self.digit_var.get().strip()
        img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        h, w = self.current_img.shape[:2]

        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 200)
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((nw, nh), Image.LANCZOS)

        # Сохраняем параметры масштаба для перевода кликов
        self._img_scale = scale
        self._img_offset_x = (cw - nw) // 2
        self._img_offset_y = (ch - nh) // 2

        mode = self.bbox_mode.get()

        if mode == "click":
            # Рисуем разделители
            splits_scaled = [int(x * scale) + self._img_offset_x
                             for x in self.click_splits]
            # Рисуем разделители относительно картинки
            splits_img_space = [int(x * scale) for x in self.click_splits]
            draw_splits(pil_img, splits_img_space, self.accent)

            # Если цифры введены и количество совпадает — рисуем боксы
            if (digit_str.isdigit() and len(digit_str) > 0
                    and len(digit_str) == len(self.current_boxes)):
                scaled_boxes = [(int(x1 * scale), int(x2 * scale))
                                for x1, x2 in self.current_boxes]
                draw_preview(pil_img, scaled_boxes, digit_str, self.accent)
        else:
            scaled_boxes = [(int(x1 * scale), int(x2 * scale))
                            for x1, x2 in self.current_boxes]
            is_valid = (
                digit_str.isdigit() and len(digit_str) > 0 and
                len(digit_str) == len(self.current_boxes) and
                (self.cls["fixed_len"] is None or
                 len(digit_str) == self.cls["fixed_len"])
            )
            if is_valid:
                draw_preview(pil_img, scaled_boxes, digit_str, self.accent)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(self._img_offset_x, self._img_offset_y,
                                 anchor="nw", image=self.photo)

    def _show_empty(self):
        self.current_img = None
        self.canvas.delete("all")
        self.canvas.create_text(300, 150,
                                text="Нет изображений\nВыберите папку с кропами",
                                fill="#222244", font=("Courier New", 12),
                                justify="center")
        self.progress_var.set(f"{self.cls['label']}   — всё размечено 🎉")

    # ── Ввод ──────────────────────────────────────────────────────────────────

    def _update_len_label(self):
        val = self.digit_var.get()
        fixed = self.cls["fixed_len"]
        mode = self.bbox_mode.get()
        if fixed:
            self.len_label.config(text=f"{len(val)} / {fixed} цифр")
        elif mode == "click":
            n_boxes = len(self.current_boxes)
            self.len_label.config(
                text=f"{len(val)} цифр  (боксов: {n_boxes})")
        else:
            self.len_label.config(text=f"{len(val)} цифр")

    def _on_digit_change(self, event=None):
        val = self.digit_var.get()
        fixed = self.cls["fixed_len"]
        max_len = fixed if fixed else 20
        clean = "".join(c for c in val if c.isdigit())[:max_len]
        if clean != val:
            self.digit_var.set(clean)
            self.digit_entry.icursor(len(clean))
        self._update_len_label()
        if self.bbox_mode.get() != "click":
            if self.cls["fixed_len"] is None:
                self._recompute_boxes()
            else:
                self._render_canvas()
        else:
            self._render_canvas()

    # ── Действия ─────────────────────────────────────────────────────────────

    def _save_and_next(self):
        digit_str = self.digit_var.get().strip()
        fixed = self.cls["fixed_len"]

        if not digit_str.isdigit() or len(digit_str) == 0:
            self._flash_entry()
            return
        if fixed and len(digit_str) != fixed:
            self._flash_entry()
            return

        # В режиме кликов — количество боксов должно совпадать с цифрами
        if self.bbox_mode.get() == "click":
            if len(self.current_boxes) != len(digit_str):
                self.bbox_status.config(
                    text=f"⚠️ Боксов {len(self.current_boxes)}, цифр {len(digit_str)}",
                    fg="#ff4444")
                self._flash_entry()
                return

        path = self.image_paths[self.current_idx]
        stem = Path(path).stem
        h, w = self.current_img.shape[:2]

        if len(self.current_boxes) != len(digit_str):
            self.current_boxes = uniform_split(w, len(digit_str))

        lines = boxes_to_yolo(self.current_boxes, w, h, digit_str)
        label_path = Path(self.cls["labels_dir"]) / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        self.image_paths.pop(self.current_idx)
        self._update_progress()

        if not self.image_paths:
            self._show_empty()
        else:
            self._show_image(min(self.current_idx, len(self.image_paths) - 1))

    def _flash_entry(self):
        self.digit_entry.config(bg="#2a0808")
        self.root.after(300, lambda: self.digit_entry.config(bg="#080810"))

    def _skip(self):
        if self.image_paths:
            self._show_image((self.current_idx + 1) % len(self.image_paths))

    def _prev(self):
        if self.image_paths:
            self._show_image((self.current_idx - 1) % len(self.image_paths))

    def _delete_current(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        if messagebox.askyesno("Удалить?", f"Удалить файл?\n{Path(path).name}"):
            os.remove(path)
            self.image_paths.pop(self.current_idx)
            self._update_progress()
            if not self.image_paths:
                self._show_empty()
            else:
                self._show_image(min(self.current_idx, len(self.image_paths) - 1))


def main():
    root = tk.Tk()
    app = DigitLabelerApp(root)
    root.bind("<Configure>",
              lambda e: app._render_canvas() if e.widget == root else None)
    root.mainloop()


if __name__ == "__main__":
    main()