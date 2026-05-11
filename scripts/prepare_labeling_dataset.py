import os

files = sorted(os.listdir('database/raw_photos/'))

# Берём каждый 3-й файл 
step = 3  
selected = files[::step]

with open('out_put/markup_order.txt', 'w') as f:
    for name in selected:
        f.write(name + '\n')

print(f"Файлов для разметки: {len(selected)}")




import os
import shutil

src = 'database/raw_photos'
dst = 'database/images_to_label'
order_file = 'out_put/markup_order.txt'

os.makedirs(dst, exist_ok=True)

with open(order_file, 'r') as f:
    files = [line.strip() for line in f.readlines()]

for fname in files:
    src_path = os.path.join(src, fname)
    dst_path = os.path.join(dst, fname)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Не найден: {fname}")

print(f"Скопировано: {len(os.listdir(dst))} файлов")