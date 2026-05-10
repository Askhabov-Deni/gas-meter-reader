
## Быстрый старт (Windows PowerShell)

```powershell
# 1. Клонировать
git clone https://github.com/Askhabov-Deni/gas-meter-reader.git
cd gas-meter-reader

# 2. Виртуальное окружение
python -m venv .venv
.venv\Scripts\activate

# 3. Зависимости
pip install -r requirements.txt

# 4. Проверить
python scripts/test.py


## Быстрый старт (linux)

# 1. Клонировать
git clone git@github.com:Askhabov-Deni/gas-meter-reader.git
cd gas-meter-reader

# 2. Виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 3. Зависимости
pip install -r requirements.txt

# 4. Проверить
python scripts/test.py