# Сторонний код (не хранится в репозитории)

- `canus.py` — CANUS, S. Shalileh (IEEE Access, 2025), https://github.com/Sorooshi/CANUS (файл без явной лицензии,
  поэтому в репозиторий не включён). Скачивается командой `make third_party` (входит в `make env`):
  `https://raw.githubusercontent.com/Sorooshi/CANUS/main/canus.py`, sha256 `dbc14daff4f6b2a4b9d377deb1bef5c76a2c390678d47922fa337983e0faa7a3`.
  Используется только в сравнении методов (`scripts/method_zoo.py`); без файла зоопарк пропускает CANUS.
