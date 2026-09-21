"""Загрузка всех исходных данных (воспроизводимость с нуля).

1. Архив хакатона СберИндекса Data→Sense (consumption/market_access/connection, territory_id) — публичная страница
   https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim
2. Справочник границ МО СберИндекса (xlsx + gpkg, CC BY-SA 4.0) — https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities
3. Показатели Росстата (БД ПМО) из каталога «Если быть точным» (CC BY 4.0) — выборочно по HTTP Range из больших zip.
4. Выгрузки API СберИндекса (parquet) — для сверки.
Запуск: .venv312/bin/python scripts/download_data.py
Проверено 21.09.2026: все файлы скачиваются и побайтно совпадают с использованными (sha256).
"""
import hashlib
import struct
import subprocess
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HACKATHON = "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"
BORDERS_RAR = "https://www.sberbank.com/common/files/t_dict_municipal.rar"
BORDERS_META = "https://www.sberbank.ru/common/img/uploaded/files/pdf/sberindex/metadata_municipal_dict_sberindex_2.pdf"
API = "https://sberindex.ru/api/dataset/v1/download/{slug}/parquet"
ROSSTAT_BASE = "https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/data_bdmo_118_v20250918/by_indicator/"
ROSSTAT = {  # (раздел zip, индикатор): показатель
    (31, "Y48112027"): "оценка численности населения на 1 января",
    (32, "Y48423007"): "среднемесячная зарплата по ОКВЭД2 (без МП)",
    (32, "Y48423005"): "среднесписочная численность по ОКВЭД2 (без МП)",
}


def fetch(url, dest: Path, insecure=False):
    if dest.exists() and dest.stat().st_size > 0:
        print("есть", dest.name)
        return
    cmd = ["curl", "-sL", "-o", str(dest), url] + (["-k"] if insecure else [])
    subprocess.run(cmd, check=True)
    print("скачано", dest.name, f"{dest.stat().st_size / 1e6:.1f} МБ", hashlib.sha256(dest.read_bytes()).hexdigest()[:16])


def get_range(url, a, b, tries=8):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            return urllib.request.urlopen(req, timeout=180).read()
        except Exception:
            time.sleep(3 * (i + 1))
    raise RuntimeError("range failed")


def zip_entries(url):
    """Оглавление zip-архива без его скачивания: читаем хвост, из него — центральный каталог.

    Архивы Росстата весят гигабайты, а нужны из них два-три csv, поэтому берём по HTTP Range
    сначала последние 64 КБ (там лежит оглавление), затем сам каталог. Смещения — из спецификации
    формата (APPNOTE.TXT): PK\x06\x07 — указатель ZIP64, PK\x05\x06 — обычный конец каталога,
    PK\x01\x02 — заголовок записи в каталоге."""
    req = urllib.request.Request(url, method="HEAD")
    n = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    tail = get_range(url, n - 65536 - 20, n - 1)
    i = tail.rfind(b"PK\x06\x07")
    if i >= 0:  # ZIP64: в указателе лежит смещение настоящего конца каталога
        off = struct.unpack("<Q", tail[i + 8:i + 16])[0]
        e = get_range(url, off, off + 55)
        cd_size, cd_off = struct.unpack("<QQ", e[40:56])
    else:
        i = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    cd = get_range(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":  # имя, метод сжатия, размеры, смещение записи
        comp, = struct.unpack("<H", cd[p + 10:p + 12])
        csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
        nl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho, = struct.unpack("<I", cd[p + 42:p + 46])
        out.append((cd[p + 46:p + 46 + nl].decode("utf-8", "replace"), comp, csz, usz, lho))
        p += 46 + nl + el + cl
    return out


def extract_rosstat(section, code, years=("2022", "2023", "2024")):
    dest = RAW / "rosstat" / f"{code}_2022_2024.csv"
    if dest.exists():
        print("есть", dest.name)
        return
    dest.parent.mkdir(exist_ok=True)
    url = ROSSTAT_BASE + f"data_section{section}_112_v20250918.zip"
    name, comp, csz, usz, lho = [e for e in zip_entries(url) if e[0] == f"data_{code}_112_v20250918.csv"][0]
    hdr = get_range(url, lho, lho + 29)
    nl, el = struct.unpack("<HH", hdr[26:30])
    start = lho + 30 + nl + el
    d = zlib.decompressobj(-15)
    buf, header, kept, yi = b"", None, 0, None
    with open(dest, "w", encoding="utf-8") as fo:
        for a in range(start, start + csz, 4_000_000):
            buf += d.decompress(get_range(url, a, min(a + 4_000_000, start + csz) - 1))
            lines = buf.split(b"\n")
            buf = lines[-1]
            for ln in lines[:-1]:
                t = ln.decode("utf-8", "replace")
                if header is None:
                    header = t
                    fo.write(t + "\n")
                    yi = t.split(";").index("year")
                    continue
                f = t.split(";")
                if len(f) > yi and f[yi] in years:
                    fo.write(t + "\n")
                    kept += 1
    print("Росстат", code, "строк", kept)


def main():
    fetch(HACKATHON, RAW / "hackathonlicence.zip")
    (RAW / "hackathon").mkdir(exist_ok=True)  # архив содержит папку hackathonlicence/ → data/raw/hackathon/hackathonlicence/*.parquet
    subprocess.run(["bsdtar", "-xf", str(RAW / "hackathonlicence.zip"), "-C", str(RAW / "hackathon")], check=True)
    fetch(BORDERS_RAR, RAW / "mo_borders.rar")
    subprocess.run(["bsdtar", "-xf", str(RAW / "mo_borders.rar"), "-C", str(RAW)], check=True)
    fetch(BORDERS_META, RAW / "metadata_municipal_dict_sberindex_2.pdf")
    for slug in ["potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij", "indeks-mobilnosti"]:
        fetch(API.format(slug=slug), RAW / f"{slug}.parquet", insecure=True)
    for (section, code), desc in ROSSTAT.items():
        extract_rosstat(section, code)
    print("готово")


if __name__ == "__main__":
    main()
