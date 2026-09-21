"""Печатная версия лендинга: site/print.html → site/landing.pdf (A4 landscape).

Каждая глава истории снимается headless Chrome в режиме ?view=k (сцена + карточка шага), разделы исследователя — в режиме
?only=<id>. Кадры сжимаются в JPEG и собираются в один HTML, который тот же Chrome печатает в PDF.

Запуск: .venv312/bin/python scripts/build_print.py  (нужен Google Chrome; ≈ 2–3 мин)
"""
import base64
import io
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
from build_report import find_chrome, stamp  # общий поиск браузера и метаданные PDF (оба скрипта лежат в scripts/)
CHROME = find_chrome()
W, H = 1400, 900
SECTIONS = [("atlas", "Атлас: каждое МО на карте, по месяцам и по трём линзам", 1000),
            ("passport", "Паспорт муниципалитета", 1800),
            ("types", "Карточки типов", 1000),
            ("method", "Метод и проверки", 1200)]


def shot(url, out, h=H):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{h}",
                    "--virtual-time-budget=8000", f"--screenshot={out}", url], check=True, capture_output=True)


def jpeg_b64(im, quality=82):
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), im.size


def load(png):
    return Image.open(png).convert("RGB")


def trim_bottom(im, thr=238):
    """Обрезать светлую пустую полосу под прологом (обложка)."""
    g = im.convert("L"); w, h = g.size; px = g.load(); y = h
    while y > h // 2 and min(px[x, y - 1] for x in range(0, w, 7)) > thr:
        y -= 1
    return im.crop((0, 0, w, y))


def split(im, page_h=900):
    """Высокий кадр раздела → страницы по page_h пикселей (последняя — по фактической высоте, не короче 500)."""
    w, h = im.size
    if h <= page_h * 1.35:
        return [im]
    out, y = [], 0
    while y < h:
        y2 = min(h, y + page_h)
        if h - y2 < 500:
            y2 = h
        out.append(im.crop((0, y, w, y2))); y = y2
    return out


def step_count():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    # число шагов = число объектов {k:..., text:...} в CHAPTERS; надёжнее считать по DOM после загрузки
    with tempfile.TemporaryDirectory() as td:
        dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=5000", "--dump-dom",
                              f"file://{SITE / 'index.html'}"], check=True, capture_output=True, text=True).stdout
    return len(re.findall(r'id="step-\d+"', dom))


def main():
    if not CHROME:
        print("Chrome/Chromium не найден — печатная версия пропущена (лендинг собран)")
        return
    url = f"file://{SITE / 'index.html'}"
    n = step_count()
    pages = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shot(f"{url}?theme=light&nofx=1", td / "cover.png"); pages.append(("cover", "Атлас типов", jpeg_b64(trim_bottom(load(td / "cover.png")))))
        for k in range(n):
            shot(f"{url}?view={k}&theme=light&nosmooth&nofx=1", td / f"v{k}.png")
            pages.append(("story", f"История · шаг {k + 1} из {n}", jpeg_b64(load(td / f"v{k}.png"))))
            print(f"шаг {k + 1}/{n}", flush=True)
        for sid, title, h in SECTIONS:
            shot(f"{url}?theme=light&only={sid}&nofx=1", td / f"{sid}.png", h)
            parts = split(load(td / f"{sid}.png"))
            for j, im in enumerate(parts):
                pages.append(("section", title + (f" ({j + 1}/{len(parts)})" if len(parts) > 1 else ""), jpeg_b64(im)))
    body = []
    for kind, title, (b64, (w, h)) in pages:
        body.append(f'<section class="page {kind}"><img src="data:image/jpeg;base64,{b64}" alt="{title}">'
                    f'<div class="cap">{title}</div></section>')
    body.append('<section class="page last"><div class="txt"><h1>Атлас типов</h1><p>Интерактивная версия — <code>site/index.html</code> '
                '(один файл, открывается локально в любом браузере). Методологический отчёт — <code>report/methodology.pdf</code>. '
                'Код, конфигурации и инструкция запуска — в репозитории (<code>README.md</code>, <code>make env</code>, <code>make all</code>).</p>'
                '<p class="muted">Онлайн-конкурс СберИндекса, направление «Кластеризация». Данные СберИндекса: потребительские расходы по '
                'категориям, доступность рынков, уровень трат; контекст — Росстат (БДМО). Печатная версия собрана автоматически из лендинга.</p></div></section>')
    html = ("<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><title>Атлас типов — печатная версия</title><style>"
            "@page{size:A4 landscape;margin:0}html,body{margin:0;background:#fff;font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a}"
            ".page{width:297mm;height:210mm;page-break-after:always;position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center;background:#fff}"
            ".page img{max-width:100%;max-height:100%;display:block;object-fit:contain}"
            ".page.section img{max-height:200mm}"
            ".cap{position:absolute;right:8mm;bottom:4mm;font-size:8pt;color:#64748b}"
            ".page.last .txt{max-width:170mm;padding:20mm}.page.last h1{font-size:28pt;margin:0 0 8mm}.page.last p{font-size:12pt;line-height:1.5}"
            ".page.last .muted{color:#64748b;font-size:10pt}code{font-family:ui-monospace,Menlo,monospace;font-size:.92em}"
            "</style></head><body>" + "".join(body) + "</body></html>")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "print.html"; src.write_text(html, encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                        f"--print-to-pdf={SITE / 'landing.pdf'}", f"file://{src}"], check=True, capture_output=True)
    stamp(SITE / "landing.pdf", "Атлас типов: печатная версия")
    print(f"site/landing.pdf: {(SITE / 'landing.pdf').stat().st_size / 1e6:.2f} МБ, страниц: {len(pages) + 1}")


if __name__ == "__main__":
    main()
