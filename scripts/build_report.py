"""Сборка методологического отчёта: report/methodology.md → report/methodology.html → report/methodology.pdf.

Markdown переводится в HTML библиотекой markdown (таблицы, сноски), стиль — печатный A4, PDF печатает headless Chrome.
Картинки подставляются как есть по относительным путям, поэтому HTML открывается и сам по себе.

Запуск: .venv312/bin/python scripts/build_report.py  (нужен браузер Chrome/Chromium; без него собирается только HTML)
"""
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
p { margin: 5pt 0; text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 6pt 0 8pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 2.5pt 4pt; vertical-align: top; }
th { background: #f2f2f2; text-align: left; }
ul { margin: 4pt 0; padding-left: 16pt; }
li { margin: 2pt 0; }
code { font-family: Menlo, monospace; font-size: 9pt; background: #f5f5f5; padding: 0 2px; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; border: 1px solid #e5e5e5; }
p > em:only-child { display: block; font-size: 9pt; color: #444; margin: 0 0 8pt; }
"""


def find_chrome():
    """Путь к Chrome/Chromium: сначала известные места macOS и Linux, затем PATH. None, если браузера нет."""
    known = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
             "/Applications/Chromium.app/Contents/MacOS/Chromium",
             "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
             "/snap/bin/chromium"]
    for p in known:
        if Path(p).exists():
            return p
    import shutil
    for name in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"]:
        p = shutil.which(name)
        if p:
            return p
    return None


def stamp(pdf: Path, title: str):
    """Проставить в готовый PDF заголовок и автора: Chrome берёт только <title>, остальное пишем сами."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        return
    r = PdfReader(pdf)
    w = PdfWriter()
    for page in r.pages:
        w.add_page(page)
    w.add_metadata({"/Title": title, "/Author": "Иван Фроловский",
                    "/Subject": "Онлайн-конкурс СберИндекса 2026, направление «Кластеризация»"})
    with open(pdf, "wb") as fh:
        w.write(fh)


def main():
    md = (REPORT / "methodology.md").read_text(encoding="utf-8")
    html_body = markdown.markdown(md, extensions=["tables", "toc", "attr_list"])
    html = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<title>Методологический отчёт</title><style>{CSS}</style></head><body>{html_body}</body></html>')
    out_html = REPORT / "methodology.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"report/methodology.html: {len(html) / 1000:.0f} КБ")

    chrome = find_chrome()
    if not chrome:
        print("Chrome/Chromium не найден — PDF не собран (HTML готов, его можно напечатать вручную)")
        return
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                    f"--print-to-pdf={REPORT / 'methodology.pdf'}", f"file://{out_html}"], check=True, capture_output=True)
    stamp(REPORT / "methodology.pdf", "Типы локальных экономик России: методологический отчёт")
    size = (REPORT / "methodology.pdf").stat().st_size / 1e6
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(REPORT / "methodology.pdf").pages)
        print(f"report/methodology.pdf: {size:.2f} МБ, страниц: {pages}")
    except Exception:
        print(f"report/methodology.pdf: {size:.2f} МБ")


if __name__ == "__main__":
    sys.exit(main())
