"""Сборка автономного лендинга «Атлас типов» в один файл site/index.html.

Шаблон site_src/index.html + стили + встроенные шрифты (base64 woff2) + библиотеки (D3, d3-sankey) + данные
(site_src/data/site_data.json) + скрипты приложения (site_src/js/*.js по алфавиту).

Запуск: .venv312/bin/python scripts/build_site.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "site_src"
SITE = ROOT / "site"


def font_css():
    subsets = json.load(open(SRC / "fonts/subsets.json"))
    out = []
    for s in subsets:
        b64 = base64.b64encode((SRC / s["file"]).read_bytes()).decode()
        out.append(f"@font-face{{font-family:'{s.get('family', 'Inter')}';font-style:normal;font-weight:{s.get('weight', '400 700')};font-display:swap;"
                   f"src:url(data:font/woff2;base64,{b64}) format('woff2');unicode-range:{s['range']};}}")
    return "\n".join(out)


def main():
    SITE.mkdir(exist_ok=True)
    tpl = (SRC / "index.html").read_text(encoding="utf-8")
    css = (SRC / "styles.css").read_text(encoding="utf-8")
    data_json = (SRC / "data/site_data.json").read_text(encoding="utf-8")
    vendor = "\n".join((SRC / "vendor" / f).read_text(encoding="utf-8") for f in ["d3.v7.min.js", "d3-sankey.min.js"])
    app = "\n".join(p.read_text(encoding="utf-8") for p in sorted((SRC / "js").glob("*.js")))
    html = (tpl.replace("__FONTS__", font_css()).replace("__CSS__", css).replace("__VENDOR__", vendor)
            .replace("__DATA__", data_json).replace("__APP__", app))
    (SITE / "index.html").write_text(html, encoding="utf-8")
    pdf = ROOT / "report/methodology.pdf"
    if pdf.exists():
        (SITE / "methodology.pdf").write_bytes(pdf.read_bytes())
    print(f"site/index.html: {len(html.encode('utf-8')) / 1e6:.2f} МБ")


if __name__ == "__main__":
    main()
