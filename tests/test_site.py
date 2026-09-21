"""Тесты лендинга «Атлас типов»: данные сайта согласованы с итоговым пакетом outputs/final, сборка без пустых плейсхолдеров."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site_src/data/site_data.json"
FINAL = ROOT / "outputs/final/mo_table.csv"
SITE = ROOT / "site/index.html"

pytestmark = pytest.mark.skipif(not (DATA.exists() and FINAL.exists()), reason="нет данных сайта или итогового пакета")


@pytest.fixture(scope="module")
def data():
    return json.load(open(DATA, encoding="utf-8"))


def test_mo_match_final_table(data):
    final = pd.read_csv(FINAL).set_index("id")
    mo = {m["id"]: m for m in data["mo"]}
    assert len(mo) == len(final) == 2016
    assert set(mo) == set(final.index)
    for i, row in final.iterrows():
        assert mo[i]["ma"] == row["macro"] and mo[i]["su"] == row["sub"]
        assert mo[i]["n"] == row["name"]
    # размеры типов на сайте = размеры модальных типов итоговой таблицы
    assert data["types"]["macro"]["sizes"] == final["macro"].value_counts().sort_index().tolist()
    assert data["types"]["sub"]["sizes"] == final["sub"].value_counts().sort_index().tolist()


def test_sequences_and_geometry(data):
    K = {"macro": 4, "sub": 6, "beh": 5}
    for m in data["mo"]:
        for key, lens in (("ms", "macro"), ("ss", "sub"), ("bs", "beh")):
            s = m[key]
            assert len(s) == 24 and set(s) <= set("0123456789"[: K[lens]])
        assert len(m["rl"]) == 24 and len(m["sh"]) == 6 and abs(sum(m["sh"]) - 1) < 1e-3
        cnt = Counter(m["ms"]); top = max(cnt.values())
        assert m["ma"] == min(int(k) for k, c in cnt.items() if c == top)  # модальный тип, при равенстве — меньший номер
    assert set(data["geo"]) == {str(m["id"]) for m in data["mo"]}
    fw, fh = data["frame"]["w"], data["frame"]["h"]
    for polygons in data["geo"].values():  # МО → полигоны → кольца → плоский список x,y в кадре карты
        for rings in polygons:
            for ring in rings:
                assert len(ring) >= 6 and len(ring) % 2 == 0
                xs, ys = ring[0::2], ring[1::2]
                assert min(xs) >= -1 and max(xs) <= fw + 1 and min(ys) >= -1 and max(ys) <= fh + 1


def test_types_block(data):
    for lens, k in (("macro", 4), ("sub", 6), ("beh", 5)):
        t = data["types"][lens]
        assert len(t["names"]) == len(t["sizes"]) == k and len(t.get("colors", t["names"])) == k
        assert sum(t["sizes"]) == 2016
        assert len(t["profiles"]) == k and all(len(p["v"]) == len(data["types"]["feature_order"]) for p in t["profiles"])
    assert data["graph"]["pos"] and data["graph"]["edges"]


@pytest.mark.skipif(not SITE.exists(), reason="сайт не собран")
def test_built_site_has_no_placeholders():
    html = SITE.read_text(encoding="utf-8")
    assert not any(ph in html for ph in ("__FONTS__", "__CSS__", "__VENDOR__", "__DATA__", "__APP__", "__FIG0__"))
    assert 'id="stage"' in html and 'id="atlas"' in html and "CHAPTERS" in html
    assert html.count("@font-face") >= 6 and "font-family:'Unbounded'" in html and "d3.sankey" in html
    assert "chapter-card" in html and "class=\"rail" in html
