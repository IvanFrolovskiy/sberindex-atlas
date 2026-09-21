"""Кейсы реальных шоков: как ведут себя уровень трат, структура и тип у затронутых МО.

В панели СберИндекса нет 12 регионов (Белгородская, Брянская, Курская, Воронежская, Ростовская области,
Краснодарский край, Крым, Севастополь, Дагестан, Чечня, Ингушетия, Бурятия), поэтому кейсы приграничных
регионов невозможны. Взяты весенние паводки 2024 г.: Оренбургская (Орск, Оренбург, Новотроицк — ЧС федерального
уровня 7.04.2024), Курганская (Курган, Кетовский район) и Тюменская (Ишим) области.
Для каждого МО: относительный уровень (к медиане месяца), г/г, отклонение от медианы своего типа, тип по месяцам.

Запуск: .venv312/bin/python scripts/case_studies.py --labels outputs/kefrin_t/labels_affect_fe_K4_learned_windowlog.csv
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mo_types import data, features  # noqa: E402

CASES = {
    "Паводок в Оренбургской области (ЧС федерального уровня, апрель 2024)": {
        "region": "Оренбургская область", "names": ["город Орск", "город Оренбург", "Новотроицк"], "event": "2024-04"},
    "Паводок в Курганской области (апрель–май 2024)": {
        "region": "Курганская область", "names": ["город Курган", "Кетовский"], "event": "2024-04"},
    "Паводок в Тюменской области (Ишим, апрель–май 2024)": {
        "region": "Тюменская область", "names": ["город Ишим", "Ишимский"], "event": "2024-04"},
}
MISSING_REGIONS = ["Белгородская область", "Брянская область", "Курская область", "Воронежская область", "Ростовская область",
                   "Краснодарский край", "Республика Крым", "Севастополь", "Республика Дагестан", "Чеченская Республика",
                   "Республика Ингушетия", "Республика Бурятия"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/kefrin_t/labels_affect_fe_K4_learned_windowlog.csv")
    args = ap.parse_args()
    cfg = data.load_config()
    panel, attrs = data.load_processed(cfg)
    feat = features.monthly_features(panel)
    lab = pd.read_csv(data.ROOT / args.labels, index_col=0)
    lab.index = lab.index.astype(int)
    months = list(lab.columns)
    rel = feat["rel_level"].unstack("date")
    rel.columns = months
    yoy = feat["yoy_log"].unstack("date")
    yoy.columns = months
    out = data.ROOT / "outputs" / "case_studies"
    out.mkdir(parents=True, exist_ok=True)
    report = ["# Кейсы шоков", "", "В панели нет регионов: " + ", ".join(MISSING_REGIONS) + "."]
    cases_json = []
    for title, c in CASES.items():
        sub = attrs[attrs.region_name == c["region"]]
        report.append(f"\n## {title}\nМО региона в панели: {len(sub)}")
        ev = c["event"]
        ei = months.index(ev)
        case = {"title": title, "event": ev, "region": c["region"], "mo": []}
        for nm in c["names"]:
            hit = sub[sub.name.str.contains(nm, na=False)]
            for tid, row in hit.iterrows():
                if tid not in rel.index:
                    continue
                r = rel.loc[tid]
                types = lab.loc[tid].to_numpy()
                same = lab.index[lab[ev] == lab.loc[tid, ev]]
                med = rel.loc[rel.index.isin(same)].median()
                dev = (r - med)
                win = months[max(ei - 3, 0): ei + 7]
                report.append(f"- **{row['name']}** (тип по месяцам {''.join(str(int(x)) for x in types)}); окно {win[0]}…{win[-1]}\n"
                              f"  - относительный уровень: {' '.join(f'{r[m]:+.2f}' for m in win)}\n"
                              f"  - г/г: {' '.join(f'{np.exp(yoy.loc[tid, m]) - 1:+.0%}' if pd.notna(yoy.loc[tid, m]) else '  —' for m in win)}\n"
                              f"  - отклонение от медианы своего типа: {' '.join(f'{dev[m]:+.2f}' for m in win)}")
                case["mo"].append({"id": int(tid), "name": str(row["name"]), "types": [int(x) for x in types],
                                   "rel": [round(float(x), 3) for x in r.to_numpy()],
                                   "dev": [round(float(x), 3) for x in dev.to_numpy()],
                                   "yoy": [None if pd.isna(x) else round(float(np.exp(x) - 1), 3) for x in yoy.loc[tid].to_numpy()]})
        cases_json.append(case)
    txt = "\n".join(report)
    (out / "case_studies.md").write_text(txt, encoding="utf-8")
    json.dump({"months": months, "missing_regions": MISSING_REGIONS, "cases": cases_json},
              open(out / "cases.json", "w"), ensure_ascii=False, indent=1)
    print(txt)


if __name__ == "__main__":
    main()
