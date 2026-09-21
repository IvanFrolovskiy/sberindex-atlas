"""Внешняя проверка типологии по индексу покупательской мобильности СберИндекса (второй набор из брифа).

Набор `indeks-mobilnosti` (API СберИндекса, parquet): значение в км по МО на две даты (конец 2024 и 2025 г.),
без territory_id — сопоставление по уникальному названию МО. Считаем медианы по макротипам, η², тест Краскела–Уоллиса,
корреляции с доступностью рынков, населением и тратами.

Запуск: .venv312/bin/python scripts/mobility_check.py
Выход: outputs/mobility/{by_type.csv, summary.json}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from mo_types import data  # noqa: E402

OUT = data.ROOT / "outputs" / "mobility"
NAMES = {0: "Крупные и средние города", 1: "Северные и ресурсные", 2: "Столичные агломерации", 3: "Сельские и малые"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = data.ROOT / "data/raw/indeks-mobilnosti.parquet"
    if not src.exists():
        print("нет файла indeks-mobilnosti.parquet — пропуск")
        return
    m = pd.read_parquet(src)
    mo = pd.read_csv(data.ROOT / "outputs/final/mo_table.csv")
    cnt = mo.name.value_counts()
    uniq = mo[mo.name.map(cnt) == 1].set_index("name")
    periods = sorted(m.period.unique())
    first = m[m.period == periods[0]].set_index("ref_area")["value"].rename("mob_first")
    last = m[m.period == periods[-1]].set_index("ref_area")["value"].rename("mob_last")
    j = uniq.join(first, how="inner").join(last, how="inner")
    j["type"] = j.macro.map(NAMES)
    by_type = j.groupby("type").agg(n=("mob_first", "size"), median_first=("mob_first", "median"), median_last=("mob_last", "median"),
                                    q25=("mob_first", lambda v: v.quantile(.25)), q75=("mob_first", lambda v: v.quantile(.75)))
    by_type.to_csv(OUT / "by_type.csv")
    groups = [g.mob_first.to_numpy() for _, g in j.groupby("type") if len(g) > 5]
    kw = stats.kruskal(*groups)
    grand = j.mob_first.mean()
    eta2 = sum(len(x) * (x.mean() - grand) ** 2 for x in groups) / ((j.mob_first - grand) ** 2).sum()
    summary = {"periods": periods, "matched": int(len(j)), "available": int(m.ref_area.nunique()),
               "by_type": by_type.round(3).to_dict(orient="index"), "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue), "eta2": float(eta2),
               "corr_market_access": float(j.mob_first.corr(j.market_access)), "corr_log_population": float(j.mob_first.corr(np.log(j.population))),
               "corr_log_total": float(j.mob_first.corr(np.log(j.total_2024))), "corr_first_last": float(j.mob_first.corr(j.mob_last))}
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(f"сопоставлено {len(j)} из {m.ref_area.nunique()} МО; периоды {periods}")
    print(by_type.round(2).to_string())
    print(f"η² по макротипам {eta2:.3f}; Краскел–Уоллис H={kw.statistic:.1f}, p={kw.pvalue:.1e}; корреляции: доступность {summary['corr_market_access']:.2f}, "
          f"log население {summary['corr_log_population']:.2f}, log траты {summary['corr_log_total']:.2f}; 2024 vs 2025 {summary['corr_first_last']:.2f}")


if __name__ == "__main__":
    main()
