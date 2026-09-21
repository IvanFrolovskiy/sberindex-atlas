"""Сводка KEFRiN-T по источникам графов: kNN vs обученные (пооконный LogModel, TGFA).

Читает outputs/kefrin_t/summary_*.csv и печатает по (граф, K, метод) устойчивость и качество.
Запуск: .venv312/bin/python scripts/compare_graphs.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from mo_types import data  # noqa: E402

OUT = data.ROOT / "outputs" / "kefrin_t"


def main():
    frames = []
    for f in sorted(OUT.glob("summary_*.csv")):
        tag = f.stem.replace("summary_", "")
        d = pd.read_csv(f)
        d["graph"] = tag
        frames.append(d)
    t = pd.concat(frames, ignore_index=True)
    t = t[t.method.isin(["kefrin_warm", "kefrin_t_affect_fe", "temporal_leiden_w2.0"])]
    cols = ["graph", "method", "K", "ami_median", "switch_median", "switch_max", "share_never_switch", "SW", "CH", "AVI", "MQ", "Q"]
    t = t[[c for c in cols if c in t.columns]].sort_values(["K", "method", "graph"])
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    t.to_csv(OUT / "compare_graphs.csv", index=False)


if __name__ == "__main__":
    main()
