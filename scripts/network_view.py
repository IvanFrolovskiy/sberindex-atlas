"""Сетевой взгляд: что находит сама обученная сеть сходства потребления без признаков.

Temporal Leiden (Mucha et al. 2010; leidenalg.find_partition_temporal) на 24 обученных пооконных графах с
межслойным весом ω = 2 и разрешением 0,5 — тот же вариант, что в сводке kefrin_t. Сообщества сравниваются с
экономической типологией (макро и подтипы) и с поведенческой: кросс-таблицы, чистота, состав сообществ
(доля горожан, население, зарплата, доступность, долгота).

Запуск: .venv312/bin/python scripts/network_view.py
Выход: outputs/network_view/{crosstab_macro.csv, crosstab_sub.csv, crosstab_beh.csv, communities.csv, summary.json}
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari  # noqa: E402

from mo_types import data, graphs  # noqa: E402

OUT = data.ROOT / "outputs" / "network_view"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    types = yaml.safe_load(open(data.ROOT / "configs" / "types.yaml", encoding="utf-8"))
    E = pd.read_csv(data.ROOT / types["labels"]["macro"], index_col=0)
    E.index = E.index.astype(int)
    S = pd.read_csv(data.ROOT / types["labels"]["sub"], index_col=0).reindex(E.index)
    B = pd.read_csv(data.ROOT / types["labels"]["beh"], index_col=0).reindex(E.index)
    mo = pd.read_csv(data.ROOT / "outputs/final/mo_table.csv").set_index("id").reindex(E.index)
    gdir = data.ROOT / types["graph"]
    As = [sparse.load_npz(gdir / f"A_t{t:02d}.npz") for t in range(E.shape[1])]
    M = graphs.temporal_leiden(As, interslice_weight=2.0, resolution=0.5, seed=42)  # T×N
    lab = E.to_numpy().T
    per = [ami(lab[t], M[t]) for t in range(len(As))]
    modal_M = pd.DataFrame(M).mode(axis=0).iloc[0].to_numpy().astype(int)
    # перенумеровать сообщества по размеру
    order = np.argsort(-np.bincount(modal_M))
    remap = {int(old): int(new) for new, old in enumerate(order)}
    modal_M = np.array([remap[int(x)] for x in modal_M])
    me = E.mode(axis=1)[0].astype(int).to_numpy()
    ms = S.mode(axis=1)[0].astype(int).to_numpy()
    mb = B.mode(axis=1)[0].astype(int).to_numpy()
    names_e = {int(k): v for k, v in types["names"]["macro"].items()}
    names_s = {int(k): v for k, v in types["names"]["sub"].items()}
    names_b = {int(k): v for k, v in types["names"]["beh"].items()}
    ct_e = pd.crosstab(pd.Series(me).map(names_e), modal_M)
    ct_s = pd.crosstab(pd.Series(ms).map(names_s), modal_M)
    ct_b = pd.crosstab(pd.Series(mb).map(names_b), modal_M)
    for name, ct in [("crosstab_macro", ct_e), ("crosstab_sub", ct_s), ("crosstab_beh", ct_b)]:
        ct.to_csv(OUT / f"{name}.csv")
    comm = pd.DataFrame({"community": modal_M, "lon": mo.lon, "population": mo.population, "wage": mo.wage,
                         "market_access": mo.market_access, "urban_share": mo.urban_share, "total_2024": mo.total_2024, "region": mo.region})
    prof = comm.groupby("community").agg(n=("lon", "size"), lon=("lon", "mean"), pop_median=("population", "median"),
                                        wage_median=("wage", "median"), access_median=("market_access", "median"),
                                        urban_median=("urban_share", "median"), total_median=("total_2024", "median"),
                                        regions=("region", "nunique"))
    prof["dominant_type"] = [ct_e[c].idxmax() for c in prof.index]
    prof["dominant_share"] = [float(ct_e[c].max() / ct_e[c].sum()) for c in prof.index]
    prof.to_csv(OUT / "communities.csv")
    summary = {
        "n_communities": int(len(np.unique(modal_M))), "ami_between_months_mean": float(np.mean(per)),
        "ami_with_macro": float(ami(me, modal_M)), "ari_with_macro": float(ari(me, modal_M)),
        "ami_with_sub": float(ami(ms, modal_M)), "ami_with_beh": float(ami(mb, modal_M)),
        "purity_macro_rows": (ct_e.max(axis=1) / ct_e.sum(axis=1)).round(3).to_dict(),
        "purity_communities": (ct_e.max(axis=0) / ct_e.sum(axis=0)).round(3).to_dict(),
        "communities": prof.round(3).to_dict(orient="index"),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    pd.set_option("display.width", 220)
    print(f"сообществ {summary['n_communities']}; AMI между месяцами {summary['ami_between_months_mean']:.3f}; "
          f"с макротипами AMI {summary['ami_with_macro']:.3f} ARI {summary['ari_with_macro']:.3f}; с подтипами AMI {summary['ami_with_sub']:.3f}; с поведенческими AMI {summary['ami_with_beh']:.3f}")
    print(ct_e.to_string()); print(ct_s.to_string()); print(ct_b.to_string())
    print(prof.round(2).to_string())


if __name__ == "__main__":
    main()
