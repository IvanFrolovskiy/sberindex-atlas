"""Разбор динамики типов по меткам KEFRiN-T: размеры типов по месяцам, кто и когда меняет тип,
матрица переходов, профили типов по правилу Миркина.

Запуск: .venv312/bin/python scripts/analyze_kefrin_t.py [--labels outputs/kefrin_t/labels_affect_K6_knn.csv]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mo_types import clustering, data, features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/kefrin_t/labels_affect_K6_knn.csv")
    ap.add_argument("--alphas", default="outputs/kefrin_t/alphas_knn.csv")
    args = ap.parse_args()
    cfg = data.load_config()
    panel, attrs = data.load_processed(cfg)
    lab = pd.read_csv(data.ROOT / args.labels, index_col=0)
    lab.index = lab.index.astype(int)
    months = list(lab.columns)
    L = lab.to_numpy()
    K = int(L.max()) + 1
    out = (data.ROOT / args.labels).parent
    tag = Path(args.labels).stem.replace("labels_", "")

    # α, β по месяцам
    if (data.ROOT / args.alphas).exists():
        al = pd.read_csv(data.ROOT / args.alphas)
        al = al[al.K == K]
        if "method" in al.columns:
            al = al[al.method == "kefrin_t_affect_fe"]
        print("α (признаки) и β (сеть) по месяцам:")
        print(al[["date", "alpha", "beta"]].round(2).T.to_string(header=False))

    # размеры типов по месяцам
    sizes = pd.DataFrame({m: pd.Series(L[:, i]).value_counts().sort_index() for i, m in enumerate(months)}).T
    print("\nразмеры типов по месяцам (первые/последние):")
    print(pd.concat([sizes.head(3), sizes.tail(3)]).to_string())

    # смены типа
    changes = (L[:, 1:] != L[:, :-1])
    per_month = changes.mean(axis=0)
    per_mo = changes.sum(axis=1)
    print("\nдоля МО, сменивших тип, по месяцам:")
    print(pd.Series(per_month, index=months[1:]).round(3).to_string())
    print(f"\nМО без единой смены типа: {(per_mo == 0).mean():.1%}; 1 смена: {(per_mo == 1).mean():.1%}; "
          f"2: {(per_mo == 2).mean():.1%}; ≥3: {(per_mo >= 3).mean():.1%}")
    names = attrs["name"].reindex(lab.index).astype(str)
    top = np.argsort(-per_mo)[:12]
    print("\nсамые неустойчивые МО (число смен, последовательность типов):")
    for i in top:
        seq = "".join(str(x) for x in L[i])
        print(f"  {per_mo[i]:2d}  {names.iloc[i][:45]:45s} {seq}")

    # матрица переходов между соседними месяцами
    T = np.zeros((K, K))
    for t in range(1, L.shape[1]):
        np.add.at(T, (L[:, t - 1], L[:, t]), 1)
    Tn = T / T.sum(axis=1, keepdims=True)
    print("\nматрица переходов (строка → столбец, доли), диагональ = устойчивость типа:")
    print(pd.DataFrame(Tn, index=[f"из {i}" for i in range(K)], columns=[f"в {j}" for j in range(K)]).round(3).to_string())
    pd.DataFrame(Tn).to_csv(out / f"transitions_{tag}.csv")

    # модальный тип и профили
    modal = pd.Series([np.bincount(row, minlength=K).argmax() for row in L], index=lab.index, name="type")
    modal.to_csv(out / f"modal_type_{tag}.csv")
    feat = features.monthly_features(panel)
    dates = sorted(panel.index.get_level_values("date").unique())
    prof = features.static_profile(feat, dates=[d for d in dates if d.year == 2024]).reindex(lab.index)
    raw = pd.DataFrame(index=lab.index)
    for c in features.SHARE_COLS:
        raw[f"share_{c}"] = prof[f"share_{c}"]
    raw["total_rub"] = panel["total"].groupby(level="territory_id").mean().reindex(lab.index)
    raw["population"] = attrs["population"].reindex(lab.index)
    raw["wage"] = attrs["wage"].reindex(lab.index)
    raw["market_access"] = attrs["market_access"].reindex(lab.index)
    raw["urban_share"] = attrs["urban_share"].reindex(lab.index)
    raw["yoy_growth"] = np.exp(prof["yoy_log_mean"]) - 1
    raw["seasonal_amplitude"] = prof["seasonal_amplitude"]
    for g in ["emp_agri", "emp_extract", "emp_industry", "emp_market_services", "emp_public"]:
        raw[g] = features.static_attributes(attrs.reindex(lab.index))[g]
    mr = clustering.mirkin_rule(raw, modal.to_numpy())
    mr.to_csv(out / f"mirkin_{tag}.csv")
    print("\nправило Миркина по модальному типу, d_kv = (c_kv − g_v)/g_v:")
    print(mr.round(2).to_string())
    pop = attrs["population"].reindex(lab.index).fillna(0)
    for c in range(K):
        idx = np.where(modal.to_numpy() == c)[0]
        top5 = idx[np.argsort(-pop.iloc[idx].to_numpy())[:6]]
        regions = attrs["region_name"].reindex(lab.index).iloc[idx].value_counts().head(4)
        print(f"\nтип {c} (n={len(idx)}): " + "; ".join(names.iloc[top5].str.replace("городской округ ", "г.о. ").str[:28].tolist()))
        print("   регионы: " + ", ".join(f"{r} ({n})" for r, n in regions.items()))


if __name__ == "__main__":
    main()
