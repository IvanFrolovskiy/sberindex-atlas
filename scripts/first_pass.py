"""Первый прогон: признаки → сеть сходства → K-means / KEFRiN / Leiden → ICVI → временной взгляд → правило Миркина.

Запуск: .venv312/bin/python scripts/first_pass.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as ami  # noqa: E402

# numpy+Accelerate на Apple Silicon даёт ложные предупреждения о переполнении в matmul при конечных входах
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

from mo_types import clustering, data, features, graphs, metrics  # noqa: E402

OUT = data.ROOT / "outputs" / "first_pass"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    cfg = data.load_config()
    panel, attrs = data.load_processed(cfg)
    dates = sorted(panel.index.get_level_values("date").unique())
    tids = panel.index.get_level_values("territory_id").unique()
    log(f"панель {len(tids)} МО × {len(dates)} мес.")

    # --- признаки -----------------------------------------------------------------
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    prof = features.static_profile(feat, dates=[d for d in dates if d.year == 2024])
    stat = features.static_attributes(attrs.reindex(prof.index))
    beh_cols = [c for c in prof.columns if c.startswith("clr_")] + [
        "rel_level", "level_volatility", "seasonal_amplitude", "yoy_log_mean", "marketplace_trend"]
    X_beh = features.zscore(prof[beh_cols]).to_numpy()
    ctx_cols = [c for c in stat.columns if stat[c].notna().mean() > 0.9]
    X_ctx = features.zscore(stat[ctx_cols]).to_numpy()
    Y = np.hstack([X_beh, X_ctx])  # атрибуты узлов: поведение + контекст
    log(f"признаки: поведение {X_beh.shape[1]}, контекст {X_ctx.shape[1]} ({ctx_cols})")

    # --- сеть: сходство структуры трат (как у жюри) ---------------------------------
    A = graphs.knn_graph(X_beh, k=cfg["graph"]["k"], metric=cfg["graph"]["metric"], mutual=cfg["graph"]["mutual"])
    deg = np.asarray((A > 0).sum(1)).ravel()
    log(f"сеть: {A.nnz // 2} рёбер, средняя степень {deg.mean():.1f}, изолятов {(deg == 0).sum()}")
    P = graphs.diffusion_profile(A, steps=2)

    # --- статические методы ---------------------------------------------------------
    rows, labels_store = [], {}
    seed, n_init = cfg["clustering"]["seed"], cfg["clustering"]["n_init"]
    for k in cfg["clustering"]["k_range"]:
        lab = clustering.kmeans(Y, k, seed, n_init)
        rows.append({"method": "kmeans", "param": k, **metrics.all_metrics(Y, A, lab)}); labels_store[("kmeans", k)] = lab
        lab, C, L, F, xi_e = clustering.kefrin(Y, P, k, rho=1.0, xi="auto", distance="euclidean", seed=seed, n_init=3)
        rows.append({"method": "kefrin_e", "param": k, **metrics.all_metrics(Y, A, lab)}); labels_store[("kefrin_e", k)] = lab
        lab, C, L, F, xi_c = clustering.kefrin(Y, P, k, rho=1.0, xi="auto", distance="cosine", seed=seed, n_init=3)
        rows.append({"method": "kefrin_c", "param": k, **metrics.all_metrics(Y, A, lab)}); labels_store[("kefrin_c", k)] = lab
        log(f"K={k} готово (xi_auto: euclid {xi_e:.1f}, cosine {xi_c:.2f})")
    for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
        lab = graphs.leiden(A, resolution=res, method="modularity", seed=seed)
        rows.append({"method": "leiden_mod", "param": res, **metrics.all_metrics(Y, A, lab)}); labels_store[("leiden_mod", res)] = lab
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "icvi_static.csv", index=False)
    cols = ["method", "param", "K", "SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "modularity"]
    log("ICVI статический срез (2024):\n" + table[cols].round(3).to_string(index=False))

    # --- временной взгляд -------------------------------------------------------------
    month_cols = [c for c in feat.columns if c.startswith("clr_")] + ["rel_level"]
    As, Xs = [], []
    for d in dates:
        Xm = features.zscore(feat.xs(d, level="date").reindex(prof.index)[month_cols]).to_numpy()
        Xs.append(Xm)
        As.append(graphs.knn_graph(Xm, k=cfg["graph"]["k"], metric="cosine", mutual=True))
    t0 = time.time()
    memb = graphs.temporal_leiden(As, interslice_weight=0.5, resolution=1.0, seed=seed)
    log(f"temporal Leiden: {time.time() - t0:.1f} с; сообществ по месяцам: "
        f"{[len(np.unique(m)) for m in memb]}")
    amis = [ami(memb[t - 1], memb[t]) for t in range(1, len(dates))]
    csr = [float((memb[t - 1] != memb[t]).mean()) for t in range(1, len(dates))]
    log(f"AMI соседних месяцев: min {min(amis):.3f} median {np.median(amis):.3f}; "
        f"доля МО, сменивших сообщество: median {np.median(csr):.3f}, max {max(csr):.3f}")
    # независимый K-means по месяцам с тёплым стартом
    k8 = 8
    prev = None
    ind_labels = []
    from sklearn.cluster import KMeans
    for Xm in Xs:
        km = KMeans(n_clusters=k8, n_init=1 if prev is not None else 10, init=prev if prev is not None else "k-means++",
                    random_state=seed).fit(Xm)
        prev = km.cluster_centers_
        ind_labels.append(km.labels_)
    amis_km = [ami(ind_labels[t - 1], ind_labels[t]) for t in range(1, len(dates))]
    csr_km = [float((ind_labels[t - 1] != ind_labels[t]).mean()) for t in range(1, len(dates))]
    log(f"K-means по месяцам (K=8, warm start): AMI median {np.median(amis_km):.3f}, "
        f"смена кластера median {np.median(csr_km):.3f}")
    pd.DataFrame(memb.T, index=prof.index, columns=[d.strftime("%Y-%m") for d in dates]).to_csv(OUT / "temporal_leiden_labels.csv")

    # --- интерпретация лучшего статического решения --------------------------------
    best = table[table.method.isin(["kmeans", "kefrin_e"])].sort_values("SW", ascending=False).iloc[0]
    key = (best["method"], best["param"])
    lab = labels_store[key]
    log(f"лучший по SW среди kmeans/kefrin: {key}, K={best['K']}")
    raw = pd.DataFrame(index=prof.index)
    for c in features.SHARE_COLS:
        raw[f"share_{c}"] = prof[f"share_{c}"]
    raw["total_rub"] = np.exp(panel["total"].groupby(level="territory_id").mean().reindex(prof.index).apply(np.log))
    raw["population"] = attrs["population"].reindex(prof.index)
    raw["wage"] = attrs["wage"].reindex(prof.index)
    raw["market_access"] = attrs["market_access"].reindex(prof.index)
    raw["yoy_growth"] = np.exp(prof["yoy_log_mean"]) - 1
    raw["seasonal_amplitude"] = prof["seasonal_amplitude"]
    mr = clustering.mirkin_rule(raw, lab)
    mr.to_csv(OUT / "mirkin_rule.csv")
    log("правило Миркина, d_kv = (c_kv − g_v)/g_v (в долях от среднего):\n" + mr.round(2).to_string())
    names = attrs["name"].reindex(prof.index)
    pop = attrs["population"].reindex(prof.index).fillna(0)
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        top = idx[np.argsort(-pop.iloc[idx].to_numpy())[:5]]
        print(f"  кластер {c} (n={len(idx)}): " + "; ".join(names.iloc[top].astype(str).tolist()))
    pd.DataFrame({"territory_id": prof.index, "cluster": lab, "name": names.values}).to_csv(OUT / "labels_static_best.csv", index=False)
    log("готово, результаты в outputs/first_pass")


if __name__ == "__main__":
    main()
