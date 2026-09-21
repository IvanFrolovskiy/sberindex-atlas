"""Выбор K: bootstrap-устойчивость (Hennig 2007) + ICVI на статическом срезе 2024 для KEFRiN и K-means.

Для каждого K: B подвыборок по 80% МО → кластеризация → ARI с полным решением на пересечении,
покластерный Жаккар (стабильность каждого типа). Запуск: .venv312/bin/python scripts/stability_k.py [--graph knn|learned]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from mo_types import clustering, data, features, graphs, metrics  # noqa: E402

OUT = data.ROOT / "outputs" / "stability"
OUT.mkdir(parents=True, exist_ok=True)


def cluster_jaccard(full: np.ndarray, boot: np.ndarray, k: int) -> np.ndarray:
    """Для каждого кластера полного решения — Жаккар с наиболее похожим кластером бутстрап-решения."""
    out = np.zeros(k)
    for c in range(k):
        a = full == c
        best = 0.0
        for d in np.unique(boot):
            b = boot == d
            j = (a & b).sum() / max((a | b).sum(), 1)
            best = max(best, j)
        out[c] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="knn")
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--k", default="4,5,6,7,8,9")
    ap.add_argument("--no-context", action="store_true", help="только признаки потребления (поведенческая линза)")
    args = ap.parse_args()
    cfg = data.load_config()
    seed = cfg["clustering"]["seed"]
    rng = np.random.default_rng(seed)
    panel, attrs = data.load_processed(cfg)
    dates = sorted(panel.index.get_level_values("date").unique())
    tids = panel.index.get_level_values("territory_id").unique()
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    prof = features.static_profile(feat, dates=[d for d in dates if d.year == 2024])
    stat = features.static_attributes(attrs.reindex(tids))
    ctx_cols = [c for c in stat.columns if stat[c].notna().mean() > 0.9]
    beh_cols = [c for c in prof.columns if c.startswith("clr_")] + [
        "rel_level", "level_volatility", "seasonal_amplitude", "yoy_log_mean", "marketplace_trend"]
    X_beh = features.zscore(prof[beh_cols]).to_numpy()
    Y = X_beh if args.no_context else np.hstack([X_beh, features.zscore(stat[ctx_cols]).to_numpy()])
    suffix = args.graph + ("_beh" if args.no_context else "")
    if args.graph == "learned":
        A = sparse.load_npz(data.ROOT / "outputs" / "learned_graph" / "A_static_learned.npz")
    else:
        A = graphs.knn_graph(X_beh, k=cfg["graph"]["k"], metric="cosine", mutual=True)
    P = graphs.diffusion_profile(A, steps=2)
    N = Y.shape[0]
    min_size = max(2, int(0.01 * N))
    rows, per_cluster = [], []
    for k in [int(x) for x in args.k.split(",")]:
        t0 = time.time()
        full_kef, C, L, F, xi = clustering.kefrin(Y, P, k, xi="auto", seed=seed, n_init=10, min_size=min_size)
        full_km = clustering.kmeans(Y, k, seed, 10)
        aris = {"kefrin": [], "kmeans": []}
        jac = {"kefrin": np.zeros(k), "kmeans": np.zeros(k)}
        for b in range(args.B):
            idx = np.sort(rng.choice(N, size=int(0.8 * N), replace=False))
            sub_A = A[idx][:, idx]
            sub_P = graphs.diffusion_profile(sub_A, steps=2)
            lab_b, *_ = clustering.kefrin(Y[idx], sub_P, k, xi=xi, seed=seed + b, n_init=3, min_size=max(2, int(0.01 * len(idx))))
            aris["kefrin"].append(adjusted_rand_score(full_kef[idx], lab_b))
            jac["kefrin"] += cluster_jaccard(full_kef[idx], lab_b, k) / args.B
            lab_k = clustering.kmeans(Y[idx], k, seed + b, 3)
            aris["kmeans"].append(adjusted_rand_score(full_km[idx], lab_k))
            jac["kmeans"] += cluster_jaccard(full_km[idx], lab_k, k) / args.B
        for m, full in [("kefrin", full_kef), ("kmeans", full_km)]:
            icv = metrics.all_metrics(Y, A, full)
            rows.append({"method": m, "K": k, "ari_mean": np.mean(aris[m]), "ari_min": np.min(aris[m]),
                         "jaccard_min": jac[m].min(), "jaccard_mean": jac[m].mean(),
                         "min_size": int(np.bincount(full).min()), **{c: icv[c] for c in ["SW", "CH", "S_Dbw", "AVI", "ANUI", "MQ", "modularity"]}})
            per_cluster.append(pd.DataFrame({"method": m, "K": k, "cluster": range(k), "jaccard": jac[m],
                                             "size": np.bincount(full, minlength=k)}))
        print(time.strftime("%H:%M:%S"), f"K={k}: kefrin ARI {np.mean(aris['kefrin']):.3f} (jac min {jac['kefrin'].min():.2f}), "
              f"kmeans ARI {np.mean(aris['kmeans']):.3f} (jac min {jac['kmeans'].min():.2f}) — {time.time() - t0:.0f} с", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"stability_{suffix}.csv", index=False)
    pd.concat(per_cluster).to_csv(OUT / f"cluster_jaccard_{suffix}.csv", index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
