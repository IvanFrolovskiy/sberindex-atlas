"""Обученный граф против kNN: статика (2024) и 24 месячных среза (TGFA).

Запуск: .venv312/bin/python scripts/learned_graph.py [--static-only] [--max-iter 300] [--l1 0.0] [--l2 0.0] [--deg 15]
Результаты: outputs/learned_graph/ (A_static.npz, A_tXX.npz, stats.csv)
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from mo_types import clustering, data, features, graphs, learned_graph as lg, metrics  # noqa: E402

OUT = data.ROOT / "outputs" / "learned_graph"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--deg", type=int, default=15)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = data.load_config()
    seed = cfg["clustering"]["seed"]
    panel, attrs = data.load_processed(cfg)
    dates = sorted(panel.index.get_level_values("date").unique())
    tids = panel.index.get_level_values("territory_id").unique()
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    prof = features.static_profile(feat, dates=[d for d in dates if d.year == 2024])
    stat = features.static_attributes(attrs.reindex(tids))
    ctx_cols = [c for c in stat.columns if stat[c].notna().mean() > 0.9]
    X_ctx = features.zscore(stat[ctx_cols]).to_numpy()
    beh_cols = [c for c in prof.columns if c.startswith("clr_")] + [
        "rel_level", "level_volatility", "seasonal_amplitude", "yoy_log_mean", "marketplace_trend"]
    X_beh = features.zscore(prof[beh_cols]).to_numpy()
    Y = np.hstack([X_beh, X_ctx])

    # ---------------- статический граф -------------------------------------------------
    rows = []
    t0 = time.time()
    A_knn = graphs.knn_graph(X_beh, k=args.deg, metric="cosine", mutual=True)
    log(f"kNN(k={args.deg}, mutual): {lg.graph_stats(A_knn)} за {time.time() - t0:.1f} с")
    t0 = time.time()
    A_learn, theta = lg.learn_static(X_beh, avg_degree=args.deg, maxit=1000, tol=1e-5)
    log(f"LogModel(avg_degree={args.deg}): {lg.graph_stats(A_learn)}, θ={theta:.3g}, за {time.time() - t0:.1f} с")
    log(f"Жаккар рёбер kNN vs learned: {lg.edge_jaccard(A_knn, A_learn):.3f}")
    sparse.save_npz(OUT / "A_static_learned.npz", A_learn)
    sparse.save_npz(OUT / "A_static_knn.npz", A_knn)
    for gname, A in [("knn", A_knn), ("learned", A_learn)]:
        P = graphs.diffusion_profile(A, steps=2)
        for res in [0.3, 0.5, 1.0]:
            lab = graphs.leiden(A, resolution=res, method="modularity", seed=seed)
            rows.append({"graph": gname, "method": f"leiden_r{res}", **metrics.all_metrics(Y, A, lab)})
        for k in [5, 6, 8]:
            lab, C, L, F, xi = clustering.kefrin(Y, P, k, xi="auto", distance="euclidean", seed=seed, n_init=3)
            rows.append({"graph": gname, "method": f"kefrin_e_K{k}", **metrics.all_metrics(Y, A, lab)})
            lab, C, L, F, xi = clustering.kefrin(Y, P, k, xi="auto", distance="cosine", seed=seed, n_init=3)
            rows.append({"graph": gname, "method": f"kefrin_c_K{k}", **metrics.all_metrics(Y, A, lab)})
        lab = clustering.kmeans(Y, args.k, seed)
        rows.append({"graph": gname, "method": f"kmeans_K{args.k}", **metrics.all_metrics(Y, A, lab)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "static_comparison.csv", index=False)
    cols = ["graph", "method", "K", "SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "modularity", "conductance"]
    log("статика, kNN vs learned:\n" + tab[cols].round(3).to_string(index=False))
    if args.static_only:
        return

    # ---------------- временные графы ---------------------------------------------------
    month_cols = [c for c in feat.columns if c.startswith("clr_")] + ["rel_level"]
    Xs = [features.zscore(feat.xs(d, level="date").reindex(tids)[month_cols]).to_numpy() for d in dates]
    A_knn_t = [graphs.knn_graph(X, k=args.deg, metric="cosine", mutual=True) for X in Xs]
    jac_knn = [lg.edge_jaccard(A_knn_t[t - 1], A_knn_t[t]) for t in range(1, len(dates))]
    log(f"kNN по месяцам: Жаккар соседних срезов медиана {np.median(jac_knn):.3f}, min {min(jac_knn):.3f}")
    t0 = time.time()
    As, model = lg.learn_temporal(Xs, avg_degree=args.deg, l1_time=args.l1, l2_time=args.l2, max_iter=args.max_iter, tol=1e-3)
    log(f"TGFA(l1={args.l1}, l2={args.l2}, iters={args.max_iter}, converged_at={model.converged_}): {time.time() - t0:.1f} с")
    jac = [lg.edge_jaccard(As[t - 1], As[t]) for t in range(1, len(dates))]
    st = pd.DataFrame([{"t": t, "date": dates[t].strftime("%Y-%m"), **lg.graph_stats(As[t]),
                        "jaccard_prev": (jac[t - 1] if t > 0 else np.nan),
                        "jaccard_vs_knn": lg.edge_jaccard(As[t], A_knn_t[t])} for t in range(len(dates))])
    st.to_csv(OUT / "temporal_stats.csv", index=False)
    log(f"TGFA по месяцам: степень {st.deg_mean.mean():.1f}, изолятов {st.isolates.sum()}, "
        f"Жаккар соседних срезов медиана {np.nanmedian(st.jaccard_prev):.3f}, Жаккар с kNN {st.jaccard_vs_knn.mean():.3f}")
    sub = OUT / (f"tgfa_l1{args.l1}_l2{args.l2}" + (f"_{args.tag}" if args.tag else ""))
    sub.mkdir(exist_ok=True)
    st.to_csv(sub / "temporal_stats.csv", index=False)
    for t, A in enumerate(As):
        sparse.save_npz(sub / f"A_t{t:02d}.npz", A)
    log(f"временные графы сохранены в {sub}")


if __name__ == "__main__":
    main()
