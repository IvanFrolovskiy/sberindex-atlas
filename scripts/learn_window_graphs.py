"""Обученные графы по месяцам: LogModel (Kalofolias 2016/2019) на каждом месячном окне сигналов.

Запуск: .venv312/bin/python scripts/learn_window_graphs.py [--deg 15] [--maxit 1000] [--out outputs/learned_graph/windowlog]
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
from graph_learn.temporal.smooth import WindowLogModel  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from mo_types import data, features, graphs, learned_graph as lg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, default=15)
    ap.add_argument("--maxit", type=int, default=1000)
    ap.add_argument("--out", default="outputs/learned_graph/windowlog")
    args = ap.parse_args()
    out = data.ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = data.load_config()
    panel, attrs = data.load_processed(cfg)
    dates = sorted(panel.index.get_level_values("date").unique())
    tids = panel.index.get_level_values("territory_id").unique()
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    month_cols = [c for c in feat.columns if c.startswith("clr_")] + ["rel_level"]
    Xs = [features.zscore(feat.xs(d, level="date").reindex(tids)[month_cols]).to_numpy() for d in dates]
    V = Xs[0].shape[1]
    x = np.vstack([X.T for X in Xs])
    t0 = time.time()
    model = WindowLogModel(window_size=V, avg_degree=args.deg, maxit=args.maxit, tol=1e-5).fit(x)
    print(f"WindowLogModel: {time.time() - t0:.0f} с, окон {model.weights_.shape[0]}", flush=True)
    A_knn = [graphs.knn_graph(X, k=args.deg, metric="cosine", mutual=True) for X in Xs]
    rows = []
    As = []
    for t in range(len(dates)):
        W = squareform(np.asarray(model.weights_[t]))
        W = np.where(W < 1e-3, 0.0, W)
        A = sparse.csr_matrix(W)
        As.append(A)
        sparse.save_npz(out / f"A_t{t:02d}.npz", A)
        rows.append({"t": t, "date": dates[t].strftime("%Y-%m"), **lg.graph_stats(A),
                     "jaccard_prev": lg.edge_jaccard(As[t - 1], A) if t > 0 else np.nan,
                     "jaccard_knn_prev": lg.edge_jaccard(A_knn[t - 1], A_knn[t]) if t > 0 else np.nan,
                     "jaccard_vs_knn": lg.edge_jaccard(A, A_knn[t])})
    st = pd.DataFrame(rows)
    st.to_csv(out / "stats.csv", index=False)
    print(st.round(3).to_string(index=False))
    print(f"итого: степень {st.deg_mean.mean():.1f}, изолятов в среднем {st.isolates.mean():.1f}, "
          f"Жаккар соседних месяцев: learned {st.jaccard_prev.median():.3f} vs kNN {st.jaccard_knn_prev.median():.3f}")


if __name__ == "__main__":
    main()
