"""Зоопарк методов на статическом срезе 2024 (критерий «сравнение методов»).

Только атрибуты: K-means, Ward, GMM. Только сеть: Leiden (модульность), Infomap, спектральная (по графу).
Совместные: KEFRiN (евклид/косинус), CANUS (Shalileh 2025, код автора), EVA и iLouvain (CDlib).
Единый протокол: один граф (обученный LogModel или kNN), K из {4, 6} где задаётся, полный ICVI, bootstrap-ARI (B),
время. Запуск: .venv312/bin/python scripts/method_zoo.py [--graph learned|knn] [--B 10]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))
warnings.filterwarnings("ignore")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, SpectralClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from mo_types import clustering, data, features, graphs, metrics  # noqa: E402

OUT = data.ROOT / "outputs" / "method_zoo"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run_method(name, Y, A, P, k, seed, min_size):
    """Возвращает метки (или None, если метод недоступен)."""
    N = Y.shape[0]
    if name == "kmeans":
        return clustering.kmeans(Y, k, seed, 10)
    if name == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Y)
    if name == "gmm":
        return GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=seed).fit_predict(Y)
    if name == "leiden_mod":
        return graphs.leiden(A, resolution=1.0, method="modularity", seed=seed)
    if name == "leiden_mod_r0.5":
        return graphs.leiden(A, resolution=0.5, method="modularity", seed=seed)
    if name == "infomap":
        from cdlib import algorithms
        G = nx.from_scipy_sparse_array(A)
        com = algorithms.infomap(G)
        lab = np.zeros(N, dtype=int)
        for i, c in enumerate(com.communities):
            lab[list(c)] = i
        return lab
    if name == "spectral_graph":
        return SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed, assign_labels="kmeans").fit_predict(A.toarray() + 1e-9)
    if name == "kefrin_e":
        return clustering.kefrin(Y, P, k, xi="auto", distance="euclidean", seed=seed, n_init=5, min_size=min_size)[0]
    if name == "kefrin_c":
        return clustering.kefrin(Y, P, k, xi="auto", distance="cosine", seed=seed, n_init=5, min_size=min_size)[0]
    if name == "canus":
        try:
            import torch  # noqa: F401
            from canus import CANUSClusterer
        except Exception as ex:
            log(f"  CANUS недоступен: {ex}")
            return None
        model = CANUSClusterer(n_clusters=k, epochs=150, seed=seed)
        model.fit(Y.astype(np.float32), A.toarray().astype(np.float32))
        return np.asarray(model.y_pred)
    if name == "dmon":
        try:
            from mo_types.dmon import dmon_cluster
            return dmon_cluster(Y, A, k, seed=seed)
        except Exception as ex:
            log(f"  DMoN недоступен: {ex}")
            return None
    if name in ("eva", "ilouvain"):
        from cdlib import algorithms
        G = nx.from_scipy_sparse_array(A)
        if name == "ilouvain":
            labels = {i: {f"f{j}": float(v) for j, v in enumerate(Y[i])} for i in range(N)}
            com = algorithms.ilouvain(G, labels)
        else:  # EVA — категориальные атрибуты: квартильные корзины основных признаков
            q = np.quantile(Y, [0.25, 0.5, 0.75], axis=0)
            binned = (Y[:, None, :] > q[None, :, :]).sum(1)
            labels = {i: {f"f{j}": int(binned[i, j]) for j in range(min(Y.shape[1], 8))} for i in range(N)}
            com = algorithms.eva(G, labels, alpha=0.5)
        lab = np.zeros(N, dtype=int)
        for i, c in enumerate(com.communities):
            lab[list(c)] = i
        return lab
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="learned")
    ap.add_argument("--B", type=int, default=10)
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
    beh_cols = [c for c in prof.columns if c.startswith("clr_")] + ["rel_level", "level_volatility", "seasonal_amplitude", "yoy_log_mean", "marketplace_trend"]
    X_beh = features.zscore(prof[beh_cols]).to_numpy()
    Y = np.hstack([X_beh, features.zscore(stat[ctx_cols]).to_numpy()])
    if args.graph == "learned":
        A = sparse.load_npz(data.ROOT / "outputs" / "learned_graph" / "A_static_learned.npz")
    else:
        A = graphs.knn_graph(X_beh, k=cfg["graph"]["k"], metric="cosine", mutual=True)
    P = graphs.diffusion_profile(A, steps=2)
    N = Y.shape[0]
    min_size = max(2, int(0.01 * N))
    plan = [("kmeans", [4, 6]), ("ward", [4, 6]), ("gmm", [4, 6]), ("spectral_graph", [4, 6]),
            ("leiden_mod", [None]), ("leiden_mod_r0.5", [None]), ("infomap", [None]),
            ("kefrin_e", [4, 6]), ("kefrin_c", [4, 6]), ("canus", [4, 6]), ("dmon", [4, 6]), ("ilouvain", [None]), ("eva", [None])]
    rows = []
    for name, ks in plan:
        for k in ks:
            t0 = time.time()
            try:
                lab = run_method(name, Y, A, P, k, seed, min_size)
            except Exception as ex:
                log(f"  {name} K={k}: ошибка {str(ex)[:120]}")
                continue
            if lab is None:
                continue
            m = metrics.all_metrics(Y, A, lab)
            aris = []
            if args.B > 0 and name not in ("ilouvain", "eva", "canus", "infomap", "dmon"):
                for b in range(args.B):
                    idx = np.sort(rng.choice(N, size=int(0.8 * N), replace=False))
                    sub_A = A[idx][:, idx]
                    try:
                        lab_b = run_method(name, Y[idx], sub_A, graphs.diffusion_profile(sub_A, 2), k, seed + b, max(2, int(0.01 * len(idx))))
                        aris.append(adjusted_rand_score(lab[idx], lab_b))
                    except Exception:
                        pass
            row = {"method": name, "K_set": k, **m, "boot_ari": float(np.mean(aris)) if aris else np.nan,
                   "min_size": int(np.bincount(lab).min()), "sec": round(time.time() - t0, 1)}
            rows.append(row)
            log(f"  {name} K={k}: сообществ {m['K']}, SW {m['SW']:.3f}, Q {m['modularity']:.3f}, AVI {m['AVI']:.3f}, ARI {row['boot_ari']:.3f} ({row['sec']} с)")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"zoo_{args.graph}.csv", index=False)
    cols = ["method", "K_set", "K", "SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "modularity", "boot_ari", "min_size", "sec"]
    log("зоопарк методов:\n" + tab[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
