"""Сравнение правил ребра × способов разрежения (критерий «построение сетевой структуры»).

Для каждой пары (правило, разрежение): свойства графа, гомофилия по региону и по опорным типам,
модульность Leiden, качество совместной кластеризации KEFRiN (K=4, 6) в пространстве признаков и на графе,
bootstrap-устойчивость KEFRiN K=6 (ARI), время. Плюс согласие правил между собой (Жаккар kNN-рёбер).

Запуск: .venv312/bin/python scripts/edge_rules.py [--B 10] [--fast]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from mo_types import clustering, data, edge_rules as er, features, graphs, metrics  # noqa: E402

OUT = data.ROOT / "outputs" / "edge_rules"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_inputs(cfg):
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
    Y = np.hstack([X_beh, features.zscore(stat[ctx_cols]).to_numpy()])
    # ряды: приросты log(total) за вычетом медианы месяца (идиосинкратическое движение), 23 точки
    lt = feat["log_total"].unstack("date").reindex(tids)
    growth = lt.diff(axis=1).iloc[:, 1:]
    growth = growth - growth.median(axis=0)
    rel = feat["rel_level"].unstack("date").reindex(tids)
    clr_cols = [c for c in feat.columns if c.startswith("clr_")]
    clr3 = np.stack([feat[c].unstack("date").reindex(tids).to_numpy() for c in clr_cols], axis=2)  # N×T×6
    clr3 = (clr3 - clr3.mean(axis=(0, 1), keepdims=True)) / (clr3.std(axis=(0, 1), keepdims=True) + 1e-12)
    conn = data.load_connection(cfg, "highway")
    D_road = er.road_matrix(conn, tids.to_numpy())
    return dict(panel=panel, attrs=attrs, tids=tids, Y=Y, X_beh=X_beh, growth=growth.to_numpy(),
                rel=rel.to_numpy(), clr3=clr3, D_road=D_road)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--fast", action="store_true", help="без DTW по композиции и без bootstrap")
    args = ap.parse_args()
    cfg = data.load_config()
    seed = cfg["clustering"]["seed"]
    rng = np.random.default_rng(seed)
    inp = build_inputs(cfg)
    Y, tids, attrs = inp["Y"], inp["tids"], inp["attrs"]
    N = Y.shape[0]
    log(f"N={N}, признаков {Y.shape[1]}")

    # опорные метки для гомофилии: регион и атрибутные типы (K-means K=4 на Y, независимо от графа)
    region = pd.factorize(attrs["region_code"].reindex(tids).fillna(-1))[0]
    ref_types = clustering.kmeans(Y, 4, seed, 10)
    logpop = np.log(attrs["population"].reindex(tids).fillna(attrs["population"].median()).to_numpy())

    # ---------------- сходства ---------------------------------------------------------
    sims = {}
    timings = {}
    t0 = time.time(); sims["cos_profile"] = er.cosine(inp["X_beh"]); timings["cos_profile"] = time.time() - t0
    t0 = time.time(); sims["corr_growth"] = er.corr(inp["growth"]); timings["corr_growth"] = time.time() - t0
    t0 = time.time(); sims["corr_growth_lag2"] = er.corr_lag(inp["growth"], 2); timings["corr_growth_lag2"] = time.time() - t0
    t0 = time.time(); sims["dtw_level"], D_dtw = er.dtw_sim(inp["rel"], window=3); timings["dtw_level"] = time.time() - t0
    log(f"DTW уровня: {timings['dtw_level']:.0f} с")
    if not args.fast:
        t0 = time.time(); sims["dtw_composition"], D_dtw3 = er.dtw_ndim_sim(inp["clr3"], window=3); timings["dtw_composition"] = time.time() - t0
        log(f"DTW композиции (6-мерный): {timings['dtw_composition']:.0f} с")
    t0 = time.time(); sims["road_gravity"] = er.road_sim(inp["D_road"], lam_km=200.0); timings["road_gravity"] = time.time() - t0
    A_learn = sparse.load_npz(data.ROOT / "outputs" / "learned_graph" / "A_static_learned.npz")
    # SNF: композиция профиля + со-движение + география
    t0 = time.time()
    aff = [er.snf_affinity(er.sim_to_dist(sims["cos_profile"])), er.snf_affinity(er.sim_to_dist(sims["corr_growth"])),
           er.snf_affinity(inp["D_road"] / 1000.0)]
    sims["snf_cos_corr_road"] = er.snf_fuse(*aff, K=20, t=20)
    timings["snf_cos_corr_road"] = time.time() - t0
    log(f"SNF: {timings['snf_cos_corr_road']:.0f} с")

    # согласие правил: Жаккар взаимных kNN-рёбер
    knn_sets = {name: er.knn_sparsify(S, k=15, mutual=True) for name, S in sims.items()}
    knn_sets["learned_logmodel"] = A_learn
    names = list(knn_sets)
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            agree.loc[a, b] = er.edge_jaccard(knn_sets[a], knn_sets[b])
    agree.to_csv(OUT / "rule_agreement_jaccard.csv")
    log("Жаккар рёбер между правилами (kNN15 mutual):\n" + agree.round(2).to_string())

    # ---------------- правило × разрежение ------------------------------------------------
    n_edges_ref = int(knn_sets["cos_profile"].nnz // 2)
    rows = []
    graphs_store = {}
    for name, S in sims.items():
        for sp in ["knn15_mutual", "knn15", "epsilon", "tmfg", "disparity"]:
            t0 = time.time()
            try:
                if sp == "knn15_mutual":
                    A = er.knn_sparsify(S, 15, mutual=True)
                elif sp == "knn15":
                    A = er.knn_sparsify(S, 15, mutual=False)
                elif sp == "epsilon":
                    A = er.epsilon_sparsify(S, n_edges_ref)
                elif sp == "tmfg":
                    A = er.tmfg_sparsify(S)
                else:
                    A = er.disparity_sparsify(S, alpha=0.05, k_pre=50)
            except Exception as ex:
                log(f"  {name}/{sp}: ошибка {ex}")
                continue
            graphs_store[(name, sp)] = A
            rows.append(evaluate(name, sp, A, Y, region, ref_types, logpop, seed, rng, args, t0, timings.get(name, 0.0)))
            log(f"  {name}/{sp}: рёбер {rows[-1]['edges']}, Q {rows[-1]['leiden_Q']:.2f}, "
                f"KEFRiN6 SW {rows[-1]['k6_SW']:.3f} ARI {rows[-1].get('k6_boot_ari', float('nan')):.3f}")
    # обученный граф как есть
    t0 = time.time()
    rows.append(evaluate("learned_logmodel", "as_is", A_learn, Y, region, ref_types, logpop, seed, rng, args, t0, 16.0))
    graphs_store[("learned_logmodel", "as_is")] = A_learn
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "edge_rules_comparison.csv", index=False)
    cols = ["rule", "sparsifier", "edges", "deg_mean", "components", "isolates", "clustering", "same_region_share",
            "assort_types", "assort_logpop", "leiden_Q", "leiden_K", "k6_SW", "k6_CH", "k6_AVI", "k6_MQ", "k6_Q", "k6_boot_ari", "sec"]
    log("сводка:\n" + tab[[c for c in cols if c in tab]].round(3).to_string(index=False))
    for (name, sp), A in graphs_store.items():
        sparse.save_npz(OUT / f"A_{name}_{sp}.npz", A)


def evaluate(name, sp, A, Y, region, ref_types, logpop, seed, rng, args, t0, sim_sec):
    P = graphs.diffusion_profile(A, steps=2)
    g = er.graph_summary(A, labels=ref_types, numeric=logpop)
    same_region = er.graph_summary(A, labels=region)["same_label_share"]
    lab_l = graphs.leiden(A, resolution=1.0, method="modularity", seed=seed)
    gm = metrics.graph_metrics(A, lab_l)
    row = {"rule": name, "sparsifier": sp, "edges": g["edges"], "deg_mean": g["deg_mean"], "deg_max": g["deg_max"],
           "components": g["components"], "isolates": g["isolates"], "clustering": g["clustering"],
           "same_region_share": same_region, "assort_types": g["assort_labels"], "assort_logpop": g["assort_numeric"],
           "leiden_Q": gm["modularity"], "leiden_K": gm["K"], "sim_sec": sim_sec}
    N = Y.shape[0]
    min_size = max(2, int(0.01 * N))
    for k in [4, 6]:
        lab, C, L, F, xi = clustering.kefrin(Y, P, k, xi="auto", seed=seed, n_init=5, min_size=min_size)
        m = metrics.all_metrics(Y, A, lab)
        for key in ["SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "modularity"]:
            row[f"k{k}_{key if key != 'modularity' else 'Q'}"] = m[key]
        if k == 6 and not args.fast:
            aris = []
            for b in range(args.B):
                idx = np.sort(rng.choice(N, size=int(0.8 * N), replace=False))
                sub_A = A[idx][:, idx]
                lab_b, *_ = clustering.kefrin(Y[idx], graphs.diffusion_profile(sub_A, 2), k, xi=xi, seed=seed + b,
                                              n_init=2, min_size=max(2, int(0.01 * len(idx))))
                aris.append(adjusted_rand_score(lab[idx], lab_b))
            row["k6_boot_ari"] = float(np.mean(aris))
    row["sec"] = round(time.time() - t0, 1)
    return row


if __name__ == "__main__":
    main()
