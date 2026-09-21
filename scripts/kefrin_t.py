"""KEFRiN-T против базовых вариантов на 24 месячных срезах.

Запуск: .venv312/bin/python scripts/kefrin_t.py [--k 5,6,7,8] [--graphs knn|learned]
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from mo_types import data, features, graphs, metrics, temporal  # noqa: E402

OUT = data.ROOT / "outputs" / "kefrin_t"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_slices(cfg, graph_source="knn", graph_dir=None, use_context=True):
    panel, attrs = data.load_processed(cfg)
    dates = sorted(panel.index.get_level_values("date").unique())
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    tids = panel.index.get_level_values("territory_id").unique()
    stat = features.static_attributes(attrs.reindex(tids))
    ctx_cols = [c for c in stat.columns if stat[c].notna().mean() > 0.9]
    X_ctx = features.zscore(stat[ctx_cols]).to_numpy()
    month_cols = [c for c in feat.columns if c.startswith("clr_")] + ["rel_level"]
    Ys, As, Xb = [], [], []
    for d in dates:
        Xm = features.zscore(feat.xs(d, level="date").reindex(tids)[month_cols]).to_numpy()
        Xb.append(Xm)
        Ys.append(np.hstack([Xm, X_ctx]) if use_context else Xm)
        if graph_source == "knn":
            As.append(graphs.knn_graph(Xm, k=cfg["graph"]["k"], metric="cosine", mutual=True))
    if graph_source == "learned":
        As = [sparse.load_npz(Path(graph_dir) / f"A_t{t:02d}.npz") for t in range(len(dates))]
    return dates, tids, Ys, As, Xb, attrs


def evaluate(name, labels, Ys, As, dates):
    st = temporal.stability(labels)
    per = []
    for t in range(len(dates)):
        m = metrics.all_metrics(Ys[t], As[t], labels[t])
        per.append({"t": t, "date": dates[t].strftime("%Y-%m"), **m})
    per = pd.DataFrame(per)
    summ = {"method": name, **st,
            "SW": per["SW"].mean(), "CH": per["CH"].mean(), "S_Dbw": per["S_Dbw"].mean(),
            "AVI": per["AVI"].mean(), "AVU": per["AVU"].mean(), "ANUI": per["ANUI"].mean(),
            "MQ": per["MQ"].mean(), "Q": per["modularity"].mean(), "K_mean": per["K"].mean()}
    return summ, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="5,6,7,8")
    ap.add_argument("--graphs", default="knn")
    ap.add_argument("--switch", type=float, default=0.0)
    ap.add_argument("--graph-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reset-z", type=float, default=2.5, help="порог теста инновации для сброса сглаживания узла")
    ap.add_argument("--no-context", action="store_true", help="только признаки потребления (без Росстата и доступности)")
    ap.add_argument("--two-pass", action="store_true", help="второй проход: стартовое разбиение = модальные типы первого прохода")
    ap.add_argument("--ma", type=int, default=1, help="центрированное скользящее среднее признаков по месяцам (1 = без сглаживания)")
    args = ap.parse_args()
    cfg = data.load_config()
    seed = cfg["clustering"]["seed"]
    dates, tids, Ys, As, Xb, attrs = build_slices(cfg, args.graphs, args.graph_dir, use_context=not args.no_context)
    if args.ma > 1:
        h = args.ma // 2
        Ys = [np.mean(Ys[max(0, t - h): t + h + 1], axis=0) for t in range(len(Ys))]
        log(f"признаки сглажены центрированным средним по {args.ma} мес.")
    suffix = args.graphs + (f"_{args.tag}" if args.tag else "")
    Ps = [graphs.diffusion_profile(A, steps=2) for A in As]
    log(f"срезов {len(dates)}, N={len(tids)}, признаков {Ys[0].shape[1]}, графы: {args.graphs}")

    rows, per_rows, alpha_rows = [], [], []
    labels_out = {}
    for k in [int(x) for x in args.k.split(",")]:
        for name, kw in [
            ("kefrin_warm", dict(smoothing="none")),
            ("kefrin_fixed0.5", dict(smoothing="fixed", alpha_fixed=0.5)),
            ("kefrin_t_affect", dict(smoothing="affect")),
            ("kefrin_t_affect_fe_noreset", dict(smoothing="affect_fe", reset_z=None)),
            ("kefrin_t_affect_fe", dict(smoothing="affect_fe", reset_z=args.reset_z)),
        ]:
            t0 = time.time()
            res = temporal.kefrin_t(Ys, Ps, k, rho=1.0, xi="auto", distance="euclidean", seed=seed, n_init=10,
                                    switch_penalty=args.switch, **kw)
            if args.two_pass:
                modal = pd.DataFrame(res["labels"]).mode(axis=0).iloc[0].to_numpy().astype(int)
                res = temporal.kefrin_t(Ys, Ps, k, rho=1.0, xi="auto", distance="euclidean", seed=seed, n_init=10,
                                        switch_penalty=args.switch, init_labels=modal, **kw)
            summ, per = evaluate(f"{name}", res["labels"], Ys, As, dates)
            summ.update({"K": k, "sec": round(time.time() - t0, 1), "alpha_mean": float(res["alphas"][1:].mean()),
                         "beta_mean": float(res["betas"][1:].mean()), "xi": res["xi"],
                         "reset_share": float(res["resets"].mean()) if "resets" in res else 0.0})
            rows.append(summ)
            per["method"], per["Kset"] = name, k
            per_rows.append(per)
            labels_out[(name, k)] = res["labels"]
            if name == "kefrin_t_affect_fe":
                pd.DataFrame(res["labels"].T, index=tids, columns=[d.strftime("%Y-%m") for d in dates]).to_csv(
                    OUT / f"labels_affect_fe_K{k}_{suffix}.csv")
            if name.startswith("kefrin_t_affect"):
                alpha_rows.append(pd.DataFrame({"method": name, "K": k, "date": [d.strftime("%Y-%m") for d in dates],
                                                "alpha": res["alphas"], "beta": res["betas"]}))
            log(f"K={k} {name}: AMI med {summ['ami_median']:.3f}, switch med {summ['switch_median']:.3f} max {summ['switch_max']:.2f}, "
                f"events {summ['n_events']}, min size {summ['min_cluster_size']}, SW {summ['SW']:.3f}, Q {summ['Q']:.3f}, "
                f"α̅={summ['alpha_mean']:.2f} β̅={summ['beta_mean']:.2f} ({summ['sec']} с)")
    # базовая линия: временной Leiden
    for omega in [0.5, 2.0]:
        t0 = time.time()
        memb = graphs.temporal_leiden(As, interslice_weight=omega, resolution=0.5, seed=seed)
        summ, per = evaluate(f"temporal_leiden_w{omega}", memb, Ys, As, dates)
        summ.update({"K": int(summ["K_mean"]), "sec": round(time.time() - t0, 1)})
        rows.append(summ)
        log(f"temporal Leiden ω={omega}: сообществ {summ['K_mean']:.1f}, AMI med {summ['ami_median']:.3f}, "
            f"switch med {summ['switch_median']:.3f}, Q {summ['Q']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"summary_{suffix}.csv", index=False)
    pd.concat(per_rows).to_csv(OUT / f"per_month_{suffix}.csv", index=False)
    if alpha_rows:
        pd.concat(alpha_rows).to_csv(OUT / f"alphas_{suffix}.csv", index=False)
    cols = ["method", "K", "ami_median", "switch_median", "switch_max", "n_events", "share_never_switch", "SW", "CH", "S_Dbw", "AVI", "ANUI", "MQ", "Q", "alpha_mean", "beta_mean", "reset_share"]
    log("сводка:\n" + table[[c for c in cols if c in table]].round(3).to_string(index=False))
    # сохранить метки лучшего affect-варианта по компромиссу (AMI + SW)
    best = table[table.method == "kefrin_t_affect_fe"].assign(score=lambda d: d.ami_median + d.SW).sort_values("score").iloc[-1]
    lab = labels_out[("kefrin_t_affect_fe", int(best["K"]))]
    pd.DataFrame(lab.T, index=tids, columns=[d.strftime("%Y-%m") for d in dates]).to_csv(OUT / f"labels_affect_K{int(best['K'])}_{suffix}.csv")
    with open(OUT / f"best_{suffix}.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in best.items()}, f, ensure_ascii=False, indent=1)
    log(f"лучший affect: K={int(best['K'])}; метки сохранены")


if __name__ == "__main__":
    main()
