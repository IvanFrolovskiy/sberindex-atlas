"""Проверка KEFRiN-T на синтетических динамических атрибутированных сетях с известной истиной.

Генератор (в духе DynBenchmark, Brisson–Bothorel–Duminy 2025, но прозрачный): K сообществ, атрибуты
y_it = μ_k(i,t) + u_i + ε_it (центр типа + устойчивый эффект узла + месячный шум), сеть — SBM с
персистентностью рёбер между месяцами; событие: в месяц t_event доля nodes migrate меняет сообщество;
постоянный фоновый «дрейф» 1%/мес. Метрики: NMI с истиной по месяцам, доля верно переназначенных мигрантов
через 0/1/2 месяца после события, ложные смены среди немигрантов, поведение α_t.

Запуск: .venv312/bin/python scripts/synthetic_validation.py [--seeds 5]
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
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score as nmi  # noqa: E402

from mo_types import graphs, temporal  # noqa: E402
from mo_types.temporal import align_labels_general  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)


def generate(N=600, K=5, T=12, V=8, sep=2.0, s_u=1.0, s_e=1.0, p_in=0.08, p_out=0.01, persist=0.7,
             t_event=6, migrate=0.15, churn=0.01, seed=0):
    """sep — σ центров типов по признаку (расстояние между центрами ≈ sep·sqrt(2V)); s_u — σ устойчивых
    эффектов узла; s_e — σ месячного шума. Откалиброванный по данным сценарий: V=7, sep=0,78, s_u=0,66, s_e=0,28."""
    rng = np.random.default_rng(seed)
    truth = np.zeros((T, N), dtype=int)
    truth[0] = rng.integers(K, size=N)
    mu = rng.normal(size=(K, V)) * sep
    u = rng.normal(size=(N, V)) * s_u
    Ys, As = [], []
    prev_edges = None
    for t in range(T):
        if t > 0:
            lab = truth[t - 1].copy()
            n_move = int(churn * N) + (int(migrate * N) if t == t_event else 0)
            movers = rng.choice(N, size=n_move, replace=False)
            lab[movers] = (lab[movers] + rng.integers(1, K, size=n_move)) % K
            truth[t] = lab
        lab = truth[t]
        Ys.append(mu[lab] + u + rng.normal(size=(N, V)) * s_e)
        # SBM с персистентностью: часть рёбер наследуется, остальное перерисовывается
        same = lab[:, None] == lab[None, :]
        prob = np.where(same, p_in, p_out)
        new = rng.random((N, N)) < prob
        new = np.triu(new, 1)
        if prev_edges is not None:
            keep = np.triu(rng.random((N, N)) < persist, 1) & prev_edges
            new = np.where(np.triu(rng.random((N, N)) < persist, 1), keep, new)
        prev_edges = new
        A = sparse.csr_matrix((new | new.T).astype(float))
        As.append(A)
    return truth, Ys, As


def evaluate(name, labels, truth, t_event, K):
    T = truth.shape[0]
    per_t = [nmi(truth[t], labels[t]) for t in range(T)]
    # выравниваем метки к истине для подсчёта мигрантов
    aligned = np.vstack([align_labels_general(truth[t], labels[t]) for t in range(T)])
    movers = truth[t_event] != truth[t_event - 1]
    stayers = ~movers
    detect = {}
    for d in range(3):
        tt = min(t_event + d, T - 1)
        detect[f"movers_correct_+{d}"] = float((aligned[tt][movers] == truth[tt][movers]).mean())
    false_switch = float((aligned[t_event][stayers] != aligned[t_event - 1][stayers]).mean())
    return {"method": name, "nmi_mean": float(np.mean(per_t)), "nmi_min": float(np.min(per_t)),
            "nmi_after_event": float(np.mean(per_t[t_event:])), **detect, "false_switch_at_event": false_switch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    rows = []
    alphas = []
    scenarios = {"noise1": dict(s_e=1.0), "noise2": dict(s_e=2.0),
                 "calibrated": dict(V=7, sep=0.78, s_u=0.66, s_e=0.28)}
    for sc_name, sc in scenarios.items():
        for seed in range(args.seeds):
            truth, Ys, As = generate(seed=seed, **sc)
            s_e = sc_name
            K, T, t_event = 5, len(Ys), 6
            Ps = [graphs.diffusion_profile(A, steps=2) for A in As]
            t0 = time.time()
            for name, kw in [("kefrin_independent", None), ("kefrin_warm", dict(smoothing="none")),
                             ("kefrin_fixed0.5", dict(smoothing="fixed", alpha_fixed=0.5)),
                             ("kefrin_t_affect", dict(smoothing="affect")),
                             ("kefrin_t_affect_fe_noreset", dict(smoothing="affect_fe", reset_z=None)),
                             ("kefrin_t_affect_fe_z4", dict(smoothing="affect_fe", reset_z=4.0)),
                             ("kefrin_t_affect_fe_z2.5", dict(smoothing="affect_fe", reset_z=2.5)),
                             ("kefrin_t_affect_fe_z2.5_2pass", dict(smoothing="affect_fe", reset_z=2.5, two_pass=True)),
                             ("kefrin_t_affect_fe_z2.5_ma3", dict(smoothing="affect_fe", reset_z=2.5, ma=3)),
                             ("kefrin_t_affect_fe_z2.5_2pass_ma3", dict(smoothing="affect_fe", reset_z=2.5, two_pass=True, ma=3))]:
                if kw is None:
                    from mo_types.clustering import kefrin
                    labels = np.vstack([kefrin(Y, P, K, xi="auto", seed=seed, n_init=5)[0] for Y, P in zip(Ys, Ps)])
                    res = {"labels": labels, "alphas": np.zeros(T)}
                else:
                    kw = dict(kw)
                    two_pass = kw.pop("two_pass", False)
                    ma = kw.pop("ma", 1)
                    Ys_in = Ys if ma <= 1 else [np.mean(Ys[max(0, t - ma // 2): t + ma // 2 + 1], axis=0) for t in range(T)]
                    res = temporal.kefrin_t(Ys_in, Ps, K, xi="auto", seed=seed, n_init=5, init_window=2, **kw)
                    if two_pass:
                        modal = pd.DataFrame(res["labels"]).mode(axis=0).iloc[0].to_numpy().astype(int)
                        res = temporal.kefrin_t(Ys_in, Ps, K, xi="auto", seed=seed, n_init=5, init_window=2, init_labels=modal, **kw)
                r = evaluate(name, res["labels"], truth, t_event, K)
                r.update({"s_e": s_e, "seed": seed})
                rows.append(r)
                if name.startswith("kefrin_t_affect"):
                    alphas.append(pd.DataFrame({"method": name, "s_e": s_e, "seed": seed, "t": range(T), "alpha": res["alphas"]}))
            # базовые линии: K-means по месяцам, temporal Leiden
            km = np.vstack([KMeans(K, n_init=5, random_state=seed).fit_predict(Y) for Y in Ys])
            rows.append({**evaluate("kmeans_monthly", km, truth, t_event, K), "s_e": s_e, "seed": seed})
            for omega in [0.3, 1.0]:
                tl = graphs.temporal_leiden(As, interslice_weight=omega, resolution=1.0, seed=seed)
                rows.append({**evaluate(f"temporal_leiden_w{omega}", tl, truth, t_event, K), "s_e": s_e, "seed": seed})
            print(time.strftime("%H:%M:%S"), f"s_e={s_e} seed={seed} готово ({time.time() - t0:.0f} с)", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "synthetic_results.csv", index=False)
    pd.concat(alphas).to_csv(OUT / "synthetic_alphas.csv", index=False)
    summ = tab.groupby(["s_e", "method"]).mean(numeric_only=True).drop(columns=["seed"]).round(3)
    print(summ.to_string())
    al = pd.concat(alphas).groupby(["method", "s_e", "t"]).alpha.mean().unstack("t").round(2)
    print("\nα_t по месяцам (событие в t=6):\n" + al.to_string())


if __name__ == "__main__":
    main()
