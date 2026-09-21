"""Тесты временного слоя: выравнивание меток, KEFRiN-T на игрушечной динамической сети (смена типа
обнаруживается без задержки, ложных смен почти нет), матрица переходов, тест log-t, сводка устойчивости."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mo_types import graphs, interpret, temporal  # noqa: E402


def _toy_dynamic(N=150, K=3, V=4, T=5, t_event=3, n_move=20, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.zeros((T, N), dtype=int)
    truth[0] = np.arange(N) % K
    mu = rng.normal(size=(K, V)) * 3.0
    u = rng.normal(size=(N, V)) * 0.5
    movers = rng.choice(N, size=n_move, replace=False)
    Ys, Ps = [], []
    for t in range(T):
        lab = truth[t - 1].copy() if t > 0 else truth[0]
        if t == t_event:
            lab[movers] = (lab[movers] + 1) % K
        truth[t] = lab
        Y = mu[lab] + u + rng.normal(size=(N, V)) * 0.3
        A = graphs.knn_graph(Y, k=10, metric="cosine", mutual=True)
        Ys.append(Y)
        Ps.append(graphs.diffusion_profile(A))
    return Ys, Ps, truth, movers


def test_align_labels_undoes_permutation():
    rng = np.random.default_rng(1)
    prev = rng.integers(0, 4, size=300)
    perm = np.array([2, 0, 3, 1])
    assert np.array_equal(temporal.align_labels(prev, perm[prev], 4), prev)


def test_kefrin_t_tracks_true_partition_and_detects_movers():
    Ys, Ps, truth, movers = _toy_dynamic()
    res = temporal.kefrin_t(Ys, Ps, k=3, smoothing="affect_fe", reset_z=2.5, seed=42, n_init=3, init_window=2)
    L = res["labels"]
    assert L.shape == truth.shape
    for t in range(len(Ys)):
        assert ami(truth[t], L[t]) > 0.9
    # мигранты: тип после события отличается от типа до события уже в месяц события
    aligned = temporal.align_labels_general(truth[-1], L[-1])
    assert (aligned[movers] == truth[-1][movers]).mean() >= 0.8
    # веса истории в допустимых границах; в первый месяц истории нет
    assert res["alphas"][0] == 0 and np.all((res["alphas"] >= 0) & (res["alphas"] <= 0.95))
    stab = temporal.stability(L)
    assert stab["switch_max"] < 0.3 and stab["n_events"] <= 1


def test_markov_transitions_simple_chain():
    L = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]])
    m = interpret.markov_transitions(L, 2)
    P = m["P"]
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.isclose(P[0, 1], 1 / 3) and np.isclose(P[1, 0], 1 / 5)
    assert np.isclose(m["stationary"].sum(), 1.0)
    assert np.isclose(m["sojourn_months"][0], 3.0) and np.isclose(m["sojourn_months"][1], 5.0)


def test_logt_verdicts():
    rng = np.random.default_rng(2)
    T, N = 24, 200
    base = rng.normal(size=N)
    converging = np.exp(base[None, :] * np.linspace(1.0, 0.2, T)[:, None]) * 100
    diverging = np.exp(base[None, :] * np.linspace(0.2, 1.0, T)[:, None]) * 100
    assert interpret.club_convergence_logt(converging)["verdict"] == "сходимость"
    assert interpret.club_convergence_logt(diverging)["verdict"] == "нет сходимости"
