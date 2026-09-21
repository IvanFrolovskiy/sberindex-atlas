"""Стек интерпретации типологии.

- mirkin_rule (в clustering.py): относительные отклонения центров, правило Миркина.
- imm_tree: пороговое дерево с K листьями (Iterative Mistake Minimization, Dasgupta, Frost, Moshkovitz,
  Rashtchian, ICML 2020) — каждый тип объясняется цепочкой порогов по одному признаку; «цена объяснимости» =
  доля объектов, попавших не в свой лист.
- surrogate_tree: обычное дерево решений как суррогат меток (sklearn), точность при 2K листьях.
- interval_patterns: интервальные описания типов (квантильные интервалы признаков, в духе pattern structures FCA).
- markov_transitions: матрица переходов, стационарное распределение, ожидаемое время пребывания;
  spatial_conditioned_transitions: переходы при условии модального типа дорожных соседей (Spatial Markov, Rey 2001).
- club_convergence_logt: тест log-t Phillips & Sul (2007) на сходимость уровней внутри типа.
- peers: «сопоставимые МО» — ближайшие соседи внутри типа.
- external_validation: согласие типов с внешними разбиениями (ARI/NMI) и доля объяснённой дисперсии (η²).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier


# ----------------------------------------------------------------------------- IMM
class ThresholdNode:
    __slots__ = ("feature", "threshold", "left", "right", "center", "n", "mistakes")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.center = None
        self.n = 0
        self.mistakes = 0


def imm_tree(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> ThresholdNode:
    """Строит пороговое дерево, разделяющее центры; каждый лист — один центр (кластер)."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)

    def build(idx: np.ndarray, cidx: np.ndarray) -> ThresholdNode:
        node = ThresholdNode()
        node.n = len(idx)
        if len(cidx) == 1:
            node.center = int(cidx[0])
            node.mistakes = int((labels[idx] != cidx[0]).sum())
            return node
        best = None
        C = centers[cidx]
        for j in range(X.shape[1]):
            cs = np.sort(C[:, j])
            for a, b in zip(cs[:-1], cs[1:]):
                if b <= a:
                    continue
                theta = (a + b) / 2.0
                left_c = centers[labels[idx], j] <= theta
                left_x = X[idx, j] <= theta
                mistakes = int((left_c != left_x).sum())
                if best is None or mistakes < best[0]:
                    best = (mistakes, j, theta)
        _, j, theta = best
        node.feature, node.threshold = int(j), float(theta)
        left_c = C[:, j] <= theta
        mask_x = X[idx, j] <= theta
        node.left = build(idx[mask_x], cidx[left_c])
        node.right = build(idx[~mask_x], cidx[~left_c])
        return node

    return build(np.arange(X.shape[0]), np.arange(centers.shape[0]))


def imm_predict(tree: ThresholdNode, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = np.empty(X.shape[0], dtype=int)
    for i, x in enumerate(X):
        node = tree
        while node.center is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        out[i] = node.center
    return out


def imm_rules(tree: ThresholdNode, feature_names: list[str]) -> dict[int, list[str]]:
    """Для каждого кластера — список условий вида «признак ≤ порог»."""
    rules = {}

    def walk(node, path):
        if node.center is not None:
            rules[node.center] = list(path)
            return
        f = feature_names[node.feature]
        walk(node.left, path + [f"{f} ≤ {node.threshold:.2f}"])
        walk(node.right, path + [f"{f} > {node.threshold:.2f}"])

    walk(tree, [])
    return rules


def surrogate_tree(X: np.ndarray, labels: np.ndarray, max_leaves: int, seed: int = 42) -> tuple[DecisionTreeClassifier, float]:
    t = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=seed).fit(X, labels)
    return t, float((t.predict(X) == labels).mean())


# ----------------------------------------------------------------------------- интервальные описания
def interval_patterns(X_raw: pd.DataFrame, labels: np.ndarray, q: tuple[float, float] = (0.1, 0.9)) -> pd.DataFrame:
    """Интервал [q_lo, q_hi] каждого признака в каждом типе + «разделяющая сила» = 1 − доля других типов внутри интервала."""
    rows = []
    for c in np.unique(labels):
        m = labels == c
        for col in X_raw.columns:
            x = X_raw[col].to_numpy(dtype=float)
            lo, hi = np.nanquantile(x[m], q[0]), np.nanquantile(x[m], q[1])
            others = x[~m]
            inside = np.nanmean((others >= lo) & (others <= hi)) if len(others) else np.nan
            rows.append({"type": int(c), "feature": col, "lo": lo, "hi": hi, "median": np.nanmedian(x[m]),
                         "others_inside": inside, "separation": 1.0 - inside})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- динамика
def markov_transitions(L: np.ndarray, k: int) -> dict:
    """L: T×N метки. Матрица переходов P, стационарное распределение π, среднее время пребывания."""
    T = np.zeros((k, k))
    for t in range(1, L.shape[0]):
        np.add.at(T, (L[t - 1], L[t]), 1.0)
    P = T / np.maximum(T.sum(axis=1, keepdims=True), 1.0)
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    pi = pi / pi.sum()
    sojourn = 1.0 / np.clip(1.0 - np.diag(P), 1e-9, None)
    return {"counts": T, "P": P, "stationary": pi, "sojourn_months": sojourn,
            "share_final": np.bincount(L[-1], minlength=k) / L.shape[1]}


def spatial_conditioned_transitions(L: np.ndarray, k: int, neighbors: np.ndarray) -> dict[int, np.ndarray]:
    """Переходы i при условии модального типа его соседей (neighbors: N×m индексы) в момент t−1.

    Возвращает словарь: тип соседей → матрица переходов k×k (Spatial Markov в категориальной форме)."""
    out = {c: np.zeros((k, k)) for c in range(k)}
    for t in range(1, L.shape[0]):
        prev = L[t - 1]
        nb = prev[neighbors]  # N×m
        modal = np.array([np.bincount(row, minlength=k).argmax() for row in nb])
        for c in range(k):
            m = modal == c
            np.add.at(out[c], (prev[m], L[t][m]), 1.0)
    return {c: M / np.maximum(M.sum(axis=1, keepdims=True), 1.0) for c, M in out.items()}


def club_convergence_logt(Y: np.ndarray, r: float = 0.3) -> dict:
    """Тест log-t Phillips & Sul (2007). Y: T×N уровни (положительные). Возвращает b̂, t-стат (HAC), вердикт.

    h_it = y_it / mean_i(y_it); H_t = mean_i (h_it − 1)²; регрессия log(H_1/H_t) − 2 log(log t) = a + b log t + ε
    по t = [rT], …, T. Сходимость не отвергается при t_b ≥ −1,65 (b ≥ 0 — полная, 0 > b > −2 — условная).
    """
    import statsmodels.api as sm
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    h = Y / Y.mean(axis=1, keepdims=True)
    H = ((h - 1.0) ** 2).mean(axis=1)
    t0 = max(int(np.floor(r * T)), 2)
    ts = np.arange(t0, T + 1)
    y = np.log(H[0] / H[ts - 1]) - 2.0 * np.log(np.log(ts))
    X = sm.add_constant(np.log(ts))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, int(np.floor(T ** (1 / 3))))})
    b, tb = float(res.params[1]), float(res.tvalues[1])
    verdict = "сходимость" if tb >= -1.65 else "нет сходимости"
    return {"b": b, "t": tb, "verdict": verdict, "n_obs": len(ts)}


# ----------------------------------------------------------------------------- сопоставимые МО и валидация
def peers(Y: np.ndarray, labels: np.ndarray, ids: np.ndarray, k: int = 5) -> pd.DataFrame:
    rows = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        Z = Y[idx]
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        order = np.argsort(D, axis=1)[:, :k]
        for a, row in zip(idx, order):
            rows.append({"territory_id": int(ids[a]), "type": int(c),
                         "peers": [int(ids[idx[b]]) for b in row], "peer_dist": [float(D[np.where(idx == a)[0][0], b]) for b in row]})
    return pd.DataFrame(rows)


def external_validation(labels: np.ndarray, attrs: pd.DataFrame, numeric_cols: list[str]) -> dict:
    out = {}
    for col in ["mo_type", "region_code"]:
        if col in attrs:
            ext = pd.factorize(attrs[col].astype(str))[0]
            out[f"ari_{col}"] = adjusted_rand_score(ext, labels)
            out[f"ami_{col}"] = adjusted_mutual_info_score(ext, labels)
    for col in numeric_cols:
        x = attrs[col].to_numpy(dtype=float)
        ok = np.isfinite(x)
        grand = x[ok].mean()
        ss_tot = ((x[ok] - grand) ** 2).sum()
        ss_between = sum(((x[ok & (labels == c)].mean() - grand) ** 2) * (ok & (labels == c)).sum()
                         for c in np.unique(labels) if (ok & (labels == c)).any())
        out[f"eta2_{col}"] = float(ss_between / ss_tot) if ss_tot > 0 else np.nan
    return out
