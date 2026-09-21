"""Алгоритмы: K-means, KEFRiN (Shalileh & Mirkin, Entropy 2022), правило Миркина для интерпретации."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def kmeans(X: np.ndarray, k: int, seed: int = 42, n_init: int = 10) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(X)


# ----------------------------------------------------------------------------- KEFRiN
def _dist(X: np.ndarray, C: np.ndarray, kind: str) -> np.ndarray:
    """Матрица расстояний (N×K) между строками X и центрами C."""
    if kind == "euclidean":  # ‖x−c‖² = ‖x‖² + ‖c‖² − 2x·c, без трёхмерного массива
        d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2.0 * X @ C.T
        return np.maximum(d, 0.0)
    if kind == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        return 1.0 - Xn @ Cn.T
    if kind == "manhattan":
        return np.abs(X[:, None, :] - C[None, :, :]).sum(-1)
    raise ValueError(kind)


def kefrin(Y: np.ndarray, P: np.ndarray, k: int, rho: float = 1.0, xi: float = 1.0,
           distance: str = "euclidean", seed: int = 42, n_init: int = 5, max_iter: int = 100,
           init_labels: np.ndarray | None = None, history: tuple | None = None, gamma: float = 0.0,
           switch_penalty: float = 0.0, min_size: int = 1):
    """K-means, расширенный на feature-rich сети (KEFRiN), с опциональной «стоимостью истории».

    Y — признаки (N×V), P — строки сетевой матрицы (N×N, обычно нормированная смежность).
    Правило минимального совместного расстояния: rho·d(y_i,c_k) + xi·d(p_i,λ_k) [+ gamma·d(c_k, c_k^{prev})...].
    history=(C_prev, L_prev) и gamma>0 дают центроидное сглаживание (эволюционный вариант, см. RESEARCH2 §2).
    switch_penalty>0 при заданных init_labels добавляет штраф за уход объекта из прежнего кластера
    (сглаживание назначений, Chakrabarti et al. 2006).
    Возвращает (labels, C, L, F, xi).
    """
    rng = np.random.default_rng(seed)
    N = Y.shape[0]
    if xi == "auto":  # уравниваем суммарный разброс двух пространств (аналог стандартизации связей в KEFRiN)
        if distance == "cosine":
            xi = float(rho)
        else:
            ss_y = _dist(Y, Y.mean(0, keepdims=True), distance).sum()
            ss_p = _dist(P, P.mean(0, keepdims=True), distance).sum()
            xi = float(rho * ss_y / ss_p) if ss_p > 0 else 1.0
    best = None
    for run in range(n_init):
        if init_labels is not None and run == 0:
            labels = init_labels.copy()
            C = np.vstack([Y[labels == c].mean(0) for c in range(k)])
            L = np.vstack([P[labels == c].mean(0) for c in range(k)])
        else:
            # K-means++ / MaxMin инициализация по совместному расстоянию
            idx = [rng.integers(N)]
            while len(idx) < k:
                d = rho * _dist(Y, Y[idx], distance) + xi * _dist(P, P[idx], distance)
                f = d.sum(1)
                f[idx] = -np.inf
                idx.append(int(np.argmax(f)))
            C, L = Y[idx].copy(), P[idx].copy()
            labels = None
        penalty = None
        if switch_penalty > 0 and init_labels is not None:
            penalty = np.full((N, k), switch_penalty)
            penalty[np.arange(N), init_labels] = 0.0
        for it in range(max_iter):
            D = rho * _dist(Y, C, distance) + xi * _dist(P, L, distance)
            if penalty is not None:
                D = D + penalty
            new = D.argmin(1)
            if min_size > 1:  # карликовый кластер: распустить и пересеять из хвоста самого большого
                sizes = np.bincount(new, minlength=k)
                for c in np.where((sizes > 0) & (sizes < min_size))[0]:
                    members = np.where(new == c)[0]
                    D2 = D.copy()
                    D2[:, c] = np.inf
                    new[members] = D2[members].argmin(1)
                    big = np.bincount(new, minlength=k).argmax()
                    cand = np.where(new == big)[0]
                    far = cand[np.argsort(-D[cand, big])[:min_size]]
                    new[far] = c
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                mask = labels == c
                if mask.any():
                    C[c] = Y[mask].mean(0)
                    L[c] = P[mask].mean(0)
                    if history is not None and gamma > 0:
                        C[c] = (C[c] + gamma * history[0][c]) / (1 + gamma)
                        L[c] = (L[c] + gamma * history[1][c]) / (1 + gamma)
                else:  # пустой кластер — переносим самый далёкий объект
                    far = int(D.min(1).argmax())
                    labels[far] = c
                    C[c], L[c] = Y[far], P[far]
            if distance == "cosine":
                C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
                L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
        F = float((rho * _dist(Y, C, distance) + xi * _dist(P, L, distance)).min(1).sum())
        if best is None or F < best[3]:
            best = (labels.copy(), C.copy(), L.copy(), F)
    return best + (xi,)


# ----------------------------------------------------------------------------- интерпретация
def mirkin_rule(X_raw: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """d_kv = (c_kv − g_v) / g_v на нестандартизованных признаках (Mirkin; Alvandyan & Shalileh 2024)."""
    g = X_raw.mean(axis=0)
    rows = {}
    for c in np.unique(labels):
        ck = X_raw[labels == c].mean(axis=0)
        rows[c] = (ck - g) / g.replace(0, np.nan)
    out = pd.DataFrame(rows).T
    out.index.name = "cluster"
    out.insert(0, "size", pd.Series(labels).value_counts().sort_index().values)
    return out
