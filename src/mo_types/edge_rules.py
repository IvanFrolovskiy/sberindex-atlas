"""Правила ребра (сходство МО) и способы разрежения сети.

Сходства (плотные N×N):
- cosine(X)            — косинус профилей (структура трат, уровень, динамика);
- corr(Z)              — корреляция рядов приростов (кто движется вместе);
- corr_lag(Z, L)       — максимум корреляции по лагам ±L (кто опережает кого);
- dtw_sim(S)           — DTW-расстояние рядов → сходство exp(−d/σ) (dtaidistance);
- dtw_ndim_sim(S3)     — многомерный DTW (композиция трат по месяцам);
- road_sim(D, λ)       — гравитация exp(−d_дор/λ);
- snf_fuse(...)        — слияние нескольких сходств (Similarity Network Fusion, Wang et al. 2014);
- обученный граф       — см. learned_graph.py.
Разрежение: knn_sparsify (обычный/взаимный), epsilon_sparsify (порог по плотности), tmfg_sparsify (планарный
максимально фильтрованный граф, Massara et al. 2016), disparity_sparsify (disparity filter, Serrano et al. 2009).
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import igraph as ig
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------------- сходства
def standardize_rows(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    m = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-12
    return (S - m) / sd


def cosine(X: np.ndarray) -> np.ndarray:
    return cosine_similarity(np.asarray(X, dtype=float))


def corr(Z: np.ndarray) -> np.ndarray:
    """Z — ряды по строкам (N×T). Корреляция Пирсона."""
    Zs = standardize_rows(Z)
    return (Zs @ Zs.T) / Zs.shape[1]


def corr_lag(Z: np.ndarray, max_lag: int = 2) -> np.ndarray:
    """Максимум корреляции по сдвигам −L..L: ловит «один регион опережает другой». Симметрична."""
    Z = np.asarray(Z, dtype=float)
    N, T = Z.shape
    best = np.full((N, N), -1.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            A, B = Z[:, lag:], Z[:, : T - lag]
        else:
            A, B = Z[:, : T + lag], Z[:, -lag:]
        A, B = standardize_rows(A), standardize_rows(B)
        C = (A @ B.T) / A.shape[1]
        best = np.maximum(best, C)
    return best


def _dtw_matrix(S3: np.ndarray, window: int) -> np.ndarray:
    """Матрица DTW-расстояний (окно Сако–Чиба = window), tslearn (numba, параллельно)."""
    from tslearn.metrics import cdist_dtw
    D = cdist_dtw(np.asarray(S3, dtype=float), sakoe_chiba_radius=window, n_jobs=-1)
    np.fill_diagonal(D, 0.0)
    return D


def dtw_sim(S: np.ndarray, window: int = 3, scale: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Одномерные ряды по строкам (N×T), стандартизованные по строке."""
    S3 = standardize_rows(S)[:, :, None]
    D = _dtw_matrix(S3, window)
    return _dist_to_sim(D, scale), D


def dtw_ndim_sim(S3: np.ndarray, window: int = 3, scale: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """S3: N×T×d (например, CLR-композиция по месяцам)."""
    D = _dtw_matrix(np.asarray(S3, dtype=float), window)
    return _dist_to_sim(D, scale), D


def _dist_to_sim(D: np.ndarray, scale: str = "median") -> np.ndarray:
    off = D[np.triu_indices_from(D, k=1)]
    sigma = np.median(off[np.isfinite(off)]) if scale == "median" else float(scale)
    return np.exp(-D / (sigma + 1e-12))


def road_matrix(conn: "pd.DataFrame", ids: np.ndarray, fill_km: float = 5000.0) -> np.ndarray:
    """Плотная матрица дорожных расстояний (км) между МО из списка ids; пары без связи → fill_km."""
    pos = {int(t): i for i, t in enumerate(ids)}
    N = len(ids)
    D = np.full((N, N), fill_km, dtype=float)
    x = conn["territory_id_x"].to_numpy()
    y = conn["territory_id_y"].to_numpy()
    d = conn["distance"].to_numpy(dtype=float)
    for a, b, v in zip(x, y, d):
        i, j = pos.get(int(a)), pos.get(int(b))
        if i is not None and j is not None:
            D[i, j] = min(D[i, j], v)
            D[j, i] = min(D[j, i], v)
    np.fill_diagonal(D, 0.0)
    return D


def road_sim(D_km: np.ndarray, lam_km: float = 200.0) -> np.ndarray:
    return np.exp(-D_km / lam_km)


def snf_affinity(D: np.ndarray, K: int = 20, mu: float = 0.5) -> np.ndarray:
    """Аффинность SNF из матрицы расстояний: W_ij = exp(−d_ij²/(μ ε_ij)), ε_ij = (mean_i + mean_j + d_ij)/3."""
    D = np.asarray(D, dtype=float)
    N = D.shape[0]
    srt = np.sort(D, axis=1)[:, 1 : K + 1]
    mean_k = srt.mean(axis=1)
    eps = (mean_k[:, None] + mean_k[None, :] + D) / 3.0 + 1e-12
    W = np.exp(-(D ** 2) / (mu * eps))
    np.fill_diagonal(W, 0.0)
    return W


def snf_fuse(*affinities: np.ndarray, K: int = 20, t: int = 20) -> np.ndarray:
    """Similarity Network Fusion (Wang et al., Nature Methods 2014), собственная реализация.

    P^(m) — полная нормированная матрица (P_ii = 1/2, остальное W_ij / 2Σ_j W_ij), S^(m) — локальная
    (kNN-ограниченная, строки нормированы). Итерации: P^(m) ← S^(m) · mean_{l≠m} P^(l) · S^(m)ᵀ, затем
    перенормировка; итог — среднее P^(m).
    """
    def full_norm(W):
        W = np.asarray(W, dtype=float).copy()
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        P = W / (2.0 * rs)
        np.fill_diagonal(P, 0.5)
        return P

    def local_norm(W):
        W = np.asarray(W, dtype=float).copy()
        np.fill_diagonal(W, 0.0)
        idx = np.argpartition(-W, kth=K, axis=1)[:, :K]
        S = np.zeros_like(W)
        rows = np.repeat(np.arange(W.shape[0]), K)
        S[rows, idx.ravel()] = W[rows, idx.ravel()]
        S = S / (S.sum(axis=1, keepdims=True) + 1e-12)
        return S

    Ps = [full_norm(W) for W in affinities]
    Ss = [local_norm(W) for W in affinities]
    m = len(Ps)
    for _ in range(t):
        new = []
        for i in range(m):
            others = sum(Ps[j] for j in range(m) if j != i) / max(m - 1, 1)
            Pi = Ss[i] @ others @ Ss[i].T
            new.append(full_norm(Pi + Pi.T))
        Ps = new
    fused = sum(Ps) / m
    fused = (fused + fused.T) / 2.0
    np.fill_diagonal(fused, 0.0)
    return fused


def sim_to_dist(S: np.ndarray) -> np.ndarray:
    """Для сходств в [−1, 1] или [0, 1]: расстояние = 1 − S (обрезано снизу нулём)."""
    return np.clip(1.0 - S, 0.0, None)


# ----------------------------------------------------------------------------- разрежение
def _sym_sparse(rows, cols, vals, n, mode="max") -> sparse.csr_matrix:
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.minimum(A.T) if mode == "min" else A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def knn_sparsify(S: np.ndarray, k: int = 15, mutual: bool = True, fix_isolates: bool = True) -> sparse.csr_matrix:
    S = np.array(S, dtype=float, copy=True)
    np.fill_diagonal(S, -np.inf)
    n = S.shape[0]
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    vals = np.clip(S[rows, idx.ravel()], 1e-9, None)
    A = _sym_sparse(rows, idx.ravel(), vals, n, mode="min" if mutual else "max")
    if mutual and fix_isolates:
        deg = np.asarray((A > 0).sum(1)).ravel()
        iso = np.where(deg == 0)[0]
        if len(iso):
            best = np.argmax(S[iso], axis=1)
            fix = sparse.csr_matrix((np.clip(S[iso, best], 1e-9, None), (iso, best)), shape=(n, n))
            A = A.maximum(fix).maximum(fix.T)
    return A


def epsilon_sparsify(S: np.ndarray, n_edges: int) -> sparse.csr_matrix:
    """Оставить n_edges самых сильных пар (глобальный порог)."""
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = S[iu]
    keep = np.argpartition(-vals, kth=min(n_edges, len(vals) - 1))[:n_edges]
    A = sparse.csr_matrix((np.clip(vals[keep], 1e-9, None), (iu[0][keep], iu[1][keep])), shape=(n, n))
    return A.maximum(A.T)


def tmfg_sparsify(S: np.ndarray) -> sparse.csr_matrix:
    """Планарный максимально фильтрованный граф (3N−6 рёбер) по матрице сходств ≥ 0."""
    import pandas as pd
    from fast_tmfg import TMFG
    W = np.clip(np.asarray(S, dtype=float), 0.0, None)
    np.fill_diagonal(W, 0.0)
    res = TMFG().fit_transform(weights=pd.DataFrame(W), output="weighted_sparse_W_matrix")
    M = res[-1] if isinstance(res, tuple) else res
    M = np.asarray(pd.DataFrame(M), dtype=float)
    M = np.maximum(M, M.T)
    np.fill_diagonal(M, 0.0)
    return sparse.csr_matrix(M)


def disparity_sparsify(S: np.ndarray, alpha: float = 0.05, k_pre: int = 50) -> sparse.csr_matrix:
    """Disparity filter (Serrano et al. 2009) на графе из k_pre сильнейших связей каждого узла."""
    from networkx_backbone import statistical, filters
    A0 = knn_sparsify(S, k=k_pre, mutual=False, fix_isolates=False)
    G = nx.from_scipy_sparse_array(A0)
    H = statistical.disparity_filter(G, weight="weight")
    B = filters.threshold_filter(H, "disparity_pvalue", alpha, mode="below")
    A = nx.to_scipy_sparse_array(B, nodelist=range(S.shape[0]), weight="weight", format="csr")
    A = sparse.csr_matrix(A)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


# ----------------------------------------------------------------------------- свойства графа
def to_igraph(A: sparse.csr_matrix) -> ig.Graph:
    U = sparse.triu(A, k=1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(U.row.tolist(), U.col.tolist())), directed=False)
    g.es["weight"] = U.data.tolist()
    return g


def graph_summary(A: sparse.csr_matrix, labels: np.ndarray | None = None, numeric: np.ndarray | None = None) -> dict:
    deg = np.asarray((A > 0).sum(1)).ravel()
    ncomp, _ = connected_components(A, directed=False)
    g = to_igraph(A)
    out = {"edges": int(A.nnz // 2), "deg_mean": float(deg.mean()), "deg_min": int(deg.min()),
           "deg_max": int(deg.max()), "isolates": int((deg == 0).sum()), "components": int(ncomp),
           "clustering": float(g.transitivity_avglocal_undirected(mode="zero")),
           "deg_assort": float(g.assortativity_degree(directed=False)) if A.nnz else np.nan}
    if labels is not None:
        out["assort_labels"] = float(g.assortativity_nominal(types=[int(x) for x in labels], directed=False))
        U = sparse.triu(A, k=1).tocoo()
        same = (labels[U.row] == labels[U.col])
        out["same_label_share"] = float(same.mean()) if len(same) else np.nan
    if numeric is not None:
        out["assort_numeric"] = float(g.assortativity(types1=[float(x) for x in numeric], directed=False))
    return out


def edge_jaccard(A: sparse.csr_matrix, B: sparse.csr_matrix) -> float:
    a = set(zip(*sparse.triu(A, 1).nonzero()))
    b = set(zip(*sparse.triu(B, 1).nonzero()))
    return len(a & b) / len(a | b) if (a | b) else 1.0
