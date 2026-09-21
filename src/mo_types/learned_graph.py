"""Обучение графа из гладких сигналов (learned similarity).

Статический граф: LogModel — Kalofolias (AISTATS 2016) с параметризацией через среднюю степень
(Kalofolias & Perraudin, ICLR 2019). Временные графы: TGFA — Yamada, Tanaka, Ortega (2020), регуляризация
временной вариации рёбер (l1 или групповая l2). Реализация: пакет LTS4/graph-learning (BSD-3).

Узлы — МО, сигналы — признаки, наблюдаемые на всех узлах (по месяцам). Граф W минимизирует
Σ_ij W_ij ‖x_i − x_j‖² − α Σ_i log d_i + β ‖W‖²_F: сигналы гладки на графе, изолятов нет, лишние рёбра штрафуются.
"""
from __future__ import annotations

import numpy as np
from graph_learn.smooth_learning import LogModel
from graph_learn.temporal.graph_factor import TGFA
from scipy import sparse
from scipy.spatial.distance import squareform


def learn_static(X: np.ndarray, avg_degree: int = 15, maxit: int = 1000, tol: float = 1e-5,
                 edge_tol: float = 1e-3) -> tuple[sparse.csr_matrix, float]:
    """X: (N узлов × V признаков). Возвращает взвешенную смежность и параметр θ."""
    model = LogModel(avg_degree=avg_degree, maxit=maxit, tol=tol, edge_tol=edge_tol)
    model.fit(np.asarray(X, dtype=float).T)
    W = np.asarray(model.weights_)
    if W.ndim == 1:
        W = squareform(W)
    W = np.where(W < edge_tol, 0.0, W)
    np.fill_diagonal(W, 0.0)
    return sparse.csr_matrix(W), float(model.theta_)


def learn_temporal(Xs: list[np.ndarray], avg_degree: int = 15, l1_time: float = 0.0, l2_time: float = 0.0,
                   max_iter: int = 300, tol: float = 1e-3, step: float = 1.0, edge_tol: float = 1e-3):
    """Xs: список (N × V) по срезам одинаковой размерности V. Возвращает список смежностей и модель."""
    V = Xs[0].shape[1]
    x = np.vstack([np.asarray(X, dtype=float).T for X in Xs])  # (T·V сигналов, N узлов)
    model = TGFA(window_size=V, gamma=step, avg_degree=avg_degree, degree_reg=1, sparse_reg=1,
                 l1_time=l1_time, l2_time=l2_time, max_iter=max_iter, tol=tol)
    model.fit(x)
    As = []
    for w in range(model.weights_.shape[0]):
        W = squareform(np.asarray(model.weights_[w]))
        W = np.where(W < edge_tol, 0.0, W)
        As.append(sparse.csr_matrix(W))
    return As, model


def edge_set(A: sparse.csr_matrix) -> set:
    U = sparse.triu(A, k=1).tocoo()
    return set(zip(U.row.tolist(), U.col.tolist()))


def edge_jaccard(A: sparse.csr_matrix, B: sparse.csr_matrix) -> float:
    a, b = edge_set(A), edge_set(B)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def graph_stats(A: sparse.csr_matrix) -> dict:
    deg = np.asarray((A > 0).sum(1)).ravel()
    w = A.data
    return {"edges": int(A.nnz // 2), "deg_mean": float(deg.mean()), "deg_min": int(deg.min()),
            "deg_max": int(deg.max()), "isolates": int((deg == 0).sum()),
            "w_median": float(np.median(w)) if len(w) else 0.0, "w_max": float(w.max()) if len(w) else 0.0}
