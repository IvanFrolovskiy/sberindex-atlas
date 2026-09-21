"""Внутренние индексы качества кластеризации.

Признаковое пространство: SW (silhouette), CH (Calinski–Harabasz), S_Dbw (Halkidi & Vazirgiannis 2001), WB.
Сетевое пространство: AVI, AVU, ANUI (Biswas & Biswas 2017; формулы воспроизводят реализацию библиотеки
Pattern, которой пользуется жюри), модульность Ньюмана Q, density modularity, MQ (Modularization Quality,
Mancoridis et al. 1998, взвешенный вариант TurboMQ).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from s_dbw import S_Dbw


def attribute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    k = len(np.unique(labels))
    out = {"K": k}
    if k < 2:
        return {**out, "SW": np.nan, "CH": np.nan, "S_Dbw": np.nan, "WB": np.nan}
    out["SW"] = float(silhouette_score(X, labels))
    out["CH"] = float(calinski_harabasz_score(X, labels))
    try:
        out["S_Dbw"] = float(S_Dbw(X, labels, centers_id=None, alg_noise="bind", centr="mean",
                                   nearest_centr=True, metric="euclidean"))
    except Exception:
        out["S_Dbw"] = np.nan
    cents = np.vstack([X[labels == c].mean(axis=0) for c in np.unique(labels)])
    lab_idx = np.searchsorted(np.unique(labels), labels)
    wss = float(((X - cents[lab_idx]) ** 2).sum())
    gc = X.mean(axis=0)
    sizes = np.bincount(lab_idx)
    bss = float((sizes * ((cents - gc) ** 2).sum(axis=1)).sum())
    out["WB"] = k * wss / bss if bss > 0 else np.inf
    return out


def _block_sums(A: sparse.csr_matrix, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    uniq, inv = np.unique(labels, return_inverse=True)
    k = len(uniq)
    M = sparse.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(k, len(inv)))
    S = (M @ A @ M.T).toarray()  # S[i,j] = сумма весов рёбер между кластерами i и j
    degrees = np.asarray(A.sum(axis=1)).ravel()
    sum_deg = np.asarray(M @ degrees).ravel()
    sizes = np.asarray(M.sum(axis=1)).ravel()
    return S, sum_deg, sizes


def graph_metrics(A: sparse.csr_matrix, labels: np.ndarray) -> dict:
    A = sparse.csr_matrix(A)
    S, sum_deg, sizes = _block_sums(A, labels)
    k = S.shape[0]
    m = A.sum() / 2.0
    # AVU / AVI / ANUI — как в Pattern (metrics/clustering_metrics.py)
    avu = 0.0
    for i in range(k):
        out_i = S[i].sum() - S[i, i]
        s = 0.0
        for j in range(k):
            if j == i:
                continue
            in_j = S[:, j].sum() - S[j, j]
            den = out_i + in_j - S[i, j]
            s += S[i, j] / den if den != 0 else 0.0
        avu += s / k
    avi = float(np.mean([S[i, i] / S[i].sum() if S[i].sum() != 0 else 0.0 for i in range(k)]))
    anui = 1.0 / (avu + 1.0 / avi) if avi != 0 else 0.0
    # модульность Ньюмана и density modularity
    Q = 0.0
    dQ = 0.0
    if m > 0:
        for i in range(k):
            internal = S[i, i] / 2.0
            Q += internal / m - (sum_deg[i] / (2 * m)) ** 2
            dQ += (internal - sum_deg[i] ** 2 / (4 * m)) / sizes[i] if sizes[i] else 0.0
    # MQ (TurboMQ, взвешенный): CF_k = mu_k / (mu_k + 0.5 * sum_j (eps_kj + eps_jk))
    mq = 0.0
    for i in range(k):
        mu = S[i, i] / 2.0
        eps = S[i].sum() - S[i, i]  # рёбра наружу (для симметричной A eps_kj = eps_jk)
        mq += mu / (mu + 0.5 * eps) if (mu + eps) > 0 else 0.0
    # средняя кондуктанса
    cond = float(np.mean([(S[i].sum() - S[i, i]) / S[i].sum() if S[i].sum() > 0 else 1.0 for i in range(k)]))
    return {"K": k, "AVI": avi, "AVU": avu, "ANUI": anui, "modularity": Q,
            "density_modularity": dQ, "MQ": mq, "conductance": cond}


def all_metrics(X: np.ndarray, A: sparse.csr_matrix, labels: np.ndarray) -> dict:
    d = attribute_metrics(X, labels)
    d.update({k: v for k, v in graph_metrics(A, labels).items() if k != "K"})
    return d
