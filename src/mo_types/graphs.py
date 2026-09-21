"""Сети сходства и обёртки над igraph/leidenalg."""
from __future__ import annotations

import numpy as np
import igraph as ig
import leidenalg as la
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def knn_graph(X: np.ndarray, k: int = 15, metric: str = "cosine", mutual: bool = True,
              weighting: str = "similarity") -> sparse.csr_matrix:
    """Взвешенная неориентированная kNN-сеть.

    weighting='similarity': вес = 1 − d (cosine) или exp(−d²/σ_iσ_j) (euclidean, адаптивный RBF);
    mutual=True оставляет только взаимные соседи.
    """
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    if metric == "cosine":
        w = 1.0 - dist
    else:
        sigma = dist[:, -1] + 1e-12
        w = np.exp(-(dist ** 2) / (sigma[:, None] * sigma[idx]))
    rows = np.repeat(np.arange(n), k)
    A = sparse.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    if mutual:
        A = A.minimum(A.T)
        # изоляты после взаимного kNN: возвращаем им сильнейшее ребро (симметрично)
        deg = np.asarray((A > 0).sum(1)).ravel()
        iso = np.where(deg == 0)[0]
        if len(iso):
            fix = sparse.csr_matrix((w[iso, 0], (iso, idx[iso, 0])), shape=(n, n))
            A = A.maximum(fix).maximum(fix.T)
    else:
        A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def diffusion_profile(A: sparse.csr_matrix, steps: int = 2) -> np.ndarray:
    """Сетевое представление узла для KEFRiN: среднее матриц переходов T, T², … (плотное N×N)."""
    T = row_normalize(A)
    out = T.copy()
    cur = T.copy()
    for _ in range(steps - 1):
        cur = cur @ T
        out += cur
    return out / steps


def to_igraph(A: sparse.csr_matrix) -> ig.Graph:
    A = sparse.triu(A, k=1).tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())), directed=False)
    g.es["weight"] = A.data.tolist()
    g.vs["id"] = list(range(A.shape[0]))
    return g


def leiden(A: sparse.csr_matrix, resolution: float = 1.0, method: str = "modularity",
           seed: int = 42, n_iterations: int = -1) -> np.ndarray:
    g = to_igraph(A)
    if method == "modularity":
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                 resolution_parameter=resolution, seed=seed, n_iterations=n_iterations)
    elif method == "cpm":
        part = la.find_partition(g, la.CPMVertexPartition, weights="weight",
                                 resolution_parameter=resolution, seed=seed, n_iterations=n_iterations)
    else:
        raise ValueError(method)
    return np.asarray(part.membership)


def temporal_leiden(As: list[sparse.csr_matrix], interslice_weight: float = 0.5,
                    resolution: float = 1.0, seed: int = 42) -> np.ndarray:
    """Многослойный Leiden по временным срезам (Mucha et al. 2010). Возвращает матрицу (T, N)."""
    graphs = [to_igraph(A) for A in As]
    memberships, _ = la.find_partition_temporal(
        graphs, la.RBConfigurationVertexPartition, interslice_weight=interslice_weight,
        resolution_parameter=resolution, weights="weight", seed=seed, vertex_id_attr="id")
    return np.asarray(memberships)


def row_normalize(A: sparse.csr_matrix) -> np.ndarray:
    D = np.asarray(A.sum(axis=1)).ravel()
    D[D == 0] = 1.0
    return (A.multiply(1.0 / D[:, None])).toarray()
