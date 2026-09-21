"""Тесты метрик и алгоритмов: сверка AVU/AVI/ANUI/модульности с реализацией библиотеки Pattern (жюри),
MQ на игрушечном графе, сходимость KEFRiN, дерево IMM."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mo_types import clustering, interpret, metrics  # noqa: E402


def pattern_adjacency_metrics(adjacency_matrix, clustering_labels):
    """Дословный порт Pattern/metrics/clustering_metrics.py::AdjacencyClusteringMetrics.get_metric."""
    clusters = np.unique(clustering_labels)
    k = len(clusters)
    mask_list = [clustering_labels == c for c in clusters]
    S = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            S[i, j] = adjacency_matrix[np.ix_(mask_list[i], mask_list[j])].sum()
    sum_edges = adjacency_matrix.sum() / 2
    degrees = adjacency_matrix.sum(axis=1)
    sum_degrees = [degrees[m].sum() for m in mask_list]
    community_sizes = [m.sum() for m in mask_list]
    avu = 0.0
    for i in range(k):
        sum_i_out = S[i].sum() - S[i, i]
        sum_u_i = 0.0
        for j in range(k):
            if j == i:
                continue
            sum_j_in = S[:, j].sum() - S[j, j]
            numerator = S[i, j]
            denominator = sum_i_out + sum_j_in - numerator
            sum_u_i += numerator / denominator if denominator != 0 else 0.0
        avu += sum_u_i / k
    avi = 0.0
    for i in range(k):
        total = S[i].sum()
        avi += (S[i, i] / total if total != 0 else 0.0) / k
    anui = 1.0 / (avu + 1.0 / avi) if avi != 0 else 0.0
    modularity = 0.0
    density_modularity = 0.0
    if sum_edges != 0:
        for i in range(k):
            sum_internal = S[i, i] / 2
            sum_d = sum_degrees[i]
            modularity += (sum_internal / sum_edges) - (sum_d / (2 * sum_edges)) ** 2
            size = community_sizes[i]
            density_modularity += (sum_internal - (sum_d ** 2) / (4 * sum_edges)) / size if size != 0 else 0.0
    return {"ANUI": anui, "AVU": avu, "AVI": avi, "modularity": modularity, "density_modularity": density_modularity}


def toy_graph(seed=0, n=60, k=3):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(k), n // k)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = 0.5 if labels[i] == labels[j] else 0.05
            if rng.random() < p:
                w = rng.uniform(0.5, 1.5)
                A[i, j] = A[j, i] = w
    return A, labels


def test_graph_metrics_match_pattern():
    A, labels = toy_graph()
    ours = metrics.graph_metrics(sparse.csr_matrix(A), labels)
    ref = pattern_adjacency_metrics(A, labels)
    for key in ["AVU", "AVI", "ANUI", "modularity", "density_modularity"]:
        assert ours[key] == pytest.approx(ref[key], rel=1e-9), key


def test_mq_perfect_partition_equals_k():
    A, labels = toy_graph(seed=1)
    # без межкластерных рёбер MQ = K (каждый CF_k = 1)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if labels[i] != labels[j]:
                A[i, j] = 0.0
    m = metrics.graph_metrics(sparse.csr_matrix(A), labels)
    assert m["MQ"] == pytest.approx(3.0)
    assert m["AVI"] == pytest.approx(1.0)


def test_kefrin_recovers_planted_clusters():
    rng = np.random.default_rng(0)
    n, k = 150, 3
    labels = np.repeat(np.arange(k), n // k)
    Y = rng.normal(size=(n, 4)) + labels[:, None] * 4.0
    A, _ = toy_graph(seed=2, n=n, k=k)
    P = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
    lab, C, L, F, xi = clustering.kefrin(Y, P, k, xi="auto", seed=0, n_init=3)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(labels, lab) > 0.95
    assert xi > 0


def test_imm_tree_separates_centers():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(50, 2)) + [0, 0], rng.normal(size=(50, 2)) + [6, 0], rng.normal(size=(50, 2)) + [0, 6]])
    labels = np.repeat([0, 1, 2], 50)
    centers = np.vstack([X[labels == c].mean(0) for c in range(3)])
    tree = interpret.imm_tree(X, centers, labels)
    pred = interpret.imm_predict(tree, X)
    assert (pred == labels).mean() > 0.95
    rules = interpret.imm_rules(tree, ["x", "y"])
    assert set(rules) == {0, 1, 2}
