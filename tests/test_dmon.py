"""DMoN восстанавливает посаженные сообщества на стохастической блочной модели с информативными признаками."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")
from mo_types.dmon import dmon_cluster  # noqa: E402


def test_dmon_recovers_planted_blocks():
    rng = np.random.default_rng(0)
    n, k = 300, 3
    truth = np.arange(n) % k
    same = truth[:, None] == truth[None, :]
    A = (rng.random((n, n)) < np.where(same, 0.12, 0.01)).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    X = np.eye(k)[truth] * 2 + rng.normal(size=(n, k)) * 0.5
    lab = dmon_cluster(X, sparse.csr_matrix(A), k, epochs=300, seed=1)
    assert len(np.unique(lab)) == k
    assert adjusted_rand_score(truth, lab) > 0.8
