"""DMoN — Deep Modularity Networks (Tsitsulin, Palowitch, Perozzi, Müller, JMLR 2023).

Компактная реализация на PyTorch по описанию статьи и официального кода (google-research/graph_embedding/dmon):
GCN-слой со skip-связью и SELU → мягкое назначение кластеров softmax(MLP) → потери
    L = −Tr(Cᵀ B C) / (2m)  +  λ · (‖Σ_i C_i‖_F · √k / n − 1),
где B = A − d dᵀ / (2m) — матрица модульности (веса рёбер учитываются), второе слагаемое — регуляризация от коллапса.
Жёсткое разбиение — argmax по строкам C. Используется в сравнении методов как представитель GNN-семейства
(оно есть в библиотеке Pattern, которую применяет жюри).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def dmon_cluster(X: np.ndarray, A: sparse.csr_matrix, k: int, hidden: int = 64, epochs: int = 500, lr: float = 1e-3,
                 collapse_weight: float = 1.0, dropout: float = 0.5, seed: int = 42, verbose: bool = False) -> np.ndarray:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    A = sparse.csr_matrix(A)
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    m2 = float(deg.sum())  # 2m
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    A_norm = sparse.diags(d_inv_sqrt) @ A @ sparse.diags(d_inv_sqrt)

    dev = torch.device("cpu")
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=dev)
    At = torch.tensor(A.toarray().astype(np.float32), device=dev)
    An = torch.tensor(A_norm.toarray().astype(np.float32), device=dev)
    dt = torch.tensor(deg.astype(np.float32), device=dev)

    gen = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.empty(X.shape[1], hidden).uniform_(-0.1, 0.1, generator=gen))
    W_skip = torch.nn.Parameter(torch.empty(X.shape[1], hidden).uniform_(-0.1, 0.1, generator=gen))
    b = torch.nn.Parameter(torch.zeros(hidden))
    Wc = torch.nn.Parameter(torch.empty(hidden, k).uniform_(-0.1, 0.1, generator=gen))
    bc = torch.nn.Parameter(torch.zeros(k))
    opt = torch.optim.Adam([W, W_skip, b, Wc, bc], lr=lr)
    drop = torch.nn.Dropout(dropout)

    def forward(train: bool):
        H = torch.selu(An @ (Xt @ W) + Xt @ W_skip + b)
        logits = (drop(H) if train else H) @ Wc + bc
        return torch.softmax(logits, dim=1)

    best = None
    for ep in range(epochs):
        opt.zero_grad()
        C = forward(True)
        # модульность: Tr(Cᵀ A C) − Tr(Cᵀ d dᵀ C)/(2m), нормированная на 2m
        CtAC = torch.trace(C.T @ (At @ C))
        Cd = C.T @ dt
        mod = (CtAC - (Cd @ Cd) / m2) / m2
        collapse = torch.norm(C.sum(dim=0)) / n * np.sqrt(k) - 1.0
        loss = -mod + collapse_weight * collapse
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            print(f"epoch {ep}: modularity {mod.item():.4f}, collapse {collapse.item():.4f}")
        if best is None or loss.item() < best[0]:
            best = (loss.item(), ep)
    with torch.no_grad():
        C = forward(False)
    return C.argmax(dim=1).cpu().numpy()
