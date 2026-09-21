"""KEFRiN-T: эволюционная кластеризация атрибутированных временных сетей.

Идея: на каждом срезе t кластеризуется не сырой срез (Y_t, P_t), а сглаженный:
    Ỹ_t = α_t Ỹ_{t−1} + (1 − α_t) Y_t,   P̃_t = β_t P̃_{t−1} + (1 − β_t) P_t,
где веса α_t, β_t оцениваются по данным shrinkage-оценкой AFFECT (Xu, Kliger, Hero, DMKD 2014),
а «истинные» значения в оценке берутся из модели восстановления данных KEFRiN (Shalileh & Mirkin 2022):
E[y_iv] = c_{k(i)v},  E[p_ij] = λ_{k(i)j}.  Сама кластеризация среза — KEFRiN с тёплым стартом от S_{t−1}
и опциональным штрафом за смену кластера (Chakrabarti et al. 2006).

Два варианта оценки веса истории:
- 'affect'    — блочная модель AFFECT: шум = отклонение от среднего кластера;
- 'affect_fe' — блочная модель с фиксированными эффектами узлов: E[y_iv] = c_{k(i)v} + u_iv, где u_iv —
  устойчивое отклонение МО от своего типа (среднее прошлых остатков); шумом считается только ε = r − u.
  Для панели территорий это корректнее: внутрикластерный разброс в основном устойчив, а не случаен.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from .clustering import kefrin


def align_labels(prev: np.ndarray, cur: np.ndarray, k: int) -> np.ndarray:
    """Перенумеровать cur так, чтобы максимизировать совпадение с prev (венгерский алгоритм)."""
    M = np.zeros((k, k), dtype=float)
    np.add.at(M, (cur, prev), 1.0)
    rows, cols = linear_sum_assignment(-M)
    mapping = np.arange(k)
    mapping[rows] = cols
    return mapping[cur]


def align_labels_general(truth: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Выравнивание при произвольном числе кластеров в cur: лишние кластеры получают метку −1."""
    ut, ic = np.unique(cur, return_inverse=True)
    kt = int(truth.max()) + 1
    M = np.zeros((len(ut), kt))
    np.add.at(M, (ic, truth), 1.0)
    rows, cols = linear_sum_assignment(-M)
    mapping = np.full(len(ut), -1)
    mapping[rows] = cols
    return mapping[ic]


def cluster_means(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    C = np.zeros((k, X.shape[1]))
    for c in range(k):
        m = labels == c
        if m.any():
            C[c] = X[m].mean(axis=0)
    return C


def _robust_var(R: np.ndarray, winsor: float = 3.0) -> np.ndarray:
    """Робастная дисперсия по столбцам: дисперсия остатков, винзоризованных на ±winsor·1,4826·MAD.

    Меньшинство больших остатков (мигранты) её не раздувает, но умеренно тяжёлые хвосты учитываются
    (чистый MAD занижает шум на негауссовых данных и обнуляется на разреженных матрицах)."""
    med = np.median(R, axis=0)
    mad = np.median(np.abs(R - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale > 0, scale, R.std(axis=0, ddof=1) + 1e-12)
    W = np.clip(R, med - winsor * scale, med + winsor * scale)
    return W.var(axis=0, ddof=1) + 1e-12


def shrinkage_alpha(prev_smoothed: np.ndarray, current: np.ndarray, labels: np.ndarray, k: int,
                    robust: bool = True) -> float:
    """AFFECT под блочной моделью: α* = Σ var / Σ [(x̃^{t−1} − E x)² + var], E x — среднее кластера."""
    num = den = 0.0
    gvar = current.var(axis=0, ddof=1) + 1e-12
    for c in range(k):
        m = labels == c
        n = int(m.sum())
        if n == 0:
            continue
        mean = current[m].mean(axis=0)
        if n > 2:
            var = _robust_var(current[m] - mean) if robust else current[m].var(axis=0, ddof=1) + 1e-12
        else:
            var = gvar
        num += n * var.sum()
        den += ((prev_smoothed[m] - mean) ** 2).sum() + n * var.sum()
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0


def shrinkage_alpha_fe(prev_smoothed: np.ndarray, current: np.ndarray, labels: np.ndarray, k: int,
                       U: np.ndarray, n_past: int, robust: bool = True) -> tuple[float, np.ndarray]:
    """AFFECT с фиксированными эффектами узлов: E x_iv = c_{k(i)v} + u_iv; шум ε = x − E x.

    var(ε) — робастная (MAD) оценка по остаткам текущего среза относительно (c + u) с поправкой на то,
    что u — среднее n_past прошлых остатков (дисперсия разности равна σ²(1 + 1/n_past)).
    Возвращает α и стандартизованный квадрат остатка каждого узла z_i = mean_v ε_iv²/var_v (тест инновации).
    """
    C = cluster_means(current, labels, k)
    expected = C[labels] + U
    eps = current - expected
    corr = 1.0 + 1.0 / max(n_past, 1)
    var = (_robust_var(eps) if robust else eps.var(axis=0, ddof=1) + 1e-12) / corr
    z = ((eps ** 2) / (var * corr)).mean(axis=1)
    num = current.shape[0] * var.sum()
    den = ((prev_smoothed - expected) ** 2).sum() + num
    alpha = float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0
    return alpha, z


def _min_size_ok(labels: np.ndarray, k: int, min_size: int) -> bool:
    return np.bincount(labels, minlength=k).min() >= min_size


def init_partition(Ys: list[np.ndarray], Ps: list[np.ndarray], k: int, window: int, rho: float, xi,
                   distance: str, seed: int, n_init: int, min_size: int, switch_penalty: float = 0.0):
    """Стартовое разбиение: KEFRiN на среднем первых `window` срезов, из n_init запусков —
    лучший по критерию среди тех, где нет карликовых кластеров (< min_size)."""
    Ybar = np.mean(Ys[:window], axis=0)
    Pbar = np.mean(Ps[:window], axis=0)
    best = None
    for r in range(n_init):
        lab, C, L, F, xi_eff = kefrin(Ybar, Pbar, k, rho=rho, xi=xi, distance=distance, seed=seed + r, n_init=1,
                                      min_size=min_size)
        ok = _min_size_ok(lab, k, min_size)
        key = (ok, -F)
        if best is None or key > best[0]:
            best = (key, lab, xi_eff)
    return best[1], best[2]


def kefrin_t(Ys: list[np.ndarray], Ps: list[np.ndarray], k: int, rho: float = 1.0, xi="auto",
             distance: str = "euclidean", smoothing: str = "affect_fe", alpha_fixed: float = 0.5,
             switch_penalty: float = 0.0, affect_iters: int = 4, seed: int = 42, n_init: int = 10,
             init_window: int = 3, min_size_frac: float = 0.01, reset_z: float | None = 4.0,
             fe_forget: float = 1.0, alpha_max: float = 0.95, verbose: bool = False,
             init_labels: np.ndarray | None = None) -> dict:
    """Возвращает labels (T×N), alphas, betas (веса истории по срезам), centers по срезам, xi, resets.

    reset_z — порог теста инновации: узел с z_i > reset_z (стандартизованный квадрат остатка относительно
    «тип + свой эффект») в текущем месяце получает α_i = 0 (сглаживание сбрасывается к сырому значению),
    а его накопленный эффект обнуляется — так настоящая смена типа проходит без задержки.
    fe_forget — забывание эффектов узла (1 = среднее всех прошлых остатков, <1 — экспоненциальное).
    alpha_max — верхняя граница веса истории (гарантия отклика на медленные изменения).
    init_labels — готовое стартовое разбиение (например, модальные типы первого прохода: «второй проход»
    убирает переходный процесс первых месяцев, когда типология ещё «усаживается»)."""
    T, N = len(Ys), Ys[0].shape[0]
    min_size = max(2, int(min_size_frac * N))
    labels = np.zeros((T, N), dtype=int)
    alphas, betas, objectives, centers = [], [], [], []
    # стартовое разбиение по первым срезам
    init_lab, xi_eff = init_partition(Ys, Ps, k, init_window, rho, xi, distance, seed, n_init, min_size)
    if init_labels is not None:
        init_lab = np.asarray(init_labels, dtype=int)
    Ysm = Psm = None
    U_y = np.zeros_like(Ys[0])  # фиксированные эффекты (среднее прошлых остатков)
    U_p = np.zeros_like(Ps[0])
    n_y = np.zeros(N)  # число накопленных остатков по узлу (для среднего и поправки дисперсии)
    a = b = 0.0
    resets = np.zeros((T, N), dtype=bool)
    z_last = np.zeros(N)
    for t in range(T):
        Y, P = Ys[t], Ps[t]
        prev = init_lab if t == 0 else labels[t - 1]
        if t == 0 or smoothing == "none":
            Ycur, Pcur = Y, P
            a = b = 0.0
            lab, C, L, F, _ = kefrin(Ycur, Pcur, k, rho=rho, xi=xi_eff, distance=distance, seed=seed,
                                     n_init=1, init_labels=prev, switch_penalty=switch_penalty, min_size=min_size)
        elif smoothing == "fixed":
            a = b = alpha_fixed
            Ycur = a * Ysm + (1 - a) * Y
            Pcur = b * Psm + (1 - b) * P
            lab, C, L, F, _ = kefrin(Ycur, Pcur, k, rho=rho, xi=xi_eff, distance=distance, seed=seed,
                                     n_init=1, init_labels=prev, switch_penalty=switch_penalty, min_size=min_size)
        elif smoothing in ("affect", "affect_fe"):
            a, b = (alphas[-1], betas[-1]) if len(alphas) > 1 else (0.5, 0.5)
            lab = prev
            for it in range(affect_iters):
                Ycur = a * Ysm + (1 - a) * Y
                Pcur = b * Psm + (1 - b) * P
                lab, C, L, F, _ = kefrin(Ycur, Pcur, k, rho=rho, xi=xi_eff, distance=distance, seed=seed,
                                         n_init=1, init_labels=lab, switch_penalty=switch_penalty, min_size=min_size)
                if smoothing == "affect":
                    a_new = shrinkage_alpha(Ysm, Y, lab, k)
                    b_new = shrinkage_alpha(Psm, P, lab, k)
                    z = np.zeros(N)
                else:
                    a_new, z = shrinkage_alpha_fe(Ysm, Y, lab, k, U_y, max(int(n_y.mean()), 1), robust=True)
                    b_new, _ = shrinkage_alpha_fe(Psm, P, lab, k, U_p, max(int(n_y.mean()), 1), robust=False)
                a_new, b_new = min(a_new, alpha_max), min(b_new, alpha_max)
                converged = abs(a_new - a) < 0.01 and abs(b_new - b) < 0.01
                a, b = a_new, b_new
                if converged:
                    break
            # тест инновации: узлы с большим остатком получают α_i = 0 (сброс сглаживания)
            a_i = np.full(N, a)
            if smoothing == "affect_fe" and reset_z is not None:
                hit = z > reset_z
                a_i[hit] = 0.0
                resets[t] = hit
                z_last = z
            Ycur = a_i[:, None] * Ysm + (1 - a_i[:, None]) * Y
            Pcur = a_i[:, None] * 0 + b * Psm + (1 - b) * P if reset_z is None else np.where(a_i[:, None] > 0, b * Psm + (1 - b) * P, P)
            lab, C, L, F, _ = kefrin(Ycur, Pcur, k, rho=rho, xi=xi_eff, distance=distance, seed=seed,
                                     n_init=1, init_labels=lab, switch_penalty=switch_penalty, min_size=min_size)
        else:
            raise ValueError(smoothing)
        lab = align_labels(prev, lab, k)
        # обновление эффектов узла: среднее (или забывающее) остатков относительно среднего своего типа;
        # у узлов со сбросом или сменой типа накопление начинается заново
        r_y = Y - cluster_means(Y, lab, k)[lab]
        r_p = P - cluster_means(P, lab, k)[lab]
        restart = resets[t] | (lab != prev)
        n_y = np.where(restart, 0.0, n_y)
        if fe_forget >= 1.0:
            w_old = n_y / (n_y + 1.0)
        else:
            w_old = np.where(n_y > 0, fe_forget, 0.0)
        U_y = w_old[:, None] * U_y + (1 - w_old[:, None]) * r_y
        U_p = w_old[:, None] * U_p + (1 - w_old[:, None]) * r_p
        n_y = n_y + 1.0
        labels[t] = lab
        alphas.append(a)
        betas.append(b)
        objectives.append(F)
        centers.append(cluster_means(Ycur, lab, k))
        Ysm, Psm = Ycur, Pcur
        if verbose:
            print(f"  t={t:2d} α={a:.2f} β={b:.2f} F={F:.1f} размеры={np.bincount(lab, minlength=k).tolist()}", flush=True)
    return {"labels": labels, "alphas": np.array(alphas), "betas": np.array(betas),
            "objectives": np.array(objectives), "centers": centers, "xi": xi_eff, "resets": resets}


def stability(labels: np.ndarray, event_threshold: float = 0.10) -> dict:
    """AMI/ARI соседних срезов, доля объектов, сменивших кластер (switching rate), месяцы-«реорганизации»."""
    T = labels.shape[0]
    amis = [adjusted_mutual_info_score(labels[t - 1], labels[t]) for t in range(1, T)]
    aris = [adjusted_rand_score(labels[t - 1], labels[t]) for t in range(1, T)]
    switch = [float((labels[t - 1] != labels[t]).mean()) for t in range(1, T)]
    n_changes = (labels[1:] != labels[:-1]).sum(axis=0)
    events = [t for t, s in enumerate(switch, start=1) if s > event_threshold]
    return {"ami_median": float(np.median(amis)), "ami_min": float(np.min(amis)),
            "ari_median": float(np.median(aris)), "switch_median": float(np.median(switch)),
            "switch_max": float(np.max(switch)), "share_never_switch": float((n_changes == 0).mean()),
            "share_switch_gt3": float((n_changes > 3).mean()), "n_events": len(events),
            "min_cluster_size": int(min(np.bincount(labels[t]).min() for t in range(T)))}
