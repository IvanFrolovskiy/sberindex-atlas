"""Признаки: композиция трат (доли, CLR), уровень, динамика; статические атрибуты."""
from __future__ import annotations

import numpy as np
import pandas as pd

SHARE_COLS = ["Продовольствие", "Здоровье", "Общественное питание", "Маркетплейсы", "Транспорт", "other"]


def monthly_features(panel: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Для каждой пары (territory_id, date): доли категорий, CLR-координаты, log уровня, прирост г/г.

    CLR: log(share_c + eps) − mean_c log(share_c + eps) (геометрия Айчисона).
    yoy_log: log(total_t / total_{t-12}); NaN в первый год.
    """
    f = pd.DataFrame(index=panel.index)
    tot = panel["total"].clip(lower=1.0)
    for c in SHARE_COLS:
        f[f"share_{c}"] = (panel[c].clip(lower=0) / tot)
    logs = np.log(f[[f"share_{c}" for c in SHARE_COLS]].to_numpy() + eps)
    clr = logs - logs.mean(axis=1, keepdims=True)
    for i, c in enumerate(SHARE_COLS):
        f[f"clr_{c}"] = clr[:, i]
    f["log_total"] = np.log(tot)
    # относительный уровень к медиане месяца (снимает общероссийскую динамику и инфляцию)
    med = f.groupby(level="date")["log_total"].transform("median")
    f["rel_level"] = f["log_total"] - med
    # прирост г/г
    lt = f["log_total"].unstack("date")
    yoy = lt - lt.shift(12, axis=1)
    f["yoy_log"] = yoy.stack(future_stack=True).reindex(f.index)
    return f


def static_profile(features: pd.DataFrame, dates=None) -> pd.DataFrame:
    """Средний профиль МО за выбранные даты (по умолчанию все): среднее CLR, уровень, амплитуда сезонности, рост."""
    f = features if dates is None else features.loc[features.index.get_level_values("date").isin(dates)]
    g = f.groupby(level="territory_id")
    prof = g[[c for c in f.columns if c.startswith("clr_") or c.startswith("share_")]].mean()
    prof["rel_level"] = g["rel_level"].mean()
    prof["level_volatility"] = g["rel_level"].std()
    prof["seasonal_amplitude"] = g["log_total"].agg(lambda s: s.max() - s.min())
    prof["yoy_log_mean"] = g["yoy_log"].mean()
    prof["marketplace_trend"] = g["share_Маркетплейсы"].agg(lambda s: s.iloc[-6:].mean() - s.iloc[:6].mean())
    return prof


EMP_GROUPS = {  # разделы ОКВЭД2 → 6 отраслевых групп занятости
    "emp_agri": ["A"], "emp_extract": ["B"], "emp_industry": ["C", "D", "E", "F"],
    "emp_market_services": ["G", "H", "I", "J", "K", "L", "M", "N", "S"],
    "emp_public": ["O", "P", "Q", "R"],
}


def static_attributes(attrs: pd.DataFrame, employment: str = "groups") -> pd.DataFrame:
    """Числовые статические атрибуты для кластеризации/валидации.

    employment: 'groups' — 5 агрегированных долей занятости; 'sections' — все разделы; 'none'.
    """
    a = pd.DataFrame(index=attrs.index)
    a["log_population"] = np.log(attrs["population"].clip(lower=100))
    a["log_market_access"] = np.log(attrs["market_access"].clip(lower=1))
    a["log_wage"] = np.log(attrs["wage"].clip(lower=1000))
    if "urban_share" in attrs:
        a["urban_share"] = attrs["urban_share"]
    emp_cols = [c for c in attrs.columns if c.startswith("emp_share_")]
    if employment == "sections":
        for c in emp_cols:
            a[c] = attrs[c]
    elif employment == "groups":
        for g, secs in EMP_GROUPS.items():
            cols = [f"emp_share_{s}" for s in secs if f"emp_share_{s}" in attrs.columns]
            a[g] = attrs[cols].sum(axis=1, min_count=1)
    a["is_urban_okrug"] = attrs["is_urban_okrug"]
    return a


def zscore(df: pd.DataFrame, robust: bool = False, clip: float | None = 4.0) -> pd.DataFrame:
    """Стандартизация по столбцам; clip ограничивает хвосты (|z| ≤ clip), чтобы редкие экстремумы
    (например, доля добычи 8,7σ) не доминировали в расстояниях и kNN-графах."""
    x = df.copy()
    for c in x.columns:
        col = x[c].astype(float)
        if robust:
            med = col.median()
            mad = (col - med).abs().median() * 1.4826
            x[c] = (col - med) / (mad if mad > 0 else 1.0)
        else:
            sd = col.std()
            x[c] = (col - col.mean()) / (sd if sd > 0 else 1.0)
    x = x.fillna(0.0)
    if clip is not None:
        x = x.clip(lower=-clip, upper=clip)
    return x
