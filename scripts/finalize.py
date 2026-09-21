"""Сборка итогового пакета типологии (outputs/final/): метки макро- и подтипов, профили, переходы,
паспорта МО, таблицы качества — единый JSON для лендинга и отчёта.

Запуск: .venv312/bin/python scripts/finalize.py
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from mo_types import clustering, data, features, interpret  # noqa: E402

OUT = data.ROOT / "outputs" / "final"
OUT.mkdir(parents=True, exist_ok=True)

RAW_NAMES = {
    "share_Продовольствие": "продовольствие", "share_Здоровье": "здоровье", "share_Общественное питание": "общепит",
    "share_Маркетплейсы": "маркетплейсы", "share_Транспорт": "транспорт", "share_other": "прочее",
    "total_rub": "траты на жителя", "population": "население", "wage": "зарплата", "market_access": "доступность рынков",
    "urban_share": "доля городского населения", "yoy_growth": "рост г/г", "seasonal_amplitude": "сезонность",
    "emp_agri": "занятость: сельское хозяйство", "emp_extract": "занятость: добыча", "emp_industry": "занятость: промышленность",
    "emp_market_services": "занятость: рыночные услуги", "emp_public": "занятость: бюджетный сектор",
}


def auto_name(row: pd.Series, level: str) -> str:
    """Имя типа по профилю Миркина (относительные отклонения). Правила простые и проверяемые."""
    d = row
    if d["занятость: добыча"] > 2.0 and d["доступность рынков"] < -0.2:
        return "Северные и ресурсные" if level == "macro" else "Северные ресурсные центры"
    if d["общепит"] > 0.8 and d["зарплата"] > 0.5:
        return "Столичные агломерации"
    if d["население"] > 0.8 and d["доля городского населения"] > 0.3:
        return "Крупные и средние города" if level == "macro" else "Региональные центры"
    if d["доступность рынков"] < -0.35 and d["транспорт"] > 0.05:
        return "Удалённые районы Востока и Сибири"
    if d["занятость: сельское хозяйство"] > 0.4 and d["траты на жителя"] < -0.15:
        return "Сельские и малые"
    if abs(d["траты на жителя"]) < 0.2 and d["занятость: промышленность"] > 0.2:
        return "Полугородские промышленные районы"
    return "Смешанный тип"


def load_labels(path):
    lab = pd.read_csv(data.ROOT / path, index_col=0)
    lab.index = lab.index.astype(int)
    return lab


def profiles(raw: pd.DataFrame, modal: np.ndarray, level: str, names_override: dict) -> tuple[pd.DataFrame, dict]:
    mr = clustering.mirkin_rule(raw, modal)
    med = raw.groupby(modal).median()
    med.index.name = "cluster"
    names = {}
    for c in mr.index:
        names[int(c)] = names_override.get(int(c)) or names_override.get(str(c)) or auto_name(mr.loc[c], level)
    # уникальность имён
    seen = {}
    for c, n in list(names.items()):
        if n in seen:
            names[c] = f"{n} ({seen[n] + 1})"
            seen[n] += 1
        else:
            seen[n] = 1
    mr = mr.copy()
    mr.insert(0, "name", [names[int(c)] for c in mr.index])
    med.insert(0, "name", [names[int(c)] for c in med.index])
    return mr, med, names


def main():
    cfg = data.load_config()
    tcfg = yaml.safe_load(open(data.ROOT / "configs" / "types.yaml", encoding="utf-8"))
    panel, attrs = data.load_processed(cfg)
    macro = load_labels(tcfg["labels"]["macro"])
    sub = load_labels(tcfg["labels"]["sub"])
    tids = macro.index.to_numpy()
    months = list(macro.columns)
    Lm, Ls = macro.to_numpy().T, sub.to_numpy().T
    Km, Ks = int(Lm.max()) + 1, int(Ls.max()) + 1
    modal_m = np.array([np.bincount(r, minlength=Km).argmax() for r in Lm.T])
    modal_s = np.array([np.bincount(r, minlength=Ks).argmax() for r in Ls.T])
    attrs = attrs.reindex(tids)
    dates = sorted(panel.index.get_level_values("date").unique())
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    prof = features.static_profile(feat, dates=[d for d in dates if d.year == 2024]).reindex(tids)
    stat = features.static_attributes(attrs)
    raw = pd.DataFrame(index=tids)
    for c in features.SHARE_COLS:
        raw[f"share_{c}"] = prof[f"share_{c}"]
    raw["total_rub"] = panel["total"].groupby(level="territory_id").mean().reindex(tids)
    for c in ["population", "wage", "market_access", "urban_share"]:
        raw[c] = attrs[c]
    raw["yoy_growth"] = np.exp(prof["yoy_log_mean"]) - 1
    raw["seasonal_amplitude"] = prof["seasonal_amplitude"]
    for g in ["emp_agri", "emp_extract", "emp_industry", "emp_market_services", "emp_public"]:
        raw[g] = stat[g]
    raw_named = raw.rename(columns=RAW_NAMES)

    mr_m, med_m, names_m = profiles(raw_named, modal_m, "macro", tcfg["names"].get("macro") or {})
    mr_s, med_s, names_s = profiles(raw_named, modal_s, "sub", tcfg["names"].get("sub") or {})
    mr_m.to_csv(OUT / "macro_profiles_mirkin.csv"); med_m.to_csv(OUT / "macro_profiles_median.csv")
    mr_s.to_csv(OUT / "sub_profiles_mirkin.csv"); med_s.to_csv(OUT / "sub_profiles_median.csv")

    # переходы и динамика
    mk_m = interpret.markov_transitions(Lm, Km)
    mk_s = interpret.markov_transitions(Ls, Ks)
    sizes_m = pd.DataFrame({m: np.bincount(Lm[i], minlength=Km) for i, m in enumerate(months)}).T
    sizes_s = pd.DataFrame({m: np.bincount(Ls[i], minlength=Ks) for i, m in enumerate(months)}).T
    switch_m = [float((Lm[t] != Lm[t - 1]).mean()) for t in range(1, len(months))]
    switch_s = [float((Ls[t] != Ls[t - 1]).mean()) for t in range(1, len(months))]
    # кросс-таблица макро × под
    cross = pd.crosstab(pd.Series(modal_m, name="macro"), pd.Series(modal_s, name="sub"))
    cross.to_csv(OUT / "macro_sub_crosstab.csv")

    # паспорта МО
    beh_cols = [c for c in prof.columns if c.startswith("clr_")] + ["rel_level", "level_volatility", "seasonal_amplitude", "yoy_log_mean", "marketplace_trend"]
    ctx_cols = [c for c in stat.columns if stat[c].notna().mean() > 0.9]
    Xz = pd.concat([features.zscore(prof[beh_cols]), features.zscore(stat[ctx_cols])], axis=1).to_numpy()
    peers_s = interpret.peers(Xz, modal_s, tids, k=5).set_index("territory_id")
    rel = feat["rel_level"].unstack("date").reindex(tids)
    tot = panel["total"].unstack("date").reindex(tids)
    cent = pd.read_csv(data.ROOT / "data/processed/mo_centroids.csv").set_index("territory_id").reindex(tids)
    n_sw_m = (Lm[1:] != Lm[:-1]).sum(axis=0)
    n_sw_s = (Ls[1:] != Ls[:-1]).sum(axis=0)
    mo = []
    for i, tid in enumerate(tids):
        a = attrs.loc[tid]
        mo.append({
            "id": int(tid), "name": str(a["name"]), "short": str(a["name_short"]), "region": str(a["region_name"]),
            "mo_type": str(a["mo_type"]), "lat": float(cent.loc[tid, "lat"]) if pd.notna(cent.loc[tid, "lat"]) else None,
            "lon": float(cent.loc[tid, "lon"]) if pd.notna(cent.loc[tid, "lon"]) else None,
            "macro": int(modal_m[i]), "sub": int(modal_s[i]),
            "macro_seq": [int(x) for x in Lm[:, i]], "sub_seq": [int(x) for x in Ls[:, i]],
            "switches_macro": int(n_sw_m[i]), "switches_sub": int(n_sw_s[i]),
            "population": None if pd.isna(a["population"]) else float(a["population"]),
            "wage": None if pd.isna(a["wage"]) else float(a["wage"]),
            "market_access": None if pd.isna(a["market_access"]) else float(a["market_access"]),
            "urban_share": None if pd.isna(a["urban_share"]) else float(a["urban_share"]),
            "total_2024": float(tot.loc[tid].iloc[12:].mean()), "total_2023": float(tot.loc[tid].iloc[:12].mean()),
            "shares": {RAW_NAMES[f"share_{c}"]: round(float(prof.loc[tid, f"share_{c}"]), 4) for c in features.SHARE_COLS},
            "rel_level": [round(float(x), 3) for x in rel.loc[tid].to_numpy()],
            "total_series": [float(x) for x in tot.loc[tid].to_numpy()],
            "emp": {RAW_NAMES[g]: (None if pd.isna(stat.loc[tid, g]) else round(float(stat.loc[tid, g]), 3)) for g in ["emp_agri", "emp_extract", "emp_industry", "emp_market_services", "emp_public"]},
            "peers": [int(p) for p in peers_s.loc[tid, "peers"]],
            "mirkin": {col: (None if pd.isna(v) else round(float(v), 3)) for col, v in ((raw_named.loc[tid] - raw_named.mean()) / raw_named.mean()).items()},
        })
    pd.DataFrame([{k: v for k, v in m.items() if not isinstance(v, (list, dict))} for m in mo]).to_csv(OUT / "mo_table.csv", index=False)

    # качество: временная сводка, зоопарк, правила ребра, синтетика, устойчивость
    def read(p):
        p = data.ROOT / p
        return pd.read_csv(p).to_dict(orient="records") if p.exists() else []
    quality = {
        "temporal_windowlog": read("outputs/kefrin_t/summary_learned_windowlog.csv"),
        "temporal_knn": read("outputs/kefrin_t/summary_knn_minsize.csv"),
        "zoo": read("outputs/method_zoo/zoo_learned.csv"),
        "edge_rules": read("outputs/edge_rules/edge_rules_comparison.csv"),
        "edge_agreement": pd.read_csv(data.ROOT / "outputs/edge_rules/rule_agreement_jaccard.csv", index_col=0).round(3).to_dict() if (data.ROOT / "outputs/edge_rules/rule_agreement_jaccard.csv").exists() else {},
        "stability_knn": read("outputs/stability/stability_knn.csv"),
        "stability_learned": read("outputs/stability/stability_learned.csv"),
        "synthetic": read("outputs/synthetic/synthetic_results.csv"),
        "alphas": read("outputs/kefrin_t/alphas_learned_windowlog.csv"),
        "graph_stats_windowlog": read("outputs/learned_graph/windowlog/stats.csv"),
    }
    bundle = {
        "months": months, "K_macro": Km, "K_sub": Ks,
        "names_macro": {int(k): v for k, v in names_m.items()}, "names_sub": {int(k): v for k, v in names_s.items()},
        "profiles_macro": mr_m.reset_index().to_dict(orient="records"), "profiles_sub": mr_s.reset_index().to_dict(orient="records"),
        "medians_macro": med_m.reset_index().to_dict(orient="records"), "medians_sub": med_s.reset_index().to_dict(orient="records"),
        "transitions_macro": mk_m["P"].round(4).tolist(), "transitions_sub": mk_s["P"].round(4).tolist(),
        "counts_macro": mk_m["counts"].tolist(), "counts_sub": mk_s["counts"].tolist(),
        "stationary_macro": mk_m["stationary"].round(4).tolist(), "stationary_sub": mk_s["stationary"].round(4).tolist(),
        "sojourn_macro": mk_m["sojourn_months"].round(1).tolist(), "sojourn_sub": mk_s["sojourn_months"].round(1).tolist(),
        "sizes_macro": sizes_m.to_dict(orient="split"), "sizes_sub": sizes_s.to_dict(orient="split"),
        "switch_macro": switch_m, "switch_sub": switch_s,
        "crosstab": cross.to_dict(orient="split"),
        "grand_means": {k: (None if pd.isna(v) else float(v)) for k, v in raw_named.mean().items()},
        "mo": mo, "quality": quality,
    }
    json.dump(bundle, open(OUT / "bundle.json", "w"), ensure_ascii=False, default=float)
    tcfg["names"]["macro"] = {int(k): v for k, v in names_m.items()}
    tcfg["names"]["sub"] = {int(k): v for k, v in names_s.items()}
    yaml.safe_dump(tcfg, open(data.ROOT / "configs" / "types.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print("макротипы:", names_m)
    print("подтипы:", names_s)
    print("кросс-таблица макро×под:\n", cross.to_string())
    print(f"bundle.json: {len(mo)} МО, {(OUT / 'bundle.json').stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
