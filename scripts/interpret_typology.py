"""Стек интерпретации для выбранной типологии (метки KEFRiN-T T×N).

Выход (outputs/interpretation/<tag>/): правило Миркина, пороговые правила IMM и их точность, суррогатное дерево,
интервальные описания, матрица переходов и стационарное распределение, переходы при условии соседей,
клубы конвергенции по типам, сопоставимые МО, внешняя валидация, сводка в markdown.

Запуск: .venv312/bin/python scripts/interpret_typology.py --labels outputs/kefrin_t/labels_affect_fe_K4_learned_windowlog.csv --tag K4
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mo_types import clustering, data, edge_rules as er, features, interpret  # noqa: E402

RAW_NAMES = {
    "share_Продовольствие": "доля продовольствия", "share_Здоровье": "доля здоровья",
    "share_Общественное питание": "доля общепита", "share_Маркетплейсы": "доля маркетплейсов",
    "share_Транспорт": "доля транспорта", "share_other": "доля прочего", "total_rub": "траты на жителя, ₽/мес",
    "population": "население", "wage": "зарплата, ₽", "market_access": "доступность рынков",
    "urban_share": "доля городского населения", "yoy_growth": "рост г/г", "seasonal_amplitude": "сезонная амплитуда",
    "emp_agri": "занятость: сельское хозяйство", "emp_extract": "занятость: добыча", "emp_industry": "занятость: промышленность",
    "emp_market_services": "занятость: рыночные услуги", "emp_public": "занятость: бюджетный сектор",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = data.load_config()
    out = data.ROOT / "outputs" / "interpretation" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    panel, attrs = data.load_processed(cfg)
    lab = pd.read_csv(data.ROOT / args.labels, index_col=0)
    lab.index = lab.index.astype(int)
    tids = lab.index.to_numpy()
    L = lab.to_numpy().T  # T×N
    K = int(L.max()) + 1
    months = list(lab.columns)
    modal = np.array([np.bincount(row, minlength=K).argmax() for row in L.T])
    attrs = attrs.reindex(tids)
    dates = sorted(panel.index.get_level_values("date").unique())
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    prof = features.static_profile(feat, dates=[d for d in dates if d.year == 2024]).reindex(tids)
    stat = features.static_attributes(attrs)

    # --- сырые (интерпретируемые) признаки ---------------------------------------------
    raw = pd.DataFrame(index=tids)
    for c in features.SHARE_COLS:
        raw[f"share_{c}"] = prof[f"share_{c}"]
    raw["total_rub"] = panel["total"].groupby(level="territory_id").mean().reindex(tids)
    raw["population"] = attrs["population"]
    raw["wage"] = attrs["wage"]
    raw["market_access"] = attrs["market_access"]
    raw["urban_share"] = attrs["urban_share"]
    raw["yoy_growth"] = np.exp(prof["yoy_log_mean"]) - 1
    raw["seasonal_amplitude"] = prof["seasonal_amplitude"]
    for g in ["emp_agri", "emp_extract", "emp_industry", "emp_market_services", "emp_public"]:
        raw[g] = stat[g]
    raw_named = raw.rename(columns=RAW_NAMES)

    # 1. правило Миркина
    mr = clustering.mirkin_rule(raw_named, modal)
    mr.to_csv(out / "mirkin.csv")

    # 2. IMM-дерево на стандартизованных признаках кластеризации (те же, что в KEFRiN-T: поведение + контекст)
    beh_cols = [c for c in prof.columns if c.startswith("clr_")] + ["rel_level", "level_volatility", "seasonal_amplitude", "yoy_log_mean", "marketplace_trend"]
    ctx_cols = [c for c in stat.columns if stat[c].notna().mean() > 0.9]
    Xz = pd.concat([features.zscore(prof[beh_cols]), features.zscore(stat[ctx_cols])], axis=1)
    centers = np.vstack([Xz.to_numpy()[modal == c].mean(0) for c in range(K)])
    tree = interpret.imm_tree(Xz.to_numpy(), centers, modal)
    pred = interpret.imm_predict(tree, Xz.to_numpy())
    imm_acc = float((pred == modal).mean())
    rules_z = interpret.imm_rules(tree, list(Xz.columns))
    # то же дерево на сырых (нестандартизованных) признаках — для читаемых порогов
    raw_for_tree = raw_named.fillna(raw_named.median())
    centers_raw = np.vstack([raw_for_tree.to_numpy()[modal == c].mean(0) for c in range(K)])
    tree_raw = interpret.imm_tree(raw_for_tree.to_numpy(), centers_raw, modal)
    pred_raw = interpret.imm_predict(tree_raw, raw_for_tree.to_numpy())
    imm_raw_acc = float((pred_raw == modal).mean())
    rules_raw = interpret.imm_rules(tree_raw, list(raw_for_tree.columns))
    sur, sur_acc = interpret.surrogate_tree(raw_for_tree.to_numpy(), modal, max_leaves=2 * K)
    sur_k, sur_k_acc = interpret.surrogate_tree(raw_for_tree.to_numpy(), modal, max_leaves=K)

    # 3. интервальные описания
    ip = interpret.interval_patterns(raw_named, modal)
    ip.to_csv(out / "interval_patterns.csv", index=False)

    # 4. динамика: Markov, пространственно обусловленные переходы
    mk = interpret.markov_transitions(L, K)
    pd.DataFrame(mk["P"]).to_csv(out / "transition_matrix.csv")
    conn = data.load_connection(cfg, "highway")
    D_road = er.road_matrix(conn, tids)
    np.fill_diagonal(D_road, np.inf)
    nbrs = np.argsort(D_road, axis=1)[:, :8]
    spm = interpret.spatial_conditioned_transitions(L, K, nbrs)
    stay_cond = {c: float(np.mean(np.diag(spm[c]))) for c in spm}
    # «притяжение к типу соседей»: P(перейти в тип c | соседи типа c, был не c)
    attract = {}
    for c in range(K):
        M = spm[c]
        mask = np.arange(K) != c
        attract[c] = float(M[mask, c].mean()) if M[mask].sum() > 0 else np.nan
    base_attract = {c: float(np.mean([mk["P"][a, c] for a in range(K) if a != c])) for c in range(K)}

    # 5. клубы конвергенции (уровни трат) внутри типов и в целом
    levels = panel["total"].unstack("date").reindex(tids).to_numpy().T  # T×N
    clubs = {"all": interpret.club_convergence_logt(levels)}
    for c in range(K):
        clubs[f"type_{c}"] = interpret.club_convergence_logt(levels[:, modal == c])

    # 6. сопоставимые МО
    pr = interpret.peers(Xz.to_numpy(), modal, tids, k=5)
    names = attrs["name"].astype(str)
    pr["name"] = pr["territory_id"].map(names)
    pr["peer_names"] = pr["peers"].map(lambda l: [names.get(i, str(i)) for i in l])
    pr.to_json(out / "peers.json", orient="records", force_ascii=False, indent=1)

    # 7. внешняя валидация
    ev = interpret.external_validation(modal, attrs, ["population", "wage", "market_access"])
    ev.update({"eta2_total_rub": interpret.external_validation(modal, raw, ["total_rub"])["eta2_total_rub"]})

    # --- сводка -------------------------------------------------------------------------
    sizes = np.bincount(modal, minlength=K)
    md = [f"# Интерпретация типологии {args.tag} (K={K})\n",
          "## Размеры и устойчивость типов\n",
          pd.DataFrame({"тип": range(K), "МО": sizes, "доля": (sizes / sizes.sum()).round(3),
                        "устойчивость (диагональ P)": np.diag(mk["P"]).round(3),
                        "ожидаемое пребывание, мес": mk["sojourn_months"].round(0),
                        "стационарная доля": mk["stationary"].round(3)}).to_string(index=False),
          "\n## Правило Миркина, d_kv = (c_kv − g_v)/g_v\n", mr.round(2).to_string(),
          f"\n## Пороговые правила IMM (стандартизованные признаки): точность {imm_acc:.3f}\n"]
    for c in range(K):
        md.append(f"- тип {c}: " + " и ".join(rules_z.get(c, [])))
    md.append(f"\n## Пороговые правила IMM на сырых признаках: точность {imm_raw_acc:.3f}\n")
    for c in range(K):
        md.append(f"- тип {c}: " + " и ".join(rules_raw.get(c, [])))
    md.append(f"\nСуррогатное дерево: {K} листьев — точность {sur_k_acc:.3f}; {2 * K} листьев — {sur_acc:.3f}\n")
    md.append("## Переходы при условии типа соседей (8 ближайших по дорогам)\n")
    md.append(pd.DataFrame({"тип соседей": range(K), "P(остаться в своём типе)": [round(stay_cond[c], 3) for c in range(K)],
                            "P(перейти в тип соседей)": [round(attract[c], 4) for c in range(K)],
                            "базовая P(перейти в этот тип)": [round(base_attract[c], 4) for c in range(K)]}).to_string(index=False))
    md.append("\n## Клубы конвергенции (log-t, Phillips & Sul 2007), уровни трат\n")
    md.append(pd.DataFrame(clubs).T.round(3).to_string())
    md.append("\n## Внешняя валидация\n" + json.dumps({k: round(v, 3) for k, v in ev.items()}, ensure_ascii=False))
    md.append("\n## Самые разделяющие интервалы (q10–q90) по типам\n")
    top = ip.sort_values("separation", ascending=False).groupby("type").head(3).sort_values(["type", "separation"], ascending=[True, False])
    md.append(top[["type", "feature", "lo", "hi", "separation"]].round(3).to_string(index=False))
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    json.dump({"imm_acc": imm_acc, "imm_raw_acc": imm_raw_acc, "surrogate_K": sur_k_acc, "surrogate_2K": sur_acc,
               "rules_raw": rules_raw, "rules_z": rules_z, "clubs": clubs, "external": ev,
               "stationary": mk["stationary"].tolist(), "sojourn": mk["sojourn_months"].tolist()},
              open(out / "summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
