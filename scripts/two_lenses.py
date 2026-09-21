"""Две линзы: экономическая типология (потребление + контекст Росстата) и поведенческая (только СберИндекс).

Что делает:
- поведенческая типология: метки KEFRiN-T только по признакам потребления (CLR-структура + относительный уровень),
  K из configs/types.yaml (labels.beh); модальный тип, смены, доля без смен;
- кросс-таблицы экономическая × поведенческая (макро и подтипы), ARI/AMI между линзами;
- «что даёт СберИндекс»: статическая KEFRiN по полным признакам / только потребление / только контекст → ARI с
  экономической типологией (сколько типологии объясняет каждая половина признаков);
- профили поведенческих типов (правило Миркина и медианы в сырых единицах), имена из конфигурации;
- устойчивость: из outputs/kefrin_t/summary_learned_beh_windowlog.csv и outputs/stability/stability_learned_beh.csv;
- рисунок для отчёта: карта поведенческих типов, кросс-таблица, профили.

Запуск: .venv312/bin/python scripts/two_lenses.py
Выход: outputs/two_lenses/{mo_beh.csv, crosstab_macro.csv, crosstab_sub.csv, beh_profiles_mirkin.csv,
        beh_profiles_median.csv, summary.json}, report/figures/fig5_two_lenses.png
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari  # noqa: E402

from finalize import RAW_NAMES  # noqa: E402
from mo_types import clustering, data, features, graphs, interpret  # noqa: E402

OUT = data.ROOT / "outputs" / "two_lenses"
FIG = data.ROOT / "report" / "figures"
BEH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4"]


def load_labels(path):
    lab = pd.read_csv(data.ROOT / path, index_col=0)
    lab.index = lab.index.astype(int)
    return lab


def raw_table(panel, attrs, tids):
    feat = features.monthly_features(panel)
    dates = sorted(panel.index.get_level_values("date").unique())
    prof = features.static_profile(feat, dates=[d for d in dates if d.year == 2024]).reindex(tids)
    stat = features.static_attributes(attrs.reindex(tids))
    raw = pd.DataFrame(index=tids)
    for c in ["Продовольствие", "Здоровье", "Общественное питание", "Маркетплейсы", "Транспорт", "other"]:
        raw[f"share_{c}"] = prof[f"share_{c}"]
    raw["total_rub"] = panel["total"].groupby(level="territory_id").mean().reindex(tids)
    for c in ["population", "wage", "market_access", "urban_share"]:
        raw[c] = attrs[c].reindex(tids)
    raw["yoy_growth"] = np.exp(prof["yoy_log_mean"]) - 1
    raw["seasonal_amplitude"] = prof["seasonal_amplitude"]
    for g in ["emp_agri", "emp_extract", "emp_industry", "emp_market_services", "emp_public"]:
        raw[g] = stat[g]
    return raw.rename(columns=RAW_NAMES), prof, stat


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = data.load_config()
    types = yaml.safe_load(open(data.ROOT / "configs" / "types.yaml", encoding="utf-8"))
    panel, attrs = data.load_processed(cfg)
    E = load_labels(types["labels"]["macro"])
    Sb = load_labels(types["labels"]["sub"]).reindex(E.index)
    B = load_labels(types["labels"]["beh"]).reindex(E.index)
    tids = E.index.to_numpy()
    me = E.mode(axis=1)[0].astype(int).to_numpy()
    ms = Sb.mode(axis=1)[0].astype(int).to_numpy()
    mb = B.mode(axis=1)[0].astype(int).to_numpy()
    K = int(mb.max()) + 1
    names_e = {int(k): v for k, v in types["names"]["macro"].items()}
    names_s = {int(k): v for k, v in types["names"]["sub"].items()}
    names_b = {int(k): v for k, v in (types["names"].get("beh") or {}).items()} or {i: f"поведенческий тип {i}" for i in range(K)}

    # --- таблица МО ---------------------------------------------------------------------------
    switches = (B.to_numpy()[:, 1:] != B.to_numpy()[:, :-1]).sum(axis=1)
    mo = pd.DataFrame({"id": tids, "beh": mb, "beh_name": [names_b[i] for i in mb], "switches_beh": switches,
                       "beh_seq": ["".join(str(int(x)) for x in row) for row in B.to_numpy()],
                       "macro": me, "sub": ms}).set_index("id")
    mo.to_csv(OUT / "mo_beh.csv")

    # --- кросс-таблицы и согласие -------------------------------------------------------------
    ct = pd.crosstab(pd.Series(me, name="macro").map(names_e), pd.Series(mb, name="beh").map(names_b))
    ct = ct.reindex(index=[names_e[i] for i in range(len(names_e))], columns=[names_b[i] for i in range(K)])
    ct.to_csv(OUT / "crosstab_macro.csv")
    cts = pd.crosstab(pd.Series(ms, name="sub").map(names_s), pd.Series(mb, name="beh").map(names_b))
    cts.to_csv(OUT / "crosstab_sub.csv")
    row_purity = (ct.max(axis=1) / ct.sum(axis=1)).round(3).to_dict()
    col_purity = (ct.max(axis=0) / ct.sum(axis=0)).round(3).to_dict()

    # --- что даёт СберИндекс: статические разбиения по половинам признаков ----------------------
    raw, prof, stat = raw_table(panel, attrs, tids)
    beh_cols = [c for c in prof.columns if c.startswith("clr_")] + ["rel_level", "level_volatility", "seasonal_amplitude", "yoy_log_mean", "marketplace_trend"]
    ctx_cols = [c for c in stat.columns if stat[c].notna().mean() > 0.9]
    X_beh = features.zscore(prof[beh_cols]).to_numpy()
    X_ctx = features.zscore(stat[ctx_cols]).to_numpy()
    A = sparse.load_npz(data.ROOT / "outputs/learned_graph/A_static_learned.npz")
    P = graphs.diffusion_profile(A, 2)
    halves = {}
    for tag, Y in [("полные признаки", np.hstack([X_beh, X_ctx])), ("только потребление", X_beh), ("только контекст", X_ctx)]:
        lab = clustering.kefrin(Y, P, 4, xi="auto", distance="euclidean", seed=cfg["clustering"]["seed"], n_init=5, min_size=20)[0]
        sub_ct = pd.crosstab(me, lab)
        halves[tag] = {"ari": float(ari(me, lab)), "ami": float(ami(me, lab)),
                       "purity_by_type": {names_e[i]: float(sub_ct.loc[i].max() / sub_ct.loc[i].sum()) for i in range(len(names_e))}}

    # --- профили поведенческих типов -----------------------------------------------------------
    mr = clustering.mirkin_rule(raw, mb)
    mr.insert(0, "name", [names_b[int(c)] for c in mr.index])
    mr.to_csv(OUT / "beh_profiles_mirkin.csv")
    med = raw.groupby(mb).median()
    med.index.name = "cluster"
    med.insert(0, "name", [names_b[int(c)] for c in med.index])
    med.to_csv(OUT / "beh_profiles_median.csv")
    examples = {}
    pop = attrs["population"].reindex(tids)
    nm = attrs["name"].reindex(tids).astype(str)
    for c in range(K):
        idx = np.where(mb == c)[0]
        top = idx[np.argsort(-pop.iloc[idx].fillna(0).to_numpy())[:5]]
        examples[names_b[c]] = [nm.iloc[i] for i in top]

    # --- устойчивость --------------------------------------------------------------------------
    summ_dyn = pd.read_csv(data.ROOT / "outputs/kefrin_t/summary_learned_beh_windowlog.csv")
    dyn = summ_dyn[(summ_dyn.method == "kefrin_t_affect_fe") & (summ_dyn.K == K)].iloc[0].to_dict()
    stab_p = data.ROOT / "outputs/stability/stability_learned_beh.csv"
    stab = pd.read_csv(stab_p) if stab_p.exists() else pd.DataFrame()
    stab_rows = stab[stab.method == "kefrin"][["K", "ari_mean", "jaccard_min"]].round(3).to_dict(orient="records") if len(stab) else []

    mk = interpret.markov_transitions(B.to_numpy().T.astype(int), K)
    summary = {
        "sojourn_beh": [float(x) for x in mk["sojourn_months"]], "stationary_beh": [float(x) for x in mk["stationary"]],
        "K_beh": K, "names_beh": names_b, "sizes_beh": {names_b[i]: int((mb == i).sum()) for i in range(K)},
        "share_never_switch_beh": float((switches == 0).mean()), "switch_median_beh": float(dyn["switch_median"]),
        "ami_median_beh": float(dyn["ami_median"]), "SW_beh": float(dyn["SW"]), "Q_beh": float(dyn["Q"]),
        "ari_lenses": float(ari(me, mb)), "ami_lenses": float(ami(me, mb)), "ami_sub_beh": float(ami(ms, mb)),
        "row_purity": row_purity, "col_purity": col_purity, "halves": halves, "stability_beh": stab_rows,
        "examples": examples, "crosstab_macro": ct.to_dict(orient="index"),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=float)
    make_figure(mo, ct, mr, names_b, names_e, K)

    pd.set_option("display.width", 240)
    print(f"поведенческая линза K={K}: размеры {summary['sizes_beh']}, без смен {summary['share_never_switch_beh']:.1%}, смена/мес {summary['switch_median_beh']:.3f}, AMI {summary['ami_median_beh']:.3f}")
    print(f"согласие линз: ARI {summary['ari_lenses']:.3f}, AMI {summary['ami_lenses']:.3f}; с подтипами AMI {summary['ami_sub_beh']:.3f}")
    print("кросс-таблица (экономическая × поведенческая):\n" + ct.to_string())
    print("чистота строк:", row_purity, "\nчистота столбцов:", col_purity)
    print("половины признаков → ARI с экономической:", {k: round(v["ari"], 3) for k, v in halves.items()})
    print("устойчивость (bootstrap, статика):", stab_rows)
    print("профили Миркина (поведенческие типы):\n" + mr.round(2).T.to_string())
    print("примеры:", examples)


def make_figure(mo, ct, mr, names_b, names_e, K):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from shapely.geometry import box

    FIG.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(data.ROOT / "data/processed/mo_polygons_simplified.geojson")
    gdf["territory_id"] = gdf["territory_id"].astype(int)
    clip = box(-179.95, -90, 179.95, 90)
    geoms = [g.intersection(box(0, -90, 179.95, 90)) if (g.bounds[0] < -100 and g.bounds[2] > 100) else g.intersection(clip) for g in gdf.geometry]
    gdf = gdf.set_geometry(geoms).set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.merge(mo[["beh"]], left_on="territory_id", right_index=True, how="left")
    gdf = gdf.to_crs("+proj=aea +lat_1=50 +lat_2=70 +lat_0=60 +lon_0=100 +datum=WGS84 +units=m")

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.25, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    cmap = ListedColormap(BEH_COLORS[:K])
    gdf.plot(column="beh", cmap=cmap, vmin=-0.5, vmax=K - 0.5, ax=ax, linewidth=0.15, edgecolor="#ffffff", missing_kwds={"color": "#e9e8e4"})
    ax.set_axis_off()
    ax.set_title("Поведенческая линза: типы потребительского поведения только по данным СберИндекса", loc="left", fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BEH_COLORS[i], label=f"{names_b[i]} ({int((mo.beh == i).sum())})") for i in range(K)], loc="lower left", fontsize=8, frameon=False)

    ax2 = fig.add_subplot(gs[1, 0])
    M = ct.to_numpy().astype(float)
    share = M / M.sum(axis=1, keepdims=True)
    ax2.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax2.text(j, i, f"{int(M[i, j])}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if share[i, j] > 0.55 else "#1a1a1a")
    import textwrap
    ax2.set_xticks(range(M.shape[1]), ["\n".join(textwrap.wrap(c, 14)) for c in ct.columns], fontsize=8)
    ax2.set_yticks(range(M.shape[0]), [r.replace(" и ", " и\n") for r in ct.index], fontsize=8)
    ax2.set_title("Экономическая × поведенческая: число МО и доля строки", loc="left", fontsize=10, fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    feats = ["общепит", "маркетплейсы", "продовольствие", "здоровье", "транспорт", "траты на жителя", "сезонность", "рост г/г"]
    feats = [f for f in feats if f in mr.columns]
    width = 0.8 / K
    for c in range(K):
        vals = mr.loc[c, feats].to_numpy(dtype=float) * 100
        ax3.barh(np.arange(len(feats)) + (c - (K - 1) / 2) * width, vals, height=width, color=BEH_COLORS[c], label=names_b[c])
    ax3.set_yticks(range(len(feats)), feats, fontsize=8)
    ax3.invert_yaxis()
    ax3.axvline(0, color="#c9c8c3", lw=1)
    ax3.set_xlabel("отклонение от среднего по России, % (правило Миркина)")
    ax3.grid(axis="x", color="#eeede8")
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.set_title("Профили поведенческих типов", loc="left", fontsize=10, fontweight="bold")
    fig.savefig(FIG / "fig5_two_lenses.png", dpi=110, bbox_inches="tight", facecolor="white")
    print("рисунок:", FIG / "fig5_two_lenses.png")


if __name__ == "__main__":
    main()
