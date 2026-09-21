"""Волна маркетплейсов: единственный сильный сдвиг структуры трат за 2023–2024 и его география.

Что считаем:
- помесячная доля маркетплейсов в тратах (среднее по МО, взвешенное по населению, по макротипам);
- прирост доли по МО за год: среднее 2024 − среднее 2023 (годовые средние снимают сезонность);
- срезы по макротипам и по квинтилям индекса доступности рынков (в т. ч. только для сельских МО);
- регрессии прироста на доступность, население, зарплату, урбанизацию, стартовую долю и тип
  (HC1; вариант с фиксированными эффектами регионов; вариант только по сельским МО);
- расхождение: std доли по месяцам, внутри типов; тест log-t по доле;
- рисунок для отчёта (карта прироста, линии по типам, горб по доступности).

Запуск: .venv312/bin/python scripts/marketplace_wave.py
Выход: outputs/marketplace_wave/{mo.csv, series.csv, by_type.csv, by_access.csv, by_access_rural.csv, ols.csv,
        top_movers.csv, summary.json}, report/figures/fig4_marketplace.png
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from mo_types import data, features, interpret  # noqa: E402

OUT = data.ROOT / "outputs" / "marketplace_wave"
FIG = data.ROOT / "report" / "figures"
NAMES = {0: "Крупные и средние города", 1: "Северные и ресурсные", 2: "Столичные агломерации", 3: "Сельские и малые"}
COLORS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#4a3aa7"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = data.load_config()
    panel, attrs = data.load_processed(cfg)
    feat = features.monthly_features(panel, eps=cfg["features"]["clr_eps"])
    mo = pd.read_csv(data.ROOT / "outputs/final/mo_table.csv").set_index("id")
    mp = feat["share_Маркетплейсы"].unstack("date")
    cafe = feat["share_Общественное питание"].unstack("date")
    food = feat["share_Продовольствие"].unstack("date")
    total = panel["total"].unstack("date")
    months = [str(d)[:7] for d in mp.columns]
    mo = mo.reindex(mp.index)
    macro = mo.macro.astype(int)
    pop = mo.population.fillna(mo.population.median())

    # --- помесячные ряды ---------------------------------------------------------------------
    w = pop / pop.sum()
    series = pd.DataFrame({
        "date": months,
        "mean_share": mp.mean().to_numpy(),
        "median_share": mp.median().to_numpy(),
        "pop_weighted_share": (mp.mul(w, axis=0)).sum().to_numpy(),
        "std_share": mp.std().to_numpy(),
        "within_type_std": np.array([np.mean([mp.loc[macro == k, c].std() for k in range(4)]) for c in mp.columns]),
        "cafe_mean_share": cafe.mean().to_numpy(),
        "food_mean_share": food.mean().to_numpy(),
    })
    for k in range(4):
        series[f"type{k}_mean"] = mp.loc[macro == k].mean().to_numpy()
    series.to_csv(OUT / "series.csv", index=False)

    # --- прирост по МО: годовые средние -------------------------------------------------------
    y23 = [c for c in mp.columns if c.year == 2023]
    y24 = [c for c in mp.columns if c.year == 2024]
    d = pd.DataFrame({
        "share_2023": mp[y23].mean(axis=1), "share_2024": mp[y24].mean(axis=1),
        "cafe_2023": cafe[y23].mean(axis=1), "cafe_2024": cafe[y24].mean(axis=1),
        "food_2023": food[y23].mean(axis=1), "food_2024": food[y24].mean(axis=1),
        "total_2023": total[y23].mean(axis=1), "total_2024": total[y24].mean(axis=1),
    })
    d["d_share_pp"] = (d.share_2024 - d.share_2023) * 100
    d["d_cafe_pp"] = (d.cafe_2024 - d.cafe_2023) * 100
    d["d_food_pp"] = (d.food_2024 - d.food_2023) * 100
    d["growth_total"] = d.total_2024 / d.total_2023 - 1
    d = d.join(mo[["name", "short", "region", "mo_type", "macro", "sub", "market_access", "population", "wage", "urban_share", "lat", "lon"]])
    d["type"] = d.macro.map(NAMES)
    d["log_access"] = np.log(d.market_access.clip(lower=1))
    d["log_pop"] = np.log(d.population.clip(lower=1))
    d["log_wage"] = np.log(d.wage.clip(lower=1))
    d["start"] = d.share_2023 * 100
    d["access_q"] = pd.qcut(d.market_access, 5, labels=["q1 (низкая)", "q2", "q3", "q4", "q5 (высокая)"])
    d.to_csv(OUT / "mo.csv")

    # --- срезы --------------------------------------------------------------------------------
    def med_iqr(g, col):
        return pd.Series({"n": len(g), f"{col}_median": g[col].median(), f"{col}_q25": g[col].quantile(.25), f"{col}_q75": g[col].quantile(.75)})
    by_type = d.groupby("type").apply(lambda g: pd.concat([med_iqr(g, "d_share_pp").drop("n"), pd.Series({
        "n": len(g), "share_2023_median": g.share_2023.median() * 100, "share_2024_median": g.share_2024.median() * 100,
        "share_std_2023": g.share_2023.std() * 100, "share_std_2024": g.share_2024.std() * 100,
        "d_cafe_pp_median": g.d_cafe_pp.median(), "d_food_pp_median": g.d_food_pp.median()})]), include_groups=False)
    by_type.to_csv(OUT / "by_type.csv")
    by_access = d.groupby("access_q", observed=True).apply(lambda g: med_iqr(g, "d_share_pp"), include_groups=False)
    by_access["access_median"] = d.groupby("access_q", observed=True).market_access.median()
    by_access["share_2023_median"] = d.groupby("access_q", observed=True).share_2023.median() * 100
    by_access.to_csv(OUT / "by_access.csv")
    rural = d[d.macro == 3].copy()
    rural["access_q"] = pd.qcut(rural.market_access, 5, labels=["q1 (низкая)", "q2", "q3", "q4", "q5 (высокая)"])
    by_access_rural = rural.groupby("access_q", observed=True).apply(lambda g: med_iqr(g, "d_share_pp"), include_groups=False)
    by_access_rural["access_median"] = rural.groupby("access_q", observed=True).market_access.median()
    by_access_rural.to_csv(OUT / "by_access_rural.csv")

    # --- регрессии ----------------------------------------------------------------------------
    base = "d_share_pp ~ log_access + log_pop + log_wage + urban_share + start + C(type)"
    specs = {
        "все МО, тип": base,
        "все МО, тип + регион (FE)": base + " + C(region)",
        "только сельские и малые": "d_share_pp ~ log_access + log_pop + log_wage + urban_share + start",
        "только сельские, регион (FE)": "d_share_pp ~ log_access + log_pop + log_wage + urban_share + start + C(region)",
    }
    rows = []
    for name, f in specs.items():
        dd = d if "сельские" not in name else rural
        m = smf.ols(f, data=dd.dropna(subset=["log_access", "log_pop", "log_wage", "urban_share", "start"])).fit(cov_type="HC1")
        for term in ["log_access", "log_pop", "log_wage", "urban_share", "start"]:
            rows.append({"spec": name, "term": term, "coef": m.params[term], "t": m.tvalues[term], "p": m.pvalues[term], "n": int(m.nobs), "r2": m.rsquared})
        for term in [t for t in m.params.index if t.startswith("C(type)")]:
            rows.append({"spec": name, "term": term.replace("C(type)[T.", "тип: ").rstrip("]"), "coef": m.params[term], "t": m.tvalues[term], "p": m.pvalues[term], "n": int(m.nobs), "r2": m.rsquared})
    ols = pd.DataFrame(rows)
    ols.to_csv(OUT / "ols.csv", index=False)

    # --- расхождение и log-t ------------------------------------------------------------------
    logt = interpret.club_convergence_logt(mp.T.to_numpy() + 1e-4)
    logt_by_type = {NAMES[k]: interpret.club_convergence_logt(mp.loc[macro == k].T.to_numpy() + 1e-4) for k in range(4)}

    # --- лидеры и отстающие -------------------------------------------------------------------
    cols = ["name", "region", "type", "share_2023", "share_2024", "d_share_pp", "market_access", "population"]
    top = pd.concat([d.nlargest(15, "d_share_pp")[cols].assign(group="лидеры"), d.nsmallest(15, "d_share_pp")[cols].assign(group="отстающие")])
    top.to_csv(OUT / "top_movers.csv")

    # --- сводка ---------------------------------------------------------------------------------
    m1 = ols[ols.spec == "все МО, тип"].set_index("term")
    m2 = ols[ols.spec == "все МО, тип + регион (FE)"].set_index("term")
    m3 = ols[ols.spec == "только сельские, регион (FE)"].set_index("term")
    summary = {
        "mean_share_first": float(series.mean_share.iloc[0]), "mean_share_last": float(series.mean_share.iloc[-1]),
        "pop_weighted_first": float(series.pop_weighted_share.iloc[0]), "pop_weighted_last": float(series.pop_weighted_share.iloc[-1]),
        "share_2023_median": float(d.share_2023.median()), "share_2024_median": float(d.share_2024.median()),
        "d_share_pp_median": float(d.d_share_pp.median()), "share_negative_delta": float((d.d_share_pp < 0).mean()),
        "std_first": float(series.std_share.iloc[0]), "std_last": float(series.std_share.iloc[-1]),
        "cafe_mean_first": float(series.cafe_mean_share.iloc[0]), "cafe_mean_last": float(series.cafe_mean_share.iloc[-1]),
        "food_mean_first": float(series.food_mean_share.iloc[0]), "food_mean_last": float(series.food_mean_share.iloc[-1]),
        "by_type": by_type.round(3).to_dict(orient="index"),
        "by_access": by_access.round(3).to_dict(orient="index"),
        "by_access_rural": by_access_rural.round(3).to_dict(orient="index"),
        "ols_all": {t: {"coef": float(m1.loc[t, "coef"]), "t": float(m1.loc[t, "t"])} for t in m1.index}, "ols_all_r2": float(m1.r2.iloc[0]),
        "ols_fe": {t: {"coef": float(m2.loc[t, "coef"]), "t": float(m2.loc[t, "t"])} for t in m2.index}, "ols_fe_r2": float(m2.r2.iloc[0]),
        "ols_rural_fe": {t: {"coef": float(m3.loc[t, "coef"]), "t": float(m3.loc[t, "t"])} for t in m3.index}, "ols_rural_fe_r2": float(m3.r2.iloc[0]),
        "logt": logt, "logt_by_type": logt_by_type,
        "corr_d_share": {k: float(d.d_share_pp.corr(d[k])) for k in ["log_access", "log_pop", "log_wage", "urban_share", "start"]},
        "corr_d_share_d_cafe": float(d.d_share_pp.corr(d.d_cafe_pp)), "corr_d_share_d_food": float(d.d_share_pp.corr(d.d_food_pp)),
        "corr_d_share_growth": float(d.d_share_pp.corr(d.growth_total)),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1, default=float)

    # --- рисунок для отчёта ---------------------------------------------------------------------
    make_figure(d, series, by_access, by_access_rural)

    # --- печать ---------------------------------------------------------------------------------
    pd.set_option("display.width", 220)
    print(f"доля маркетплейсов (среднее по МО): {months[0]} {summary['mean_share_first']:.3f} → {months[-1]} {summary['mean_share_last']:.3f}; "
          f"взвешенно по населению {summary['pop_weighted_first']:.3f} → {summary['pop_weighted_last']:.3f}; std {summary['std_first']:.4f} → {summary['std_last']:.4f}")
    print(f"медианный прирост 2024−2023: {summary['d_share_pp_median']:.2f} п.п.; МО с отрицательным приростом: {summary['share_negative_delta']:.1%}")
    print("по типам:\n", by_type.round(2).to_string())
    print("по квинтилям доступности:\n", by_access.round(2).to_string())
    print("сельские, по квинтилям доступности:\n", by_access_rural.round(2).to_string())
    print("регрессии:\n", ols.round(3).to_string(index=False))
    print(f"log-t по доле: b={logt['b']:.2f}, t={logt['t']:.1f} → {logt['verdict']}; по типам: " + ", ".join(f"{k}: b={v['b']:.2f} ({v['verdict']})" for k, v in logt_by_type.items()))
    print("корреляции прироста:", {k: round(v, 3) for k, v in summary["corr_d_share"].items()}, "с Δобщепит", round(summary["corr_d_share_d_cafe"], 3), "с Δпродовольствие", round(summary["corr_d_share_d_food"], 3), "с ростом трат", round(summary["corr_d_share_growth"], 3))
    print("лидеры и отстающие:\n", top.round(3).to_string())


def make_figure(d, series, by_access, by_access_rural):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import box

    FIG.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(data.ROOT / "data/processed/mo_polygons_simplified.geojson")
    gdf["territory_id"] = gdf["territory_id"].astype(int)
    clip = box(-179.95, -90, 179.95, 90)
    geoms = []
    for g in gdf.geometry:
        b = g.bounds
        geoms.append(g.intersection(box(0, -90, 179.95, 90)) if (b[0] < -100 and b[2] > 100) else g.intersection(clip))
    gdf = gdf.set_geometry(geoms).set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.merge(d[["d_share_pp"]], left_on="territory_id", right_index=True, how="left")
    gdf = gdf.to_crs("+proj=aea +lat_1=50 +lat_2=70 +lat_0=60 +lon_0=100 +datum=WGS84 +units=m")

    fig = plt.figure(figsize=(12, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.28, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    vmax = float(np.nanpercentile(d.d_share_pp, 97))
    gdf.plot(column="d_share_pp", cmap="Blues", vmin=0, vmax=vmax, ax=ax, linewidth=0.15, edgecolor="#ffffff",
             missing_kwds={"color": "#e9e8e4", "label": "нет данных"}, legend=True,
             legend_kwds={"label": "прирост доли, п.п.", "shrink": 0.5, "pad": 0.01})
    ax.set_axis_off()
    ax.set_title("Прирост доли маркетплейсов в тратах по МО за год (среднее 2024 − среднее 2023, п.п.)", loc="left", fontsize=12, fontweight="bold")

    ax2 = fig.add_subplot(gs[1, 0])
    x = pd.to_datetime(series.date)
    for k in range(4):
        ax2.plot(x, series[f"type{k}_mean"] * 100, color=COLORS[k], lw=2, label=NAMES[k])
    ax2.plot(x, series.mean_share * 100, color="#1a1a1a", lw=1.2, ls="--", label="все МО")
    ax2.set_ylabel("доля маркетплейсов, %")
    ax2.grid(axis="y", color="#eeede8")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    ax2.set_title("Доля маркетплейсов по макротипам, среднее по МО", loc="left", fontsize=10, fontweight="bold")

    ax3 = fig.add_subplot(gs[1, 1])
    order = [3, 1, 0, 2]
    bt = d.groupby("macro").d_share_pp.agg(median="median", q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75))
    pos = np.arange(len(order))
    ax3.bar(pos, [bt.loc[k, "median"] for k in order], width=0.6, color=[COLORS[k] for k in order])
    ax3.errorbar(pos, [bt.loc[k, "median"] for k in order],
                 yerr=[[bt.loc[k, "median"] - bt.loc[k, "q25"] for k in order], [bt.loc[k, "q75"] - bt.loc[k, "median"] for k in order]],
                 fmt="none", ecolor="#555", capsize=4, lw=1)
    for i, k in enumerate(order):
        ax3.text(pos[i], bt.loc[k, "q75"] + 0.12, f"{bt.loc[k, 'median']:.1f}", ha="center", fontsize=9)
    ax3.set_xticks(pos, [NAMES[k].replace(" и ", " и\n") for k in order], fontsize=8)
    ax3.set_ylabel("прирост доли за год, п.п. (медиана, межквартильный размах)")
    ax3.grid(axis="y", color="#eeede8")
    for sp in ("top", "right"):
        ax3.spines[sp].set_visible(False)
    ax3.set_title("Село растёт вдвое быстрее столичных агломераций", loc="left", fontsize=10, fontweight="bold")
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    fig.savefig(FIG / "fig4_marketplace.png", dpi=110, bbox_inches="tight", facecolor="white")
    print("рисунок:", FIG / "fig4_marketplace.png")


if __name__ == "__main__":
    main()
