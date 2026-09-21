"""Подготовка компактных данных для сайта «Атлас типов» (site_src/data/site_data.json).

Собирает из outputs/ всё, что нужно лендингу: геометрию МО в заранее спроецированных координатах
(коническая конформная проекция, единичный кадр 1000 px шириной), таблицу МО с последовательностями типов,
профили и правила типов, динамику, волну маркетплейсов, две линзы, проверки, обученный граф с силовой раскладкой.

Запуск: .venv312/bin/python scripts/prepare_site_data.py
"""
import json
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import geopandas as gpd  # noqa: E402
import igraph as ig  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy import sparse  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.ops import transform as shp_transform  # noqa: E402

from mo_types import data  # noqa: E402

ROOT = data.ROOT
OUT = ROOT / "site_src" / "data"
FRAME_W = 1000.0
SIMPLIFY_TOL = 0.03
COLORS_MACRO = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
COLORS_BEH = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100"]
FEATURE_ORDER = ["продовольствие", "здоровье", "общепит", "маркетплейсы", "транспорт", "прочее", "траты на жителя", "население",
                 "зарплата", "доступность рынков", "доля городского населения", "рост г/г", "сезонность",
                 "занятость: сельское хозяйство", "занятость: добыча", "занятость: промышленность", "занятость: рыночные услуги",
                 "занятость: бюджетный сектор"]
SHARE_ORDER = ["продовольствие", "здоровье", "общепит", "маркетплейсы", "транспорт", "прочее"]


def r(x, nd=3):
    if x is None:
        return None
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass
    return round(float(x), nd)


def load_json(p):
    return json.load(open(p, encoding="utf-8"))


def geometry(id_order):
    gdf = gpd.read_file(ROOT / "data/processed/mo_polygons_simplified.geojson")
    gdf["territory_id"] = gdf["territory_id"].astype(int)
    clip_east = box(0, -90, 179.99, 90)
    geoms = []
    for g in gdf.geometry:
        b = g.bounds
        geoms.append(g.intersection(clip_east) if (b[0] < -100 and b[2] > 100) else g)
    gdf = gdf.set_geometry(geoms)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOL, preserve_topology=True)
    tr = Transformer.from_crs("EPSG:4326", "+proj=lcc +lat_1=50 +lat_2=70 +lat_0=60 +lon_0=100 +datum=WGS84 +units=m", always_xy=True)
    proj = lambda x, y, z=None: tr.transform(x, y)  # noqa: E731
    gdf["geometry"] = gdf.geometry.apply(lambda g: shp_transform(proj, g))
    minx, miny, maxx, maxy = gdf.total_bounds
    scale = FRAME_W / (maxx - minx)
    H = (maxy - miny) * scale

    def px(x, y):
        return round((x - minx) * scale, 1), round((maxy - y) * scale, 1)

    geo = {}
    cent = {}
    nverts = 0
    by_id = {int(i): g for i, g in zip(gdf.territory_id, gdf.geometry)}
    for tid in id_order:
        g = by_id.get(tid)
        if g is None or g.is_empty:
            continue
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        rings_out = []
        for p in polys:
            rings = [p.exterior] + list(p.interiors)
            poly_rings = []
            for ring in rings:
                flat = []
                for x, y in ring.coords[:-1]:
                    a, b = px(x, y)
                    flat.extend([a, b])
                nverts += len(flat) // 2
                poly_rings.append(flat)
            rings_out.append(poly_rings)
        geo[tid] = rings_out
        c = g.representative_point()
        cent[tid] = px(c.x, c.y)
    print(f"геометрия: {len(geo)} МО, вершин {nverts}, кадр {FRAME_W:.0f}×{H:.0f}")
    return geo, cent, {"w": FRAME_W, "h": round(H, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    types = yaml.safe_load(open(ROOT / "configs/types.yaml", encoding="utf-8"))
    bundle = load_json(ROOT / "outputs/final/bundle.json")
    mo = bundle["mo"]
    ids = [m["id"] for m in mo]
    idx = {tid: i for i, tid in enumerate(ids)}
    geo, cent, frame = geometry(ids)

    # --- поведенческая линза и волна по МО --------------------------------------------------------
    beh = pd.read_csv(ROOT / "outputs/two_lenses/mo_beh.csv", index_col=0, dtype={"beh_seq": str})
    beh["beh_seq"] = beh["beh_seq"].astype(str).str.zfill(24)
    wave_mo = pd.read_csv(ROOT / "outputs/marketplace_wave/mo.csv", index_col=0)
    mob = pd.read_csv(ROOT / "outputs/mobility/by_type.csv", index_col=0) if (ROOT / "outputs/mobility/by_type.csv").exists() else None

    mo_out = []
    for m in mo:
        tid = m["id"]
        b = beh.loc[tid] if tid in beh.index else None
        w = wave_mo.loc[tid] if tid in wave_mo.index else None
        rec = {
            "id": tid, "n": m["name"], "s": m.get("short") or m["name"], "r": m["region"], "t": m["mo_type"],
            "ma": int(m["macro"]), "su": int(m["sub"]), "be": (int(b["beh"]) if b is not None else None),
            "ms": "".join(str(int(x)) for x in m["macro_seq"]), "ss": "".join(str(int(x)) for x in m["sub_seq"]),
            "bs": (str(b["beh_seq"]) if b is not None else None),
            "sw": [int(m["switches_macro"]), int(m["switches_sub"]), (int(b["switches_beh"]) if b is not None else 0)],
            "pop": r(m.get("population"), 0), "wage": r(m.get("wage"), 0), "acc": r(m.get("market_access"), 1),
            "urb": r(m.get("urban_share"), 3), "t23": r(m.get("total_2023"), 0), "t24": r(m.get("total_2024"), 0),
            "sh": [r(m["shares"].get(k), 4) for k in SHARE_ORDER] if isinstance(m.get("shares"), dict) else None,
            "rl": [r(x, 3) for x in m["rel_level"]], "pe": [int(p) for p in m.get("peers", [])],
            "mk": [r(m["mirkin"].get(k), 3) for k in FEATURE_ORDER] if isinstance(m.get("mirkin"), dict) else None,
            "dmp": (r(w["d_share_pp"], 2) if w is not None else None), "mp23": (r(w["share_2023"], 4) if w is not None else None),
            "mp24": (r(w["share_2024"], 4) if w is not None else None),
            "c": list(cent.get(tid, (None, None))),
        }
        mo_out.append(rec)

    # --- типы: имена, профили, медианы, размеры ----------------------------------------------------
    def prof_rows(rows):
        return [{"cluster": int(p["cluster"]), "name": p["name"], "size": int(p["size"]), "v": [r(p.get(k), 3) for k in FEATURE_ORDER]} for p in rows]
    def med_rows(rows):
        return [{"cluster": int(p["cluster"]) if "cluster" in p else int(i), "v": [r(p.get(k), 3) for k in FEATURE_ORDER]} for i, p in enumerate(rows)]
    beh_mirkin = pd.read_csv(ROOT / "outputs/two_lenses/beh_profiles_mirkin.csv", index_col=0)
    beh_median = pd.read_csv(ROOT / "outputs/two_lenses/beh_profiles_median.csv", index_col=0)
    lens = load_json(ROOT / "outputs/two_lenses/summary.json")
    names_beh = {int(k): v for k, v in types["names"]["beh"].items()}
    types_out = {
        "months": bundle["months"],
        "macro": {"names": [bundle["names_macro"][str(i)] for i in range(4)], "colors": COLORS_MACRO,
                  "sizes": [int((beh["macro"] == i).sum()) for i in range(4)], "sojourn": bundle["sojourn_macro"], "stationary": bundle["stationary_macro"],
                  "profiles": prof_rows(bundle["profiles_macro"]), "medians": med_rows(bundle["medians_macro"]),
                  "transitions": bundle["transitions_macro"], "sizes_by_month": bundle["sizes_macro"], "switch_by_month": bundle["switch_macro"]},
        "sub": {"names": [bundle["names_sub"][str(i)] for i in range(6)], "sizes": [int((beh["sub"] == i).sum()) for i in range(6)],
                "sojourn": bundle["sojourn_sub"], "profiles": prof_rows(bundle["profiles_sub"]), "medians": med_rows(bundle["medians_sub"]),
                "transitions": bundle["transitions_sub"], "sizes_by_month": bundle["sizes_sub"], "switch_by_month": bundle["switch_sub"]},
        "beh": {"names": [names_beh[i] for i in range(len(names_beh))], "colors": COLORS_BEH,
                "sizes": [int(lens["sizes_beh"][names_beh[i]]) for i in range(len(names_beh))], "sojourn": lens["sojourn_beh"], "stationary": lens["stationary_beh"],
                "profiles": [{"cluster": int(c), "name": row["name"], "size": int(row["size"]), "v": [r(row.get(k), 3) for k in FEATURE_ORDER]} for c, row in beh_mirkin.iterrows()],
                "medians": [{"cluster": int(c), "v": [r(row.get(k), 3) for k in FEATURE_ORDER]} for c, row in beh_median.iterrows()]},
        "crosstab_macro_sub": bundle["crosstab"], "grand_means": [r(bundle["grand_means"].get(k), 4) for k in FEATURE_ORDER],
        "feature_order": FEATURE_ORDER, "share_order": SHARE_ORDER,
        "lens": {"crosstab": lens["crosstab_macro"], "ari": r(lens["ari_lenses"]), "ami": r(lens["ami_lenses"]), "halves": {k: r(v["ari"]) for k, v in lens["halves"].items()},
                 "halves_purity": {k: {kk: r(vv, 2) for kk, vv in v["purity_by_type"].items()} for k, v in lens["halves"].items()},
                 "stability": lens["stability_beh"], "never": r(lens["share_never_switch_beh"]), "switch": r(lens["switch_median_beh"]), "examples": lens["examples"]},
    }
    # интерпретация
    interp = {}
    for tag in ["K4_windowlog", "K6_windowlog", "K5_beh_windowlog"]:
        s = load_json(ROOT / f"outputs/interpretation/{tag}/summary.json")
        interp[tag] = {k: s.get(k) for k in ["imm_raw_acc", "imm_std_acc", "rules_raw", "surrogate_K", "surrogate_2K", "external", "spatial", "clubs", "transition", "sojourn"] if k in s}
        interp[tag]["keys"] = list(s.keys())
    # динамика: до/после
    ba = pd.read_csv(ROOT / "outputs/dynamics_cleanup/before_after.csv")
    sbm = pd.read_csv(ROOT / "outputs/dynamics_cleanup/switches_by_month.csv", index_col=0)
    cleanup = {"table": ba.round(3).to_dict(orient="records"), "months": list(sbm.index), "switches": {c: [int(x) for x in sbm[c]] for c in sbm.columns},
               "summary": load_json(ROOT / "outputs/dynamics_cleanup/summary.json")}
    # волна
    wsum = load_json(ROOT / "outputs/marketplace_wave/summary.json")
    wser = pd.read_csv(ROOT / "outputs/marketplace_wave/series.csv")
    top = pd.read_csv(ROOT / "outputs/marketplace_wave/top_movers.csv")
    wave = {"series": {c: [r(x, 4) for x in wser[c]] for c in wser.columns if c != "date"}, "dates": list(wser.date),
            "by_type": wsum["by_type"], "by_access": wsum["by_access"], "ols_all": wsum["ols_all"], "ols_fe": wsum["ols_fe"], "r2": [r(wsum["ols_all_r2"], 2), r(wsum["ols_fe_r2"], 2)],
            "logt": wsum["logt"], "logt_by_type": wsum["logt_by_type"], "corr": {"food": r(wsum["corr_d_share_d_food"], 2), "cafe": r(wsum["corr_d_share_d_cafe"], 2)},
            "kpi": {k: r(wsum[k], 4) for k in ["mean_share_first", "mean_share_last", "pop_weighted_first", "pop_weighted_last", "d_share_pp_median", "std_first", "std_last", "food_mean_first", "food_mean_last", "cafe_mean_first", "cafe_mean_last"]},
            "top": top[["territory_id", "name", "region", "type", "share_2023", "share_2024", "d_share_pp", "group"]].round(3).to_dict(orient="records")}
    # синтетика и устойчивость, зоопарк, правила ребра, качество
    syn = pd.read_csv(ROOT / "outputs/synthetic/synthetic_results.csv")
    g = syn.groupby(["s_e", "method"]).mean(numeric_only=True).drop(columns=["seed"]).round(3).reset_index()
    stab = pd.read_csv(ROOT / "outputs/stability/stability_learned.csv"); stab_beh = pd.read_csv(ROOT / "outputs/stability/stability_learned_beh.csv")
    zoo = pd.read_csv(ROOT / "outputs/method_zoo/zoo_learned.csv")
    edge = pd.read_csv(ROOT / "outputs/edge_rules/edge_rules_comparison.csv")
    edge_sel = edge[(edge.sparsifier == "knn15_mutual") | (edge.rule == "learned_logmodel")]
    q = pd.read_csv(ROOT / "outputs/kefrin_t/summary_learned_windowlog.csv")
    alphas = pd.read_csv(ROOT / "outputs/kefrin_t/alphas_learned_windowlog.csv")
    a4 = alphas[(alphas.method == "kefrin_t_affect_fe") & (alphas.K == 4)]
    checks = {
        "synthetic": g.to_dict(orient="records"),
        "stability": stab[stab.method == "kefrin"][["K", "ari_mean", "jaccard_min"]].round(3).to_dict(orient="records"),
        "stability_beh": stab_beh[stab_beh.method == "kefrin"][["K", "ari_mean", "jaccard_min"]].round(3).to_dict(orient="records"),
        "zoo": zoo[["method", "K", "SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "modularity", "boot_ari"]].round(3).to_dict(orient="records"),
        "edge": edge_sel[["rule", "sparsifier", "edges", "assort_types", "same_region_share", "components", "isolates", "k4_Q", "k4_SW"]].round(3).to_dict(orient="records"),
        "edge_agreement": pd.read_csv(ROOT / "outputs/edge_rules/rule_agreement_jaccard.csv", index_col=0).round(3).to_dict(orient="index"),
        "temporal": q[q.method.isin(["kefrin_warm", "kefrin_fixed0.5", "kefrin_t_affect", "kefrin_t_affect_fe_noreset", "kefrin_t_affect_fe", "temporal_leiden_w2.0"])][["method", "K", "ami_median", "switch_median", "switch_max", "share_never_switch", "SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ", "Q", "alpha_mean", "beta_mean"]].round(3).to_dict(orient="records"),
        "alphas": {"date": list(a4.date), "alpha": [r(x) for x in a4.alpha], "beta": [r(x) for x in a4.beta]},
        "network_view": load_json(ROOT / "outputs/network_view/summary.json"),
        "network_crosstab": pd.read_csv(ROOT / "outputs/network_view/crosstab_macro.csv", index_col=0).to_dict(orient="index"),
        "mobility": load_json(ROOT / "outputs/mobility/summary.json") if (ROOT / "outputs/mobility/summary.json").exists() else None,
        "graph_stats": pd.read_csv(ROOT / "outputs/learned_graph/windowlog/stats.csv")[["date", "edges", "deg_mean", "isolates", "jaccard_prev", "jaccard_knn_prev"]].round(3).to_dict(orient="records"),
    }
    cases = load_json(ROOT / "outputs/case_studies/cases.json")

    # --- обученный граф и раскладка ------------------------------------------------------------------
    A = sparse.load_npz(ROOT / "outputs/learned_graph/A_static_learned.npz").tocoo()
    lab = pd.read_csv(ROOT / "outputs/kefrin_t/labels_affect_fe_K4_learned_windowlog.csv", index_col=0)
    node_ids = [int(x) for x in lab.index]
    edges = [(int(i), int(j), float(w)) for i, j, w in zip(A.row, A.col, A.data) if i < j]
    gi = ig.Graph(n=len(node_ids), edges=[(i, j) for i, j, _ in edges])
    gi.es["weight"] = [w for _, _, w in edges]
    random.seed(42)
    ig.set_random_number_generator(random)
    # гигантскую компоненту раскладываем отдельно: graphopt уносит отсоединённые куски на край кадра
    comp = gi.connected_components(); memb = np.array(comp.membership); big = int(np.argmax([len(c) for c in comp]))
    main_idx = np.where(memb == big)[0]; rest_idx = np.where(memb != big)[0]
    lay = gi.induced_subgraph(main_idx.tolist()).layout_graphopt(niter=800)  # graphopt даёт самое ясное разделение сообществ (проверено 21.09)
    Lm = np.array(lay.coords)
    lo, hi = np.percentile(Lm, 1, axis=0), np.percentile(Lm, 99, axis=0)
    Z = (Lm - lo) / (hi - lo)  # мягкое сжатие выбросов вместо обрезки
    Z = 0.5 + 0.5 * np.tanh(1.8 * (Z - 0.5)) / np.tanh(0.9)
    Z = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0))
    L = np.zeros((len(node_ids), 2)); L[main_idx] = Z * [0.94, 0.84] + [0.03, 0.03]
    # малые компоненты и изоляты — ровной строкой у нижнего края, сгруппированы по компонентам (крупные слева)
    order = sorted(rest_idx.tolist(), key=lambda k: (-len(comp[memb[k]]), memb[k], k))
    xs = np.linspace(0.12, 0.88, len(order)) if len(order) > 1 else [0.5]
    for pos_k, k in enumerate(order):
        L[k] = (xs[pos_k], 0.945)
    net_w, net_h = 1000.0, 537.4  # кадр сети совпадает с кадром карты, чтобы точки не уходили за сцену
    pos = {node_ids[k]: (round(float(L[k, 0] * net_w), 1), round(float(L[k, 1] * net_h), 1)) for k in range(len(node_ids))}
    order_map = {tid: idx[tid] for tid in node_ids if tid in idx}
    graph = {"w": net_w, "h": net_h, "n_small": int(len(rest_idx)), "n_isolates": int(sum(1 for c in comp if len(c) == 1)),
             "pos": [list(pos.get(tid, (None, None))) for tid in ids],
             "edges": [[order_map[node_ids[i]], order_map[node_ids[j]], round(w, 3)] for i, j, w in edges if node_ids[i] in order_map and node_ids[j] in order_map]}
    print(f"граф: узлов {len(node_ids)}, рёбер {len(graph['edges'])}")

    site = {"frame": frame, "geo": geo, "mo": mo_out, "types": types_out, "interp": interp, "cleanup": cleanup, "wave": wave,
            "checks": checks, "cases": cases, "graph": graph}
    js = json.dumps(site, ensure_ascii=False, separators=(",", ":"))
    (OUT / "site_data.json").write_text(js, encoding="utf-8")
    print(f"site_data.json: {len(js) / 1e6:.2f} МБ; geo {len(json.dumps(geo, separators=(',', ':'))) / 1e6:.2f} МБ; mo {len(json.dumps(mo_out, ensure_ascii=False, separators=(',', ':'))) / 1e6:.2f} МБ; graph {len(json.dumps(graph, separators=(',', ':'))) / 1e6:.2f} МБ")


if __name__ == "__main__":
    main()
