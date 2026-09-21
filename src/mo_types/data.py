"""Загрузка данных и сборка панели МО × месяц и таблицы статических атрибутов."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _p(cfg: dict, key: str) -> Path:
    return ROOT / cfg["data"][key]


# ----------------------------------------------------------------------------- потребление
def load_consumption(cfg: dict) -> pd.DataFrame:
    """Длинная таблица: territory_id, date (Timestamp, 1-е число месяца), category, value (руб.)."""
    df = pd.read_parquet(_p(cfg, "consumption"))
    df["date"] = pd.to_datetime(df["date"].astype(str) + "-01")
    df["territory_id"] = df["territory_id"].astype(int)
    df["value"] = df["value"].astype(float)
    return df[["territory_id", "date", "category", "value"]]


def build_panel(cfg: dict) -> pd.DataFrame:
    """Широкая панель: индекс (territory_id, date); колонки total, категории, other.

    Оставляются только МО с полной историей по всем колонкам (min_months месяцев).
    """
    cats = cfg["panel"]["categories"]
    total = cfg["panel"]["total"]
    long = load_consumption(cfg)
    wide = long.pivot_table(index=["territory_id", "date"], columns="category", values="value")
    wide = wide.rename(columns={total: "total"})
    missing = [c for c in cats if c not in wide.columns]
    if missing:
        raise ValueError(f"нет категорий в данных: {missing}")
    wide = wide[["total", *cats]]
    wide["other"] = wide["total"] - wide[cats].sum(axis=1)
    complete = wide.dropna()
    n_months = complete.groupby(level="territory_id").size()
    keep = n_months[n_months >= cfg["panel"]["min_months"]].index
    panel = complete.loc[complete.index.get_level_values("territory_id").isin(keep)].sort_index()
    panel.columns.name = None
    return panel


# ----------------------------------------------------------------------------- справочник
def load_dictionary(cfg: dict) -> pd.DataFrame:
    """Одна строка на territory_id: версия, действующая в dictionary_year, иначе последняя."""
    year = cfg["panel"]["dictionary_year"]
    d = pd.read_excel(_p(cfg, "dictionary"))
    d = d.sort_values(["territory_id", "year_from"])
    valid = d[(d.year_from <= year) & (d.year_to > year)]
    latest = d.groupby("territory_id").tail(1)
    out = pd.concat([valid, latest[~latest.territory_id.isin(valid.territory_id)]])
    out = out.drop_duplicates("territory_id").copy()
    out["oktmo8"] = out["oktmo"].astype(str).str.replace("-", "", regex=False).str[:8]
    out = out.rename(
        columns={
            "municipal_district_name": "name",
            "municipal_district_name_short": "name_short",
            "municipal_district_type": "mo_type",
            "municipal_district_status": "mo_status",
            "municipal_district_center": "center",
            "municipal_district_center_lat": "lat",
            "municipal_district_center_lon": "lon",
        }
    )
    cols = ["territory_id", "name", "name_short", "mo_type", "mo_status", "center",
            "region_code", "region_name", "oktmo", "oktmo8", "lat", "lon", "year_from", "year_to"]
    return out[cols].reset_index(drop=True)


# ----------------------------------------------------------------------------- Росстат
def _read_rosstat(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, sep=";", dtype=str)
    r = r[r["mun_level"].str.contains("верхнего", na=False)].copy()
    r["oktmo8"] = r["oktmo"].str.zfill(8).str[:8]
    r["year"] = r["year"].astype(int)
    r["value"] = pd.to_numeric(r["indicator_value"].str.replace(",", "."), errors="coerce")
    return r


def _annual(r: pd.DataFrame) -> pd.DataFrame:
    """Годовые значения: у накопительных показателей оставляем период «Январь-декабрь»."""
    if "indicator_period" in r.columns and r["indicator_period"].eq("Январь-декабрь").any():
        return r[r["indicator_period"] == "Январь-декабрь"]
    return r


def _pick_years(r: pd.DataFrame, years=(2024, 2023, 2022)) -> pd.DataFrame:
    """Для каждого oktmo8 берём самое свежее непустое значение из списка лет."""
    r = r[r.year.isin(years) & r["value"].notna()].copy()
    r["_rank"] = r["year"].map({y: i for i, y in enumerate(years)})
    r = r.sort_values(["oktmo8", "_rank"]).drop_duplicates("oktmo8")
    return r


def load_population(cfg: dict) -> pd.DataFrame:
    r = _read_rosstat(_p(cfg, "rosstat_population"))
    if "mest" in r.columns:
        allpop = r[r["mest"].str.contains("Все", na=False)]
        urban = r[r["mest"].str.contains("Городск", na=False)]
    else:
        allpop, urban = r, r.iloc[0:0]
    out = _pick_years(allpop)[["oktmo8", "year", "value"]].rename(
        columns={"value": "population", "year": "population_year"})
    if len(urban):
        u = _pick_years(urban)[["oktmo8", "value"]].rename(columns={"value": "urban_population"})
        out = out.merge(u, on="oktmo8", how="left")
        # нет строки «Городское население» при наличии общей численности = городского населения нет
        out["urban_population"] = out["urban_population"].where(out["population"].isna(), out["urban_population"].fillna(0.0))
        out["urban_share"] = out["urban_population"] / out["population"]
    return out


_SECTION_RE = re.compile(r"^Раздел\s+(\S)\b")
_CYR2LAT = str.maketrans("АВЕНКМОРСТХ", "ABEHKMOPCTX")  # кириллические буквы-двойники в кодах разделов


def _okved_key(s: str) -> str:
    if s.startswith("Всего"):
        return "total"
    m = _SECTION_RE.match(s)
    if m:
        return m.group(1).translate(_CYR2LAT)
    return re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", s.strip().lower())[:30]


def load_wages(cfg: dict) -> pd.DataFrame:
    r = _annual(_read_rosstat(_p(cfg, "rosstat_wages")))
    r["key"] = r["okved2"].map(_okved_key)
    tot = _pick_years(r[r.key == "total"])[["oktmo8", "year", "value"]].rename(
        columns={"value": "wage", "year": "wage_year"})
    return tot


def load_employment(cfg: dict) -> pd.DataFrame:
    """Среднесписочная численность (без МП): total и доли по разделам ОКВЭД2 (буквы)."""
    r = _annual(_read_rosstat(_p(cfg, "rosstat_employment")))
    r["key"] = r["okved2"].map(_okved_key)
    # один год на МО: тот, где есть total (свежий)
    tot = _pick_years(r[r.key == "total"])[["oktmo8", "year", "value"]].rename(
        columns={"value": "emp_total", "year": "emp_year"})
    sec = r[(r.key.str.len() == 1)].merge(tot[["oktmo8", "emp_year"]], on="oktmo8")
    sec = sec[sec.year == sec.emp_year]
    shares = sec.pivot_table(index="oktmo8", columns="key", values="value", aggfunc="first")
    shares = shares.div(tot.set_index("oktmo8")["emp_total"], axis=0)
    shares.columns = [f"emp_share_{c}" for c in shares.columns]
    out = tot.merge(shares.reset_index(), on="oktmo8", how="left")
    # отсутствие раздела в отчётности = нулевая численность в обследуемых организациях
    share_cols = [c for c in out.columns if c.startswith("emp_share_")]
    out.loc[out["emp_total"].notna(), share_cols] = out.loc[out["emp_total"].notna(), share_cols].fillna(0.0)
    return out


# ----------------------------------------------------------------------------- прочее
def load_market_access(cfg: dict) -> pd.DataFrame:
    m = pd.read_parquet(_p(cfg, "market_access"))
    m["territory_id"] = m["territory_id"].astype(int)
    return m


def load_connection(cfg: dict, kind: str = "highway") -> pd.DataFrame:
    c = pd.read_parquet(_p(cfg, "connection"))
    c = c[c["type"] == kind]
    return c[["territory_id_x", "territory_id_y", "distance"]].astype(
        {"territory_id_x": int, "territory_id_y": int, "distance": float})


def build_attributes(cfg: dict, territory_ids) -> pd.DataFrame:
    """Статические атрибуты МО: справочник + Росстат + доступность рынков."""
    d = load_dictionary(cfg)
    d = d[d.territory_id.isin(territory_ids)]
    pop = load_population(cfg)
    wage = load_wages(cfg)
    emp = load_employment(cfg)
    ma = load_market_access(cfg)
    out = (d.merge(pop, on="oktmo8", how="left")
             .merge(wage, on="oktmo8", how="left")
             .merge(emp, on="oktmo8", how="left")
             .merge(ma, on="territory_id", how="left"))
    out["is_urban_okrug"] = (out["mo_type"] == "городской округ").astype(int)
    return out.set_index("territory_id").sort_index()


def save_processed(cfg: dict, panel: pd.DataFrame, attrs: pd.DataFrame) -> None:
    out = ROOT / cfg["paths"]["processed"]
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "panel.parquet")
    attrs.to_parquet(out / "attributes.parquet")


def load_processed(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = ROOT / cfg["paths"]["processed"]
    return pd.read_parquet(out / "panel.parquet"), pd.read_parquet(out / "attributes.parquet")
