"""Сборка панели МО × месяц и статических атрибутов. Запуск: .venv312/bin/python scripts/build_panel.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mo_types import data  # noqa: E402


def main():
    cfg = data.load_config()
    panel = data.build_panel(cfg)
    tids = panel.index.get_level_values("territory_id").unique()
    print(f"панель: {panel.shape[0]} строк, {len(tids)} МО, "
          f"{panel.index.get_level_values('date').nunique()} месяцев")
    neg_other = (panel["other"] < 0).mean()
    print(f"доля строк с other<0: {neg_other:.4f}; медианные доли: "
          + ", ".join(f"{c}={(panel[c] / panel['total']).median():.3f}" for c in panel.columns if c != "total"))
    attrs = data.build_attributes(cfg, tids)
    print(f"атрибуты: {attrs.shape[0]} МО × {attrs.shape[1]} колонок")
    for c in ["population", "urban_share", "wage", "emp_total", "market_access", "lat"]:
        if c in attrs:
            print(f"  {c}: заполнено {attrs[c].notna().sum()} ({attrs[c].notna().mean():.1%})")
    emp_cols = [c for c in attrs.columns if c.startswith("emp_share_")]
    print(f"  разделов ОКВЭД2 в занятости: {len(emp_cols)}: {[c.replace('emp_share_', '') for c in emp_cols]}")
    print("  типы МО:", attrs["mo_type"].value_counts().to_dict())
    print("  годы данных: население", attrs["population_year"].value_counts().to_dict(),
          "| зарплата", attrs["wage_year"].value_counts().to_dict(),
          "| занятость", attrs["emp_year"].value_counts().to_dict())
    data.save_processed(cfg, panel, attrs)
    print("сохранено в", cfg["paths"]["processed"])


if __name__ == "__main__":
    main()
