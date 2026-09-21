"""Чистка динамики: «до/после» второго прохода KEFRiN-T.

Сравнивает базовый прогон (стартовое разбиение по среднему первых 3 месяцев; копии в
outputs/dynamics_cleanup/baseline/) с итоговым (второй проход: старт от модальных типов первого прохода):
доля смен по месяцам, всплески, согласие месячных меток с модальным типом, согласие модальных типологий,
плюс синтетика (2pass, скользящее среднее). Рисунок для отчёта: смены по месяцам до/после.

Базовый прогон — это kefrin_t без --two-pass; его метки и сводка сохранены в outputs/dynamics_cleanup/baseline,
потому что итоговый пайплайн считает сразу второй проход и первый воспроизводить незачем.

Запуск: .venv312/bin/python scripts/dynamics_cleanup.py  (make cleanup)
Выход: outputs/dynamics_cleanup/{before_after.csv, switches_by_month.csv, summary.json}, report/figures/fig6_cleanup.png
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari  # noqa: E402

from mo_types import data  # noqa: E402

OUT = data.ROOT / "outputs" / "dynamics_cleanup"
BASE = OUT / "baseline"
FIG = data.ROOT / "report" / "figures"


def switches_by_month(L: pd.DataFrame) -> pd.Series:
    A = L.to_numpy()
    return pd.Series((A[:, 1:] != A[:, :-1]).sum(axis=0), index=L.columns[1:])


def main():
    rows, sw_rows = [], {}
    summary = {}
    for k, tag in [(4, "K4"), (6, "K6"), (5, "K5_beh")]:
        suffix = "learned_windowlog" if tag != "K5_beh" else "learned_beh_windowlog"
        Lb = pd.read_csv(BASE / f"labels_affect_fe_K{k}_{suffix}.csv", index_col=0)
        La = pd.read_csv(data.ROOT / "outputs/kefrin_t" / f"labels_affect_fe_K{k}_{suffix}.csv", index_col=0).reindex(Lb.index)
        Sb = pd.read_csv(BASE / f"summary_{suffix}.csv"); Sa = pd.read_csv(data.ROOT / "outputs/kefrin_t" / f"summary_{suffix}.csv")
        sb = Sb[(Sb.method == "kefrin_t_affect_fe") & (Sb.K == k)].iloc[0]; sa = Sa[(Sa.method == "kefrin_t_affect_fe") & (Sa.K == k)].iloc[0]
        mb = Lb.mode(axis=1)[0].astype(int); ma = La.mode(axis=1)[0].astype(int)
        swb, swa = switches_by_month(Lb), switches_by_month(La)
        sw_rows[f"{tag}_before"] = swb; sw_rows[f"{tag}_after"] = swa
        first_b = np.mean([ami(mb, Lb[c]) for c in Lb.columns[:3]]); first_a = np.mean([ami(ma, La[c]) for c in La.columns[:3]])
        for name, s, L, m, sw, first in [("до (старт по первым 3 мес.)", sb, Lb, mb, swb, first_b), ("после (второй проход)", sa, La, ma, swa, first_a)]:
            rows.append({"typology": tag, "variant": name, "AMI_months": s.ami_median, "switch_median": s.switch_median, "switch_max": s.switch_max,
                         "never_switch": s.share_never_switch, "switch_gt3": s.share_switch_gt3, "switches_total": int(sw.sum()),
                         "switches_2023_06": int(sw.get("2023-06", 0)), "switches_nov_dec_mean": float(sw[[c for c in sw.index if c[5:] in ("11", "12")]].mean()),
                         "switches_other_mean": float(sw[[c for c in sw.index if c[5:] not in ("11", "12", "01")]].mean()),
                         "AMI_first3_vs_modal": first, "SW": s.SW, "Q": s.Q, "AVI": s.AVI, "alpha_mean": s.alpha_mean, "reset_share": s.reset_share})
        summary[tag] = {"ari_modal_before_after": float(ari(mb, ma)), "same_modal_share": float((mb == ma).mean())}
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "before_after.csv", index=False)
    pd.DataFrame(sw_rows).to_csv(OUT / "switches_by_month.csv")
    # синтетика
    syn = pd.read_csv(data.ROOT / "outputs/synthetic/synthetic_results.csv")
    g = syn.groupby(["s_e", "method"]).mean(numeric_only=True)
    syn_rows = {}
    for sc in ["calibrated", "noise1", "noise2"]:
        for m in ["kefrin_t_affect_fe_z2.5", "kefrin_t_affect_fe_z2.5_2pass", "kefrin_t_affect_fe_z2.5_ma3", "kefrin_t_affect_fe_z2.5_2pass_ma3"]:
            if (sc, m) in g.index:
                x = g.loc[(sc, m)]
                syn_rows[f"{sc}/{m}"] = {"nmi": float(x.nmi_mean), "movers": [float(x["movers_correct_+0"]), float(x["movers_correct_+1"]), float(x["movers_correct_+2"])], "false": float(x.false_switch_at_event)}
    summary["synthetic"] = syn_rows
    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    make_figure(sw_rows)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "synthetic"}, ensure_ascii=False))


def make_figure(sw_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
    for ax, tag, title in [(axes[0], "K4", "Макротипы (K = 4)"), (axes[1], "K6", "Подтипы (K = 6)")]:
        b, a = sw_rows[f"{tag}_before"], sw_rows[f"{tag}_after"]
        x = np.arange(len(b))
        ax.bar(x - 0.2, b.to_numpy(), width=0.4, color="#b9b8b3", label="до: старт по первым 3 месяцам")
        ax.bar(x + 0.2, a.to_numpy(), width=0.4, color="#2a78d6", label="после: второй проход")
        ax.set_xticks(x[::2], [c for c in b.index[::2]], rotation=45, fontsize=8, ha="right")
        ax.set_ylabel("число смен типа за месяц")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", color="#eeede8")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_cleanup.png", dpi=110, bbox_inches="tight", facecolor="white")
    print("рисунок:", FIG / "fig6_cleanup.png")


if __name__ == "__main__":
    main()
