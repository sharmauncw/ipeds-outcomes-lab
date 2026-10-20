"""03 — Exploratory analysis of the IPEDS panel (four-year institutions).

Produces publication-quality charts in outputs/charts/ and appends headline
EDA statistics to outputs/metrics.json. Every number written to metrics.json
is computed here from the executed panel — the README quotes this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CHART_DIR, GOLD, GREY, NAVY, OUT_DIR, PROCESSED_DIR, TEAL  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#B0BEC5",
    "axes.labelcolor": NAVY, "text.color": NAVY, "xtick.color": GREY,
    "ytick.color": GREY, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 150,
})
METRICS_PATH = OUT_DIR / "metrics.json"


def save(fig, name: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  chart: {name}")


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "ipeds_panel.parquet")
    p4 = panel[panel["four_year"] == 1].copy()
    metrics: dict = {"eda": {}}
    eda = metrics["eda"]
    eda["panel_rows_total"] = int(len(panel))
    eda["institutions_total"] = int(panel["unitid"].nunique())
    eda["four_year_rows"] = int(len(p4))
    eda["four_year_institutions_latest"] = int(p4[p4.year == 2023]["unitid"].nunique())

    # ---- trend medians by year -------------------------------------------
    trend = p4.groupby("year")[["grad_rate_150", "retention_ft", "admission_rate"]].median()
    eda["median_grad_rate_150_by_year"] = {int(k): round(v, 1) for k, v in trend["grad_rate_150"].items()}
    eda["median_retention_by_year"] = {int(k): round(v, 1) for k, v in trend["retention_ft"].items()}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(trend.index, trend["grad_rate_150"], marker="o", color=NAVY, lw=2.2, label="Median graduation rate (150%)")
    ax.plot(trend.index, trend["retention_ft"], marker="s", color=TEAL, lw=2.2, label="Median first-year retention")
    ax.plot(trend.index, trend["admission_rate"], marker="^", color=GOLD, lw=2.2, label="Median admission rate")
    ax.set_ylabel("Percent"); ax.set_xlabel("Fall year (panel)")
    ax.set_title("Completion drifts up; admissions loosen (median admit rate 70% → 78%)")
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "trend_medians.png")

    # ---- 2022 cross-section relationships ---------------------------------
    cs = p4[p4.year == 2022].copy()
    corr_vars = ["admission_rate", "sat_total_mid", "retention_ft",
                 "pell_share_ug", "instruction_per_fte", "student_faculty_ratio"]
    eda["correlations_with_grad_rate_2022"] = {
        v: round(float(cs[[v, "grad_rate_150"]].corr().iloc[0, 1]), 3)
        for v in corr_vars if cs[v].notna().sum() > 100
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    plot_df = cs.dropna(subset=["admission_rate", "grad_rate_150", "total_enrollment"])
    colors = {"Public": NAVY, "Private nonprofit": TEAL, "Private for-profit": GOLD}
    for ctrl, sub in plot_df.groupby("control"):
        ax.scatter(sub["admission_rate"], sub["grad_rate_150"],
                   s=np.sqrt(sub["total_enrollment"]) * 1.6, alpha=0.45,
                   color=colors.get(ctrl, GREY), label=ctrl, edgecolors="none")
    for uid, label in ((166027, "Harvard"), (199218, "UNC Wilmington"), (152567, "Trine")):
        r = plot_df[plot_df.unitid == uid]
        if len(r):
            ax.annotate(label, (r.admission_rate.iloc[0], r.grad_rate_150.iloc[0]),
                        fontsize=9, fontweight="bold", xytext=(6, 6),
                        textcoords="offset points")
    ax.set_xlabel("Admission rate (%)"); ax.set_ylabel("Graduation rate, 150% (%)")
    ax.set_title("Selectivity vs completion, fall 2022 (dot size = enrollment)")
    ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "scatter_selectivity.png")

    # ---- graduation-rate distribution by control ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharex=True, sharey=True)
    med = {}
    for ax, ctrl in zip(axes, ["Public", "Private nonprofit", "Private for-profit"]):
        vals = cs.loc[cs.control == ctrl, "grad_rate_150"].dropna()
        med[ctrl] = round(float(vals.median()), 1)
        ax.hist(vals, bins=20, color=colors[ctrl], alpha=0.85)
        ax.axvline(vals.median(), color=NAVY, ls="--", lw=1.2)
        ax.set_title(f"{ctrl}\nmedian {vals.median():.0f}%  (n={len(vals)})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Institutions"); axes[1].set_xlabel("Graduation rate, 150% (%)")
    fig.suptitle("Completion is a different sport by sector", fontweight="bold", color=NAVY)
    save(fig, "hist_grad_by_control.png")
    eda["median_grad_rate_2022_by_control"] = med

    # ---- race/ethnicity completion medians over time -----------------------
    race_cols = {"grad_rate_150_white": "White", "grad_rate_150_black": "Black",
                 "grad_rate_150_hispanic": "Hispanic", "grad_rate_150_asian": "Asian"}
    stable = p4[(p4.gr_cohort_adj >= 100)]  # suppress noisy micro-cohorts
    fig, ax = plt.subplots(figsize=(9, 4.6))
    race_palette = {"White": NAVY, "Black": GOLD, "Hispanic": TEAL, "Asian": "#7FB3B0"}
    race_medians = {}
    for col, label in race_cols.items():
        s = stable.groupby("year")[col].median()
        race_medians[label] = {int(k): round(v, 1) for k, v in s.items()}
        ax.plot(s.index, s.values, marker="o", lw=2.0, label=label, color=race_palette[label])
    ax.set_ylabel("Median graduation rate, 150% (%)"); ax.set_xlabel("Panel year (cohort entered 6 yrs earlier)")
    ax.set_title("Completion gaps by race/ethnicity persist across cohorts")
    ax.legend(frameon=False, fontsize=9, ncol=4); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "race_gap_trend.png")
    eda["race_medians_by_year"] = race_medians
    latest = {k: v.get(2023) for k, v in race_medians.items()}
    eda["race_gap_2023_pp"] = {
        "white_minus_black": round(latest["White"] - latest["Black"], 1),
        "white_minus_hispanic": round(latest["White"] - latest["Hispanic"], 1),
    }

    # ---- resources vs outcomes ---------------------------------------------
    spend = cs.dropna(subset=["instruction_per_fte", "grad_rate_150"])
    spend = spend[spend.instruction_per_fte > 0]
    spend["log_spend"] = np.log10(spend["instruction_per_fte"])
    bins = pd.qcut(spend["log_spend"], 12)
    binned = spend.groupby(bins, observed=True).agg(
        spend=("instruction_per_fte", "median"), rate=("grad_rate_150", "median")).dropna()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.scatter(spend["instruction_per_fte"], spend["grad_rate_150"], s=10, alpha=0.30, color=TEAL, edgecolors="none")
    ax.plot(binned["spend"], binned["rate"], color=NAVY, lw=2.4, marker="o", label="Median by spend band")
    ax.set_xscale("log")
    ax.set_xlabel("Instructional spending per FTE student ($, log scale)")
    ax.set_ylabel("Graduation rate, 150% (%)")
    ax.set_title("Money helps — with steeply diminishing returns")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "spend_vs_outcomes.png")
    eda["median_instruction_per_fte_2022"] = round(float(spend["instruction_per_fte"].median()), 0)
    eda["n_2022_cross_section"] = int(len(cs))

    # COVID-cohort note: 4-year (100%) rate for the fall-2017 entering cohort
    both = p4.dropna(subset=["grad_rate_100", "grad_rate_150"])
    medians = both.groupby("year")[["grad_rate_100", "grad_rate_150"]].median()
    eda["median_grad_rate_100_by_year"] = {int(k): round(v, 1) for k, v in medians["grad_rate_100"].items()}

    existing = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    existing.update(metrics)
    METRICS_PATH.write_text(json.dumps(existing, indent=2))
    print(json.dumps(eda, indent=2)[:1200])


if __name__ == "__main__":
    main()
