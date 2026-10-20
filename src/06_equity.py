"""06 — Equity cut: Pell vs non-Pell completion in Outcome Measures.

The IPEDS Outcome Measures (OM) survey follows entering cohorts for 8 years
and reports award attainment separately for Pell Grant recipients and
non-recipients — the cleanest federal Pell-gap measure available (unlike
the Graduation Rates survey, OM covers part-time and transfer-in students
too; here we use the full-time, first-time cohort for comparability).

OM cohort codes: OMCHRT 10 = FTFT all, 11 = FTFT Pell, 12 = FTFT non-Pell.
(Verified by construction: Pell + non-Pell adjusted cohorts sum to the
all-student cohort in the raw files.) Rates are recomputed from counts
(OMAWDN{horizon} / OMACHRT), not copied from IPEDS' rounded percents.

Outputs: data/processed/om_equity.parquet, equity charts, metrics.json.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    CHART_DIR, GOLD, NAVY, OM_YEARS, OUT_DIR, PROCESSED_DIR, TEAL,
    drop_flags, read_ipeds_csv,
)

COHORTS = {10: "all", 11: "pell", 12: "nonpell"}


def build_om() -> pd.DataFrame:
    frames = []
    cols = ["UNITID", "OMCHRT", "OMACHRT", "OMAWDN4", "OMAWDN6", "OMAWDN8"]
    for year in OM_YEARS:
        if year < 2017:
            # OM2016 (the pilot collection) coded cohorts 1-4 with NO Pell
            # subcohort split; the two-digit scheme incl. Pell (x1) and
            # non-Pell (x2) subcohorts starts with OM2017. Documented in
            # the data dictionary as a definition change.
            print(f"  OM{year}: pre-2017 coding (no Pell split), excluded by design")
            continue
        try:
            df = drop_flags(read_ipeds_csv(f"OM{year}", usecols=cols))
        except FileNotFoundError:
            print(f"  OM{year}: not available, skipped")
            continue
        df = df[df["OMCHRT"].isin(COHORTS)]
        df["survey_year"] = year
        frames.append(df)
    om = pd.concat(frames, ignore_index=True)
    for c in ["OMACHRT", "OMAWDN4", "OMAWDN6", "OMAWDN8"]:
        om[c] = pd.to_numeric(om[c], errors="coerce")
    om["group"] = om["OMCHRT"].map(COHORTS)
    om["cohort_entry_year"] = om["survey_year"] - 8  # 8-year OM window
    wide = om.pivot_table(index=["UNITID", "survey_year", "cohort_entry_year"],
                          columns="group",
                          values=["OMACHRT", "OMAWDN4", "OMAWDN6", "OMAWDN8"],
                          aggfunc="first")
    wide.columns = [f"{v.lower()}_{g}" for v, g in wide.columns]
    wide = wide.reset_index().rename(columns={"UNITID": "unitid"})
    for h in (4, 6, 8):
        for g in ("pell", "nonpell", "all"):
            wide[f"award_rate_{h}_{g}"] = np.where(
                wide[f"omachrt_{g}"] > 0,
                wide[f"omawdn{h}_{g}"] / wide[f"omachrt_{g}"] * 100, np.nan)
    wide["gap_8_pp"] = wide["award_rate_8_nonpell"] - wide["award_rate_8_pell"]
    wide["gap_6_pp"] = wide["award_rate_6_nonpell"] - wide["award_rate_6_pell"]
    wide["gap_4_pp"] = wide["award_rate_4_nonpell"] - wide["award_rate_4_pell"]
    return wide


def weighted_rate(df: pd.DataFrame, num: str, den: str) -> float:
    return float(df[num].sum() / df[den].sum() * 100)


def main() -> None:
    om = build_om()
    panel = pd.read_parquet(PROCESSED_DIR / "ipeds_panel.parquet")
    attrs = (panel.sort_values("year").groupby("unitid").tail(1)
             [["unitid", "institution_name", "state", "control", "sector",
               "carnegie_group", "four_year"]])
    om = om.merge(attrs, on="unitid", how="left")
    om.to_parquet(PROCESSED_DIR / "om_equity.parquet", index=False)
    print(f"OM tidy: {len(om):,} institution-survey rows, {om.survey_year.min()}-{om.survey_year.max()}")

    eq: dict = {"om": {}}
    # ---- national weighted gap trend --------------------------------------
    trend = []
    for year, g in om.groupby("survey_year"):
        valid = g.dropna(subset=["omachrt_pell", "omachrt_nonpell"])
        trend.append({
            "survey_year": int(year),
            "cohort_entry_year": int(year - 8),
            "pell_rate_8": weighted_rate(valid, "omawdn8_pell", "omachrt_pell"),
            "nonpell_rate_8": weighted_rate(valid, "omawdn8_nonpell", "omachrt_nonpell"),
            "n_institutions": int(len(valid)),
        })
    trend = pd.DataFrame(trend)
    trend["gap_pp"] = trend["nonpell_rate_8"] - trend["pell_rate_8"]
    eq["om"]["weighted_gap_trend"] = trend.round(2).to_dict("records")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(trend.survey_year, trend.pell_rate_8, marker="o", color=GOLD, lw=2.2, label="Pell recipients")
    ax.plot(trend.survey_year, trend.nonpell_rate_8, marker="s", color=NAVY, lw=2.2, label="Non-Pell")
    ax.fill_between(trend.survey_year, trend.pell_rate_8, trend.nonpell_rate_8, color=TEAL, alpha=0.15)
    ax.set_ylabel("Earned an award within 8 years (%)")
    ax.set_xlabel("Outcome Measures survey year (cohort entered 8 years earlier)")
    ax.set_title("The Pell completion gap: large, and barely moving")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / "equity_gap_trend.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  chart: equity_gap_trend.png")

    # ---- latest-year institution-level gaps --------------------------------
    latest_year = int(om.survey_year.max())
    lat = om[(om.survey_year == latest_year) & (om.omachrt_pell >= 50)
             & (om.omachrt_nonpell >= 50)].copy()
    eq["om"]["latest_survey_year"] = latest_year
    eq["om"]["latest_cohort_entry_year"] = latest_year - 8
    eq["om"]["inst_gap_median_pp"] = round(float(lat.gap_8_pp.median()), 1)
    eq["om"]["inst_gap_iqr_pp"] = [round(float(lat.gap_8_pp.quantile(0.25)), 1),
                                   round(float(lat.gap_8_pp.quantile(0.75)), 1)]
    eq["om"]["share_institutions_pell_ge_nonpell_pct"] = round(
        float((lat.gap_8_pp <= 0).mean() * 100), 1)
    eq["om"]["n_institutions_latest"] = int(len(lat))

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.hist(lat.gap_8_pp.clip(-30, 60), bins=36, color=TEAL, alpha=0.9)
    ax.axvline(0, color=NAVY, lw=1.4)
    ax.axvline(lat.gap_8_pp.median(), color=GOLD, lw=2, ls="--",
               label=f"Median gap {lat.gap_8_pp.median():.0f} pp")
    ax.set_xlabel("Non-Pell minus Pell 8-year award rate (percentage points)")
    ax.set_ylabel("Institutions")
    ax.set_title(f"Where the gap lives, institution by institution (OM {latest_year})")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(CHART_DIR / "equity_gap_distribution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  chart: equity_gap_distribution.png")

    # ---- does the gap close with more time? --------------------------------
    valid = om[om.survey_year == latest_year].dropna(
        subset=["omachrt_pell", "omachrt_nonpell"])
    horizons = {}
    for h in (4, 6, 8):
        horizons[h] = {
            "pell": weighted_rate(valid, f"omawdn{h}_pell", "omachrt_pell"),
            "nonpell": weighted_rate(valid, f"omawdn{h}_nonpell", "omachrt_nonpell"),
        }
    eq["om"]["weighted_rates_by_horizon_latest"] = {
        str(h): {k: round(v, 1) for k, v in d.items()} | {
            "gap_pp": round(d["nonpell"] - d["pell"], 1)} for h, d in horizons.items()}
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, [horizons[h]["pell"] for h in (4, 6, 8)], w, color=GOLD, label="Pell")
    ax.bar(x + w / 2, [horizons[h]["nonpell"] for h in (4, 6, 8)], w, color=NAVY, label="Non-Pell")
    ax.set_xticks(x, ["Within 4 years", "Within 6 years", "Within 8 years"])
    ax.set_ylabel("Award rate (%)")
    ax.set_title("More time helps everyone — the gap survives it")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(CHART_DIR / "equity_rates_by_horizon.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  chart: equity_rates_by_horizon.png")

    # ---- gap by sector / Carnegie ------------------------------------------
    lat4 = lat.merge(panel[["unitid"]].drop_duplicates(), on="unitid", how="inner")
    by_sector = (lat.groupby("control")["gap_8_pp"].agg(["median", "count"])
                 .round(1).to_dict("index"))
    eq["om"]["gap_by_control_latest"] = {str(k): {"median_pp": float(v["median"]),
                                                  "n": int(v["count"])}
                                         for k, v in by_sector.items()}
    by_carn = (lat.groupby("carnegie_group")["gap_8_pp"].agg(["median", "count"])
               .round(1).to_dict("index"))
    eq["om"]["gap_by_carnegie_latest"] = {str(k): {"median_pp": float(v["median"]),
                                                   "n": int(v["count"])}
                                          for k, v in by_carn.items()}

    # ---- spotlights ---------------------------------------------------------
    spots = {}
    for uid, name in ((199218, "UNC Wilmington"), (152567, "Trine"), (166027, "Harvard")):
        r = lat[lat.unitid == uid]
        if len(r):
            r = r.iloc[0]
            spots[name] = {
                "pell_rate_8": round(float(r.award_rate_8_pell), 1),
                "nonpell_rate_8": round(float(r.award_rate_8_nonpell), 1),
                "gap_pp": round(float(r.gap_8_pp), 1),
                "pell_cohort_n": int(r.omachrt_pell),
            }
    eq["om"]["spotlights_latest"] = spots

    metrics_path = OUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update(eq)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(eq, indent=2)[:1400])


if __name__ == "__main__":
    main()
