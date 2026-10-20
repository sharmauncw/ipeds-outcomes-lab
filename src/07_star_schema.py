"""07 — Power BI-ready star schema export.

Grain decisions (the part BI teams actually argue about):
  * fact_institution_year — one row per institution per fall year, measures
    only (rates, counts, dollars). This is the panel, conformed.
  * fact_equity_om — one row per institution per Outcome Measures survey
    year, Pell / non-Pell award outcomes. Kept as its own fact because OM
    cohorts and windows do not align 1:1 with the fall panel.
  * dim_institution — one row per institution, latest known attributes
    (a Type-1 dimension; history lives in the fact tables).
  * dim_year — calendar of panel years with cohort annotations.
Exports CSV (Power BI's lingua franca) under data/star_schema/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROCESSED_DIR, STAR_DIR  # noqa: E402


def main() -> None:
    STAR_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROCESSED_DIR / "ipeds_panel.parquet")

    dim_cols = ["unitid", "institution_name", "city", "state", "sector", "control",
                "carnegie_group", "carnegie_basic", "locale_code", "hbcu",
                "land_grant", "four_year"]
    dim = (panel.sort_values("year").groupby("unitid").tail(1)[dim_cols]
           .sort_values("unitid"))
    span = panel.groupby("unitid")["year"].agg(["min", "max"]).rename(
        columns={"min": "first_panel_year", "max": "last_panel_year"})
    dim = dim.merge(span, on="unitid", how="left")
    dim.to_csv(STAR_DIR / "dim_institution.csv", index=False)

    fact_cols = [c for c in panel.columns if c not in
                 ("institution_name", "city", "state", "sector", "control",
                  "carnegie_group", "carnegie_basic", "carnegie_vintage",
                  "locale_code", "hbcu", "land_grant")]
    fact_all = panel[fact_cols].sort_values(["unitid", "year"])
    fact_all.to_parquet(STAR_DIR / "fact_institution_year.parquet", index=False)
    # The CSV is the four-year analysis population — the full-panel CSV would
    # be ~17 MB of mostly 2-year/less-than-2-year rows; the complete panel
    # ships as the parquet above and in data/processed/.
    fact = fact_all[panel.sort_values(["unitid", "year"])["four_year"] == 1]
    fact.to_csv(STAR_DIR / "fact_institution_year.csv", index=False)

    years = pd.DataFrame({"year": sorted(panel.year.unique())})
    years["gr_cohort_entry_year"] = years["year"] - 6
    years["note"] = ""
    years.loc[years.year == 2020, "note"] = "COVID-19 disruption year"
    years.loc[years.year == 2023, "note"] = (
        "Aid/finance not yet posted by NCES at build time")
    years.to_csv(STAR_DIR / "dim_year.csv", index=False)

    om_path = PROCESSED_DIR / "om_equity.parquet"
    if om_path.exists():
        om = pd.read_parquet(om_path)
        keep = ["unitid", "survey_year", "cohort_entry_year",
                "omachrt_pell", "omachrt_nonpell",
                "award_rate_4_pell", "award_rate_4_nonpell",
                "award_rate_6_pell", "award_rate_6_nonpell",
                "award_rate_8_pell", "award_rate_8_nonpell",
                "gap_4_pp", "gap_6_pp", "gap_8_pp"]
        om[[c for c in keep if c in om.columns]].to_csv(
            STAR_DIR / "fact_equity_om.csv", index=False)

    for f in sorted(STAR_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")
    print(f"dim_institution rows: {len(dim):,} | fact rows: {len(fact):,}")


if __name__ == "__main__":
    main()
