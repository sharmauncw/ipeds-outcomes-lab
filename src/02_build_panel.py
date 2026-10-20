"""02 — Build the harmonized IPEDS institutional panel, 2015-2023.

Reads the NCES bulk files downloaded by 01_download.py, extracts a harmonized
variable set per survey, and joins everything into one tidy panel:
data/processed/ipeds_panel.parquet (one row per institution per fall year).

Harmonization work performed here (all documented in data_dictionary.md):
  * Carnegie Basic classification coalesced across vintages
    (2021 > 2018 > 2015 > legacy) and grouped into broad levels.
  * Graduation rates rebuilt from the raw Graduation Rates survey cells
    (SECTION/GRTYPE/CHRTSTAT), not taken from any precomputed source.
  * Instructional expense unified across the three finance reporting forms
    (GASB F1A expense matrix F1C011; FASB F2E011; for-profit F3E011).
  * Aid/finance files aligned to the academic year starting in the panel
    (fall) year; 12-month FTE from EFIA of that same AY.

Outputs: panel parquet, outputs/missingness.csv, outputs/coverage.json,
outputs/validation_panel_extract.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    OUT_DIR, PANEL_YEARS, PROCESSED_DIR, drop_flags, read_ipeds_csv, write_json,
)

COVERAGE: dict = {"years": {}}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------- HD
CARNEGIE_GROUP = {}
for _c in (15, 16, 17):
    CARNEGIE_GROUP[_c] = "Doctoral"
for _c in (18, 19, 20):
    CARNEGIE_GROUP[_c] = "Master's"
for _c in (21, 22, 23):
    CARNEGIE_GROUP[_c] = "Baccalaureate"
for _c in (24, 25, 26, 27):
    CARNEGIE_GROUP[_c] = "Associate's"
for _c in (28, 29, 30, 31):
    CARNEGIE_GROUP[_c] = "Special Focus"
CARNEGIE_GROUP[32] = "Tribal"

SECTOR_LABEL = {
    1: "Public, 4-year", 2: "Private nonprofit, 4-year", 3: "Private for-profit, 4-year",
    4: "Public, 2-year", 5: "Private nonprofit, 2-year", 6: "Private for-profit, 2-year",
    7: "Public, less-than-2-year", 8: "Private nonprofit, less-than-2-year",
    9: "Private for-profit, less-than-2-year",
}
CONTROL_LABEL = {1: "Public", 2: "Private nonprofit", 3: "Private for-profit"}


def extract_hd(year: int) -> pd.DataFrame:
    stem = f"HD{year}"
    cols = ["UNITID", "INSTNM", "CITY", "STABBR", "FIPS", "SECTOR", "CONTROL",
            "ICLEVEL", "HLOFFER", "DEGGRANT", "LOCALE", "INSTSIZE", "HBCU",
            "LANDGRNT", "C21BASIC", "C18BASIC", "C15BASIC", "CARNEGIE"]
    df = read_ipeds_csv(stem, usecols=cols)
    out = pd.DataFrame({"unitid": df["UNITID"]})
    out["institution_name"] = df.get("INSTNM")
    out["city"] = df.get("CITY")
    out["state"] = df.get("STABBR")
    out["state_fips"] = _num(df.get("FIPS"))
    out["sector_code"] = _num(df.get("SECTOR"))
    out["control"] = _num(df.get("CONTROL")).map(CONTROL_LABEL)
    out["level_code"] = _num(df.get("ICLEVEL"))
    out["highest_degree"] = _num(df.get("HLOFFER"))
    out["degree_granting"] = _num(df.get("DEGGRANT"))
    out["locale_code"] = _num(df.get("LOCALE"))
    out["size_category"] = _num(df.get("INSTSIZE"))
    out["hbcu"] = _num(df.get("HBCU"))
    out["land_grant"] = _num(df.get("LANDGRNT"))
    out["sector"] = out["sector_code"].map(SECTOR_LABEL)
    # Carnegie: coalesce newest vintage available per row.
    carn = pd.Series(np.nan, index=df.index, dtype="float64")
    vintage = pd.Series(pd.NA, index=df.index, dtype="string")
    for col, tag in (("C21BASIC", "2021"), ("C18BASIC", "2018"),
                     ("C15BASIC", "2015"), ("CARNEGIE", "legacy")):
        if col in df.columns:
            vals = _num(df[col])
            take = carn.isna() & vals.notna() & (vals > 0)
            carn = carn.mask(take, vals)
            vintage = vintage.mask(take, tag)
    out["carnegie_basic"] = carn
    out["carnegie_vintage"] = vintage
    out["carnegie_group"] = carn.map(CARNEGIE_GROUP).fillna("Other/Not classified")
    out["four_year"] = ((out["level_code"] == 1) & (out["degree_granting"] == 1)).astype(int)
    return out.set_index("unitid")


# --------------------------------------------------------------------- ADM
def extract_adm(year: int) -> pd.DataFrame:
    stem = f"ADM{year}"
    cols = ["UNITID", "APPLCN", "ADMSSN", "ENRLT", "ENRLFT",
            "SATVR25", "SATVR75", "SATMT25", "SATMT75",
            "ACTCM25", "ACTCM75", "SATPCT", "ACTPCT"]
    df = drop_flags(read_ipeds_csv(stem, usecols=cols))
    out = pd.DataFrame({"unitid": df["UNITID"]})
    for c in cols[1:]:
        if c in df.columns:
            out[c.lower()] = _num(df[c])
    appl = out.get("applcn")
    adm = out.get("admssn")
    out["admission_rate"] = np.where((appl.notna()) & (appl > 0), adm / appl * 100, np.nan)
    out["yield_rate"] = np.where((adm.notna()) & (adm > 0), out.get("enrlt") / adm * 100, np.nan)
    if "satmt25" in out and "satmt75" in out:
        out["sat_math_mid"] = (out["satmt25"] + out["satmt75"]) / 2
    if "satvr25" in out and "satvr75" in out:
        out["sat_ebrw_mid"] = (out["satvr25"] + out["satvr75"]) / 2
    if "sat_math_mid" in out and "sat_ebrw_mid" in out:
        out["sat_total_mid"] = out["sat_math_mid"] + out["sat_ebrw_mid"]
    if "actcm25" in out and "actcm75" in out:
        out["act_mid"] = (out["actcm25"] + out["actcm75"]) / 2
    keep = ["unitid", "applcn", "admssn", "enrlt", "admission_rate", "yield_rate",
            "sat_math_mid", "sat_ebrw_mid", "sat_total_mid", "act_mid",
            "satpct", "actpct"]
    out = out.rename(columns={"satpct": "sat_submitters_pct", "actpct": "act_submitters_pct",
                              "applcn": "applicants", "admssn": "admissions",
                              "enrlt": "enrolled_new"})
    keep = [c.replace("satpct", "sat_submitters_pct").replace("actpct", "act_submitters_pct")
            .replace("applcn", "applicants").replace("admssn", "admissions")
            .replace("enrlt", "enrolled_new") for c in keep]
    return out[[c for c in keep if c in out.columns]].set_index("unitid")


# --------------------------------------------------------------------- EF A
RACE_COLS = {  # EF-A undergraduate row race/ethnicity count columns
    "EFAIANT": "ug_ai_an", "EFASIAT": "ug_asian", "EFBKAAT": "ug_black",
    "EFHISPT": "ug_hispanic", "EFNHPIT": "ug_nh_pi", "EFWHITT": "ug_white",
    "EF2MORT": "ug_two_plus", "EFUNKNT": "ug_unknown", "EFNRALT": "ug_nonresident",
}


def extract_efa(year: int) -> pd.DataFrame:
    stem = f"EF{year}A"
    cols = ["UNITID", "EFALEVEL", "LINE", "EFTOTLT"] + list(RACE_COLS)
    df = drop_flags(read_ipeds_csv(stem, usecols=cols))
    df["EFALEVEL"] = _num(df["EFALEVEL"])
    df["LINE"] = _num(df["LINE"])
    df["EFTOTLT"] = _num(df["EFTOTLT"])
    total = (df[(df["EFALEVEL"] == 1) & (df["LINE"] == 29)]
             .set_index("UNITID")["EFTOTLT"].rename("total_enrollment"))
    ug = df[df["EFALEVEL"] == 2].copy()
    ug = ug.sort_values("EFTOTLT").groupby("UNITID").tail(1).set_index("UNITID")
    out = pd.DataFrame(index=ug.index)
    out["undergrad_enrollment"] = ug["EFTOTLT"]
    for src, dst in RACE_COLS.items():
        if src in ug.columns:
            out[dst] = _num(ug[src])
    out = out.join(total, how="outer")
    out.index.name = "unitid"
    return out


# --------------------------------------------------------------------- EF D
def extract_efd(year: int) -> pd.DataFrame:
    stem = f"EF{year}D"
    # NOTE: EF-D also ships PGRCOHRT, whose magnitudes are inconsistent with a
    # Pell-recipient count (e.g. 60 for Harvard vs ~366 FTFT Pell recipients
    # in SFA) and which we could not verify against the IPEDS dictionary —
    # so it is deliberately NOT carried into the panel. Pell exposure is
    # measured with the SFA published percentages instead (see dictionary).
    cols = ["UNITID", "GRCOHRT", "RET_PCF", "RET_PCP", "STUFACR"]
    df = drop_flags(read_ipeds_csv(stem, usecols=cols))
    out = pd.DataFrame({"unitid": df["UNITID"]})
    out["entering_cohort_ftft"] = _num(df.get("GRCOHRT"))
    out["retention_ft"] = _num(df.get("RET_PCF"))
    out["retention_pt"] = _num(df.get("RET_PCP"))
    out["student_faculty_ratio"] = _num(df.get("STUFACR"))
    return out.set_index("unitid")


# --------------------------------------------------------------------- GR
GR_RACE = {  # GR race/ethnicity count columns -> panel suffix
    "GRBKAAT": "black", "GRHISPT": "hispanic", "GRWHITT": "white",
    "GRASIAT": "asian",
}


def _gr_cell(df: pd.DataFrame, section: int, grtype: int, chrtstat: int,
             value_col: str = "GRTOTLT") -> pd.Series:
    sub = df[(df["SECTION"] == section) & (df["GRTYPE"] == grtype)
             & (df["CHRTSTAT"] == chrtstat)]
    return sub.set_index("UNITID")[value_col]


def extract_gr(year: int) -> pd.DataFrame:
    """Rebuild bachelor's graduation rates from raw GR survey cells.

    Cell map (validated against published figures for Harvard/UNCW, see
    RUN_LOG.md). SECTION 2 is the bachelor's degree-seeking section:
      adjusted cohort      GRTYPE 6  / CHRTSTAT 10
      completers <=100%    GRTYPE 13 / CHRTSTAT 17
      completers <=150%    GRTYPE 12 / CHRTSTAT 16
    Fallback (institutions reporting only SECTION 1):
      adjusted cohort      GRTYPE 2  / CHRTSTAT 12
      completers <=150%    GRTYPE 3  / CHRTSTAT 13
    """
    stem = f"GR{year}"
    cols = ["UNITID", "GRTYPE", "CHRTSTAT", "SECTION", "GRTOTLT"] + list(GR_RACE)
    df = drop_flags(read_ipeds_csv(stem, usecols=cols))
    for c in ("GRTYPE", "CHRTSTAT", "SECTION", "GRTOTLT"):
        df[c] = _num(df[c])
    for c in GR_RACE:
        if c in df.columns:
            df[c] = _num(df[c])
    out = pd.DataFrame(index=sorted(df["UNITID"].unique()))
    out.index.name = "unitid"
    cohort = _gr_cell(df, 2, 6, 10)
    comp150 = _gr_cell(df, 2, 12, 16)
    comp100 = _gr_cell(df, 2, 13, 17)
    # Fallback where Section 2 is absent
    cohort_fb = _gr_cell(df, 1, 2, 12)
    comp150_fb = _gr_cell(df, 1, 3, 13)
    out["gr_cohort_adj"] = cohort.combine_first(cohort_fb)
    out["gr_completers_150"] = comp150.combine_first(comp150_fb)
    out["gr_completers_100"] = comp100
    for src, suffix in GR_RACE.items():
        if src in df.columns:
            c_race = _gr_cell(df, 2, 6, 10, src).reindex(out.index)
            n_race = _gr_cell(df, 2, 12, 16, src).reindex(out.index)
            rate = np.where((c_race.notna()) & (c_race > 0), n_race / c_race * 100, np.nan)
            rate = np.where(n_race > c_race, np.nan, rate)  # same >100% suppression
            out[f"grad_rate_150_{suffix}"] = rate
            out[f"gr_cohort_{suffix}"] = c_race
    out["grad_rate_150"] = np.where(
        out["gr_cohort_adj"] > 0, out["gr_completers_150"] / out["gr_cohort_adj"] * 100, np.nan)
    out["grad_rate_100"] = np.where(
        out["gr_cohort_adj"] > 0, out["gr_completers_100"] / out["gr_cohort_adj"] * 100, np.nan)
    # Reporting artifact: a small number of (almost exclusively micro-cohort)
    # institutions report more completers than adjusted-cohort members, which
    # would imply a rate >100%. Suppress those rates to NaN (counts are kept)
    # and log how many were suppressed — see data_dictionary.md.
    over = out["gr_completers_150"] > out["gr_cohort_adj"]
    COVERAGE.setdefault("gr_over100_suppressed", {})[year] = int(over.fillna(False).sum())
    out.loc[over, ["grad_rate_150", "grad_rate_100"]] = np.nan
    out["gr_cohort_year"] = year - 6  # bachelor's 150% window: entered fall year-6
    return out


# --------------------------------------------------------------------- SFA
def extract_sfa(year: int) -> pd.DataFrame:
    yy, yy_next = f"{year % 100:02d}", f"{(year + 1) % 100:02d}"
    stem = f"SFA{yy}{yy_next}"
    cols = ["UNITID", "UPGRNTP", "UPGRNTN", "PGRNT_P", "FGRNT_P", "ANYAIDP"]
    try:
        df = drop_flags(read_ipeds_csv(stem, usecols=cols))
    except FileNotFoundError:
        return pd.DataFrame(index=pd.Index([], name="unitid"))
    out = pd.DataFrame({"unitid": df["UNITID"]})
    out["pell_share_ug"] = _num(df.get("UPGRNTP"))
    out["pell_n_ug"] = _num(df.get("UPGRNTN"))
    out["pell_share_ftft_aid"] = _num(df.get("PGRNT_P"))
    out["fed_grant_share_ftft"] = _num(df.get("FGRNT_P"))
    out["any_aid_share_ftft"] = _num(df.get("ANYAIDP"))
    return out.set_index("unitid")


# ------------------------------------------------------------------ Finance
def extract_finance(year: int) -> pd.DataFrame:
    yy, yy_next = f"{year % 100:02d}", f"{(year + 1) % 100:02d}"
    frames = []
    for form, col in (("F1A", "F1C011"), ("F2", "F2E011"), ("F3", "F3E011")):
        stem = f"F{yy}{yy_next}_{form}"
        try:
            df = drop_flags(read_ipeds_csv(stem, usecols=["UNITID", col]))
        except FileNotFoundError:
            continue
        if col in df.columns:
            s = pd.DataFrame({"unitid": df["UNITID"],
                              "instruction_expense": _num(df[col]),
                              "finance_form": form})
            frames.append(s.set_index("unitid"))
    if not frames:
        return pd.DataFrame(index=pd.Index([], name="unitid"))
    allf = pd.concat(frames)
    # An institution reports on exactly one form in almost all cases; if a
    # duplicate slips through, keep the first (F1A > F2 > F3 order above).
    allf = allf[~allf.index.duplicated(keep="first")]
    return allf


# -------------------------------------------------------------------- EFIA
def extract_efia(year: int) -> pd.DataFrame:
    stem = f"EFIA{year + 1}"
    try:
        probe = read_ipeds_csv(stem, usecols=None)
    except FileNotFoundError:
        return pd.DataFrame(index=pd.Index([], name="unitid"))
    cols = list(probe.columns)
    out = pd.DataFrame({"unitid": probe["UNITID"]})
    if "FTE12MNTH" in cols:
        out["fte_12mo"] = _num(probe["FTE12MNTH"])
        COVERAGE.setdefault("efia_fte_source", {})[year] = "FTE12MNTH"
    elif "EFTEUG" in cols and "EFTEGD" in cols:
        out["fte_12mo"] = _num(probe["EFTEUG"]) + _num(probe["EFTEGD"])
        COVERAGE.setdefault("efia_fte_source", {})[year] = "EFTEUG+EFTEGD"
    else:
        out["fte_12mo"] = np.nan
        COVERAGE.setdefault("efia_fte_source", {})[year] = "NOT FOUND"
    return out.set_index("unitid")


# ------------------------------------------------------------------ assembly
def build_year(year: int) -> pd.DataFrame:
    parts = {
        "hd": extract_hd(year),
        "adm": extract_adm(year),
        "efa": extract_efa(year),
        "efd": extract_efd(year),
        "gr": extract_gr(year),
        "sfa": extract_sfa(year),
        "fin": extract_finance(year),
        "efia": extract_efia(year),
    }
    panel = parts["hd"]
    for name, part in parts.items():
        if name == "hd":
            continue
        panel = panel.join(part, how="left")
    panel["year"] = year
    # Derived measures (guard: a survey missing for a year yields no column)
    for _col in ("instruction_expense", "fte_12mo"):
        if _col not in panel.columns:
            panel[_col] = np.nan
    panel["instruction_per_fte"] = np.where(
        panel["fte_12mo"].notna() & (panel["fte_12mo"] > 0),
        panel["instruction_expense"] / panel["fte_12mo"], np.nan)
    for race, col in (("black", "ug_black"), ("hispanic", "ug_hispanic"),
                      ("white", "ug_white"), ("asian", "ug_asian")):
        if col in panel.columns:
            panel[f"ug_share_{race}"] = np.where(
                panel["undergrad_enrollment"] > 0,
                panel[col] / panel["undergrad_enrollment"] * 100, np.nan)
    panel["grad_share_of_total"] = np.where(
        panel["total_enrollment"] > 0,
        (panel["total_enrollment"] - panel["undergrad_enrollment"])
        / panel["total_enrollment"] * 100, np.nan)
    stats = {name: int(len(part)) for name, part in parts.items()}
    stats["panel_rows"] = int(len(panel))
    stats["four_year_rows"] = int(panel["four_year"].sum())
    sub = panel[panel["four_year"] == 1]
    stats["coverage_4yr"] = {
        "grad_rate_150": float(sub["grad_rate_150"].notna().mean()),
        "retention_ft": float(sub["retention_ft"].notna().mean()),
        "admission_rate": float(sub["admission_rate"].notna().mean()),
        "pell_share_ug": float(sub["pell_share_ug"].notna().mean()) if "pell_share_ug" in sub else 0.0,
        "instruction_per_fte": float(sub["instruction_per_fte"].notna().mean()),
    }
    COVERAGE["years"][year] = stats
    print(f"{year}: " + " | ".join(f"{k}={v}" for k, v in stats.items() if k != "coverage_4yr"))
    return panel.reset_index()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = [build_year(y) for y in PANEL_YEARS]
    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["unitid", "year"]).reset_index(drop=True)
    # Sanity: rates must live in [0, 100]
    for col in ["admission_rate", "yield_rate", "retention_ft", "retention_pt",
                "grad_rate_150", "grad_rate_100", "pell_share_ug", "pell_share_entering"]:
        if col in panel:
            bad = panel[(panel[col] < 0) | (panel[col] > 100)]
            if len(bad):
                print(f"  WARNING: {len(bad)} rows outside [0,100] in {col}; clipping to NaN")
                panel.loc[(panel[col] < 0) | (panel[col] > 100), col] = np.nan
    ppath = PROCESSED_DIR / "ipeds_panel.parquet"
    panel.to_parquet(ppath, index=False)
    print(f"\nPanel: {len(panel):,} rows x {panel.shape[1]} cols -> {ppath} "
          f"({ppath.stat().st_size / 1e6:.1f} MB)")
    # Missingness report
    miss = (panel.isna().mean().rename("share_missing").to_frame()
            .assign(n_missing=panel.isna().sum()).sort_values("share_missing"))
    miss.to_csv(OUT_DIR / "missingness.csv")
    # Validation extract: three institutions a reader can check by hand
    val_ids = {166027: "Harvard University", 199218: "UNC Wilmington", 152567: "Trine University"}
    vcols = ["unitid", "year", "institution_name", "admission_rate", "sat_total_mid",
             "total_enrollment", "undergrad_enrollment", "retention_ft",
             "grad_rate_100", "grad_rate_150", "pell_share_ug", "instruction_per_fte"]
    val = panel[panel["unitid"].isin(val_ids)][[c for c in vcols if c in panel.columns]]
    val.to_csv(OUT_DIR / "validation_panel_extract.csv", index=False)
    write_json(COVERAGE, "coverage.json")
    print("Wrote missingness.csv, coverage.json, validation_panel_extract.csv")


if __name__ == "__main__":
    main()
