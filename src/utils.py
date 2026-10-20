"""Shared helpers for the IPEDS Outcomes Lab pipeline.

All scripts run from the repo root with the project venv:
    .venv/bin/python src/01_download.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
STAR_DIR = ROOT / "data" / "star_schema"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"

NCES_BASE = "https://nces.ed.gov/ipeds/datacenter/data"

# Panel years = fall (academic-year-start) years.
PANEL_YEARS = list(range(2015, 2024))
# Outcome Measures survey years (collection/spring year of the file).
OM_YEARS = list(range(2016, 2025))


def ay_suffix(year: int) -> str:
    """Two-digit suffix for the spring year of the AY starting in `year`."""
    return f"{(year + 1) % 100:02d}"


def file_stems_for_year(year: int) -> list[str]:
    """IPEDS Data Center file stems feeding one panel (fall) year.

    Alignment convention (documented in data_dictionary.md):
      * HD / ADM / EF-A / EF-D / GR  -> the fall-`year` collections
      * SFA{yy}{yy+1}                -> aid awarded in the AY starting fall `year`
      * F{yy}{yy+1}_F1A/F2/F3        -> fiscal year of that same AY
      * EFIA{year+1}                 -> 12-month activity of that same AY
    (NCES names the SFA/Finance files with two-digit years, e.g. SFA2122.)
    """
    yy = f"{year % 100:02d}"
    yy_next = ay_suffix(year)
    return [
        f"HD{year}", f"ADM{year}", f"EF{year}A", f"EF{year}D", f"GR{year}",
        f"EFIA{year + 1}",
        f"SFA{yy}{yy_next}",
        f"F{yy}{yy_next}_F1A", f"F{yy}{yy_next}_F2", f"F{yy}{yy_next}_F3",
    ]


def read_ipeds_csv(stem: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read the (single) CSV inside a downloaded IPEDS zip.

    Column names are stripped of whitespace (some IPEDS files ship trailing
    spaces, e.g. ADM's 'ACTMT75   ') and upper-cased for consistent mapping.
    `usecols` filters by the *stripped* names; missing requested columns are
    tolerated (returned frame simply lacks them) so per-year schema drift is
    visible to callers instead of crashing the read.
    """
    zpath = RAW_DIR / f"{stem}.zip"
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError(f"No CSV inside {zpath}")
        with zf.open(names[0]) as fh:
            raw = fh.read()
    # IPEDS files are latin-1 in some years; errors='replace' guards the rest.
    # NOTE: some IPEDS headers carry trailing spaces ('ACTMT75   '), so match
    # requested columns through a callable on the stripped name — pandas
    # validates a plain list against the raw header and would reject them.
    if usecols is not None:
        wanted = {str(c).strip().upper() for c in usecols}
        selector = lambda c: str(c).strip().upper() in wanted  # noqa: E731
    else:
        selector = None
    df = pd.read_csv(
        io.BytesIO(raw),
        encoding_errors="replace",
        low_memory=False,
        usecols=selector,
    )
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def drop_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Remove IPEDS imputation-flag columns (X-prefixed) in place-safe way."""
    return df[[c for c in df.columns if not c.startswith("X")]]


def write_json(obj: dict, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


# House palette (matches Akanksha's other repos)
NAVY = "#0A2A4A"
TEAL = "#0E7C7B"
GOLD = "#D9A441"
TEAL_DARK = "#0B5F5E"
GREY = "#5A6B7B"
LIGHT = "#C9BFA8"
