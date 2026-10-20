# Run Log — IPEDS Outcomes Lab (initial build)

**Build dates:** October 19–20, 2026 · **Environment:** Python 3.12.3 venv
(`requirements.txt` pins the exact library versions) · All numbers below are
copied from `outputs/metrics.json` / `outputs/coverage.json` as written by
the pipeline scripts during this run.

## Data source & access

- **Source of record:** NCES IPEDS Data Center bulk files,
  `https://nces.ed.gov/ipeds/datacenter/data/{SURVEY}.zip`, accessed
  Oct 19–20, 2026. The Urban Institute Education Data Portal API was tried
  first (documented plan A) and was unusable at build time — HTTP 404 on
  all `/api/v1/college-university/ipeds/*` endpoints and HTTP 500 on every
  `/api/colleges/ipeds/*` topic — so the pipeline reads the NCES originals.
- **Downloads:** 94 of 99 requested survey files present (~150 MB of zips,
  git-ignored; manifest in `outputs/download_manifest.json`). Not yet
  posted by NCES at build time: `F2324_F1A`, `F2324_F2`, `F2324_F3`,
  `SFA2324`, and `OM2024` (does not exist). Effect: panel year 2023 has no
  aid/finance measures; OM analysis ends at survey year 2023.

## Pipeline run

| Step | Script | Result |
|---|---|---|
| Download | `src/01_download.py` | 94/99 files in `data/raw/` |
| Panel build | `src/02_build_panel.py` | **60,885 rows × 74 cols**, 8,670 institutions, 2015–2023 → `data/processed/ipeds_panel.parquet` (5.7 MB) |
| EDA | `src/03_eda.py` | 6 charts; trend/correlation stats in `metrics.json` |
| Clustering | `src/04_clustering.py` | k = 4 (silhouette 0.199; k=3: 0.157, k=5: 0.169), n = 1,753 institutions (2022) |
| Model | `src/05_model.py` | Test (2021–2023) **R² 0.745, MAE 7.15 pp, RMSE 10.01 pp**; train n = 9,474, test n = 5,389 |
| Equity | `src/06_equity.py` | OM tidy table 25,176 rows (2017–2023); 3 charts |
| Star schema | `src/07_star_schema.py` | dim_institution 8,670 rows; fact CSV 27,005 rows (four-year) + full parquet 60,885 |

## Integrity checks performed

- **Spot validation** (`outputs/validation_panel_extract.csv`): Harvard
  2022 admission rate 3.2% (published 3.2%), graduation rate (150%) 97.8%
  (published ~98%), total enrollment 30,631 (published 30,631); UNCW
  graduation rate 74.2% (published band ~74–77%); Trine/Harvard finance
  figures cross-checked against the F2 form totals.
- **GR rebuild verified by construction:** time-band cells
  (≤100% + 101–150% bands) sum exactly to the 150% completer total in both
  the 2022 and 2023 files (Harvard GR2022: 1,440 + 127 + 55 = 1,622
  completers on a 1,659 adjusted cohort → 97.8%).
- **OM cohort arithmetic verified:** Pell + non-Pell adjusted cohorts equal
  the all-student cohort (e.g., UNCW OM2022: 524 + 1,828 = 2,352).
- **Suppression:** 209 institution-years with completers > adjusted cohort
  had their rates set to NaN (counts retained); per-year counts in
  `outputs/coverage.json` (`gr_over100_suppressed`).
- **EFIA FTE source:** the two-part sum (EFTEUG + EFTEGD) was used for all
  years 2015–2023 (`FTE12MNTH` not present in the files as downloaded);
  logged per year in `outputs/coverage.json`.
- **Missingness:** `outputs/missingness.csv` (per variable × year).

## Judgment calls made during the build (all documented in the dictionary)

1. **GR rates computed from raw cells** (SECTION 2 bachelor's cells with a
   Section-1 fallback) instead of trusting any precomputed field.
2. **Instruction expense** uses F1A `F1C011` (GASB current-funds matrix),
   not the single-digit `F1E0x` columns, which sum to ~half of total
   expenses in recent files.
3. **`PGRCOHRT` dropped** — semantics unverifiable and magnitudes
   inconsistent with a Pell-count reading; Pell measured via SFA instead.
4. **Clustering population** restricted to institutions with undergraduate
   programs that report admissions selectivity (n = 1,753); two broader
   variants were run and rejected (one manufactured a 1,809-institution
   mega-cluster from imputed values; one lost the selective tier entirely
   when selectivity was removed from the inputs).
5. **OM2016 excluded** from equity analysis — pilot-year coding with no
   Pell subcohort split (definition change, NCES-side).
6. **2020 excluded from model train and test** (pandemic-transition year);
   time-based split otherwise.

## Output inventory

- `data/processed/`: `ipeds_panel.parquet`, `om_equity.parquet`,
  `cluster_assignments_2022.csv`
- `data/star_schema/`: `dim_institution.csv`, `dim_year.csv`,
  `fact_institution_year.csv`, `fact_institution_year.parquet`,
  `fact_equity_om.csv`
- `outputs/`: `metrics.json`, `coverage.json`, `missingness.csv`,
  `download_manifest.json`, `validation_panel_extract.csv`,
  `cluster_profiles.csv`, `cluster_labels.json`, `shap_importance.csv`,
  `test_predictions.csv`, `charts/` (11 PNGs)
- Docs/code: `README.md`, `data_dictionary.md`, `app.py`, `src/01–07`,
  `dax/measures.dax`, `requirements.txt`, `LICENSE`

## Known shortfalls (stated plainly)

- **No R component.** R/Rscript is not installed on this machine, so the
  optional mixed-effects model was not built. The Python pipeline covers
  the full brief without it.
- **XGBoost was not installable** in this environment (package download
  failure); the model uses sklearn's HistGradientBoosting, which the
  brief explicitly allows and which SHAP's TreeExplainer supports
  natively.
- **Silhouette scores are modest** (best 0.199) — reported as-is; peer
  groups are useful segmentations, not natural kinds.
- **Panel year 2023 lacks aid/finance** until NCES posts those files;
  rerunning steps 01–02 will backfill.
- **Model is least accurate for for-profit and high-Pell institutions**
  (MAE 13.9 pp / 8.5 pp) — quantified in the README rather than smoothed
  over.
