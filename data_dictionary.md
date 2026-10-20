# Data Dictionary — IPEDS Outcomes Lab

**Panel:** `data/processed/ipeds_panel.parquet` — one row per institution
(`unitid`) per fall year (`year`), 2015–2023, 60,885 rows × 74 columns,
8,670 distinct institutions. Built by `src/02_build_panel.py` from NCES
IPEDS Data Center bulk files (see *Provenance*). Rates are percents (0–100)
unless stated otherwise.

## Alignment convention (read this first)

IPEDS surveys do not share a calendar. The panel year `Y` means **fall Y**,
and each component is mapped to the fall it describes:

| Component | File used for panel year Y | What it describes |
|---|---|---|
| HD (directory) | `HD{Y}` | Institution as of fall Y |
| Admissions | `ADM{Y}` | Fall Y entering admissions cycle |
| Fall Enrollment A | `EF{Y}A` | Fall Y enrollment snapshot |
| Fall Enrollment D | `EF{Y}D` | Retention of the fall Y−1 cohort; cohort sizes |
| Graduation Rates | `GR{Y}` | Cohort that entered fall **Y−6**, tracked to Y (150% window) |
| Student Financial Aid | `SFA{yy}{yy+1}` | Academic year starting fall Y |
| Finance | `F{yy}{yy+1}_F1A/F2/F3` | Fiscal year starting fall Y |
| 12-Month Enrollment | `EFIA{Y+1}` | The 12 months of academic year Y/Y+1 |
| Outcome Measures | `OM{Z}` (separate table) | Cohort that entered fall **Z−8**, tracked 8 years |

Consequence: a row mixes measurement windows by design — the graduation
rate in row Y describes students who entered in Y−6, while the admission
rate describes fall Y applicants. The panel models *institutional
profiles*, not single cohorts. This is stated in the README's limitations.

## Variables

### Identity & characteristics (HD)
| Variable | Definition / construction |
|---|---|
| `unitid` | IPEDS institution ID (stable key). |
| `institution_name`, `city`, `state` | `INSTNM`, `CITY`, `STABBR` for fall Y. |
| `sector` | `SECTOR` recoded: 1 Public 4-yr, 2 Private nonprofit 4-yr, 3 Private for-profit 4-yr, 4/5/6 = 2-year analogs, 7/8/9 = less-than-2-year. |
| `control` | Public / Private nonprofit / Private for-profit (from `SECTOR`). |
| `four_year` | 1 if `sector` ∈ {1,2,3} **and** degree-granting (`DEGGRANT`=1) **and** bachelor's offered (`HLOFFER` ≥ 5). The analysis population. |
| `carnegie_group`, `carnegie_basic`, `carnegie_vintage` | Basic Classification coalesced per row: `C21BASIC` → `C18BASIC` → `C15BASIC` → `CARNEGIE`; vintage recorded so classification changes are visible, not silent. Groups: Doctoral (15–17), Master's (18–20), Baccalaureate (21–23), Associate's (24–27), Special Focus (28–31), Tribal (32). |
| `locale_code` | `LOCALE` (urbanicity). |
| `hbcu`, `land_grant` | `HBCU`, `LANDGRNT` flags (1/0 after IPEDS −1/−2 recode). |

### Admissions (ADM)
| Variable | Definition |
|---|---|
| `applicants`, `admissions`, `enrolled_ft` | `APPLCN`, `ADMSSN`, `ENRLFT`. |
| `admission_rate` | `admissions / applicants × 100`. Only institutions with selective admissions report ADM — expect ~61% coverage among four-year institutions, near-100% in the clustering population by construction. |
| `yield_rate` | `enrolled_ft / admissions × 100`. |
| `sat_math_mid`, `sat_ebrw_mid`, `sat_total_mid`, `act_mid` | Midpoints of the 25th/75th percentile scores of enrolled FTFT students: (`SATMT25`+`SATMT75`)/2 etc. `sat_total_mid` = math + EBRW midpoints. ⚠️ See *Definition changes*: SAT redesign. Coverage falls in the test-optional era (2021+). |

### Enrollment (EF-A)
| Variable | Definition |
|---|---|
| `total_enrollment` | Row `EFALEVEL`=1, `LINE`=29 ("all students, total") `EFTOTLT`. Headcount, fall snapshot — includes graduate and non-degree students. Validated: Harvard 2022 = 30,631. |
| `undergrad_enrollment`, `grad_enrollment` | `EFALEVEL`=2 / 12 rows (max per institution; files occasionally carry duplicate level rows). |
| `ug_share_black`, `ug_share_hispanic`, `ug_share_white`, `ug_share_asian` | Race/ethnicity counts on the undergraduate row (`EFBKAAT`, `EFHISPT`, `EFWHITT`, `EFASIAT`) ÷ undergraduate total × 100. |
| `grad_share_of_total` | `grad_enrollment / total_enrollment × 100` (derived). |

### Retention & entering cohort (EF-D)
| Variable | Definition |
|---|---|
| `retention_ft`, `retention_pt` | `RET_PCF` / `RET_PCP`: % of the prior fall's full-/part-time first-time cohort still enrolled (or completed) the next fall. |
| `entering_cohort_ftft` | `GRCOHRT`: revised full-time, first-time cohort size. |
| `student_faculty_ratio` | `STUFACR`. |

> **Excluded on purpose:** EF-D also ships `PGRCOHRT`. Its magnitudes are
> irreconcilable with a "Pell recipients in cohort" reading (Harvard: 60,
> against ~366 full-time first-time Pell recipients in SFA), and the exact
> definition could not be verified against the survey materials at build
> time. It is **not** carried into the panel; Pell exposure is measured
> with the SFA published percentages below. Shipping a variable whose
> meaning is guessed is worse than shipping one fewer variable.

### Graduation rates (GR) — computed from raw cells
IPEDS does not publish a simple rate column; rates here are **rebuilt from
the cell structure** of the GR file:
- Bachelor's section (`SECTION`=2): adjusted cohort = (`GRTYPE` 6,
  `CHRTSTAT` 10); completers ≤150% = (12, 16); completers ≤100% = (13, 17).
- Institutions reporting only Section 1 fall back to (2, 12) / (3, 13) /
  (14, 14).
- `grad_rate_150` = completers150 / cohort × 100; `grad_rate_100` likewise.
  `gr_cohort_year` = Y−6 records which entering cohort the rate describes.
- Race-specific rates (`grad_rate_150_black`, `_hispanic`, `_white`,
  `_asian`, `_aian`, `_nhpi`, `_two_or_more`, `_nonresident`, `_unknown`)
  use the same cells read on the race columns (`GRBKAAT`, …).
- **Suppression rule:** where completers exceed the adjusted cohort (a
  reporting artifact concentrated in micro-cohorts — median cohort size 2),
  the *rate* is set to NaN and the *counts* are kept. 209 institution-years
  suppressed across 2015–2023 (count by year in `outputs/coverage.json`).
- Validation: Harvard GR2022 rebuilt rate = 97.8%, matching IPEDS'
  published figure; the time-band cells sum exactly to the 150% total in
  both 2022 and 2023 files.

### Student aid (SFA)
| Variable | Definition |
|---|---|
| `pell_share_ug` | `UPGRNTP`: % of **all undergraduates** receiving a Pell Grant. The headline Pell-exposure measure. |
| `pell_share_ftft_aid` | `PGRNT_P`: % of full-time, first-time students receiving Pell. |
| `pell_ftft_n`, `pell_ug_n` | `PGRNT_N`, `UPGRNTN` counts. |
| `any_aid_share_ftft`, `federal_aid_share_ftft` | `ANYAIDP`, `FGRNT_P`. |

### Finance (F1A/F2/F3) & instructional intensity
Institutions report on exactly one form by accounting standard: GASB
(publics, F1A), FASB (nonprofits, F2), for-profits (F3).
| Variable | Definition |
|---|---|
| `instruction_expense` | Instruction function total, coalesced: F1A `F1C011` (GASB current-funds matrix) → F2 `F2E011` → F3 `F3E011`. Validated: UNCW FY-start-2022 = $152.1M; Harvard = $1,307.6M. ⚠️ The single-digit F1A columns (`F1E0x`) in recent files are *not* the functional totals — they sum to roughly half of total expenses and are not used. |
| `fte_12mo` | 12-month FTE enrollment from EFIA (`FTE12MNTH` where present, else `EFTEUG`+`EFTEGD`; source per year logged in `outputs/coverage.json` — for 2015–2023 the two-part sum was used throughout). |
| `instruction_per_fte` | `instruction_expense / fte_12mo` — instructional spending intensity per FTE student. |

### Equity table — `data/processed/om_equity.parquet` (from OM)
One row per institution per OM survey year. Cohort codes `OMCHRT`: tens
digit = cohort type (1x full-time first-time used here), units digit =
0 all / 1 Pell / 2 non-Pell (verified: Pell + non-Pell adjusted cohorts sum
to the all-student cohort). `award_rate_{4,6,8}_{pell,nonpell,all}` =
`OMAWDN{h}` / `OMACHRT` × 100 recomputed from counts (IPEDS' own
`OMAWDP*` percents are rounded). `gap_{4,6,8}_pp` = non-Pell − Pell rate.
`cohort_entry_year` = survey year − 8.

## Definition changes & known breaks (the honest part)

1. **SAT redesign (2016–17).** The fall-2015 cycle reports the old SAT;
   2017+ the redesigned SAT (higher scores on the same students). Visible
   in the data: median `sat_total_mid` jumps 1,028 (2016) → 1,100 (2017).
   Cross-2016 SAT comparisons are not like-for-like.
2. **Outcome Measures pilot (2016).** OM2016 used cohort codes 1–4 with
   **no Pell subcohort split**. The Pell/non-Pell breakdown starts with
   OM2017 — the equity analysis therefore spans survey years 2017–2023
   (cohorts entering fall 2009–2015).
3. **Finance/FASB-GASB forms.** One institution, one form per year; a
   small number of institutions switch standards across the panel (rare,
   but the coalesce order F1A → F2 → F3 handles it deterministically).
4. **Carnegie vintages.** Classifications were revised in 2015, 2018 and
   2021 editions; the panel takes each year's own vintage and records it
   (`carnegie_vintage`) instead of back-casting today's labels.
5. **Missing 2023 aid & finance.** At build time (Oct 2026) NCES had not
   yet posted `SFA2324` or the `F2324_*` finance files, so panel year 2023
   has no `pell_share_*`, `instruction_expense` or `instruction_per_fte`.
   OM2024 likewise does not exist yet. Rerunning `src/01_download.py` +
   `src/02_build_panel.py` will fill these in when NCES posts them.
6. **COVID cohorts.** The GR2023 100%-rate describes the fall-2017 cohort
   whose senior year was 2020–21: on-time completion drops sharply
   (Harvard 86.8% → 65.8%) while its 150% rate holds at 97.3% — students
   took leaves and finished late rather than not at all. Real effect, not
   a data error; the band cells sum exactly.
7. **Test-optional era.** SAT/ACT reporter coverage thins after 2020;
   treat score variables as "among reporters."

## Provenance

- **Source:** NCES IPEDS Data Center bulk data files,
  `https://nces.ed.gov/ipeds/datacenter/data/{SURVEY}.zip`, accessed
  **October 19–20, 2026**. 94 of 99 requested files were available (see
  break #5). The Urban Institute Education Data Portal API was checked
  first and was returning 404/500 errors on all IPEDS endpoints at build
  time, so the NCES originals are the source of record.
- IPEDS imputation-flag columns (`X`-prefixed) are dropped at read time;
  suppressed/derived flags (−1, −2) are recoded to NaN.
