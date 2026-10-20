"""05 — Predict the bachelor's graduation rate (150% of normal time).

Model: gradient-boosted trees (sklearn HistGradientBoostingRegressor).
Discipline:
  * TIME-BASED split — train on panel years <= 2019, test on 2021-2023.
    (2020 is excluded from both: a pandemic-transition collection year.)
  * Cohort-size floor: only institutions with adjusted bachelor's cohort
    >= 50 contribute — rates from micro-cohorts are noise, not signal.
  * Robustness: GroupKFold (by institution) CV on the training years, so
    the same institution never appears on both sides of a fold.
  * Explainability: SHAP values (TreeExplainer; falls back to a classic
    GradientBoostingRegressor if the backend rejects HistGB).
  * Fairness slices: test error broken out by control and by Pell-share
    tercile — a model that is only accurate for wealthy institutions is
    not a good model, and the README says so either way.

This is an institutional-PROFILE model: features are measured in the same
panel year the outcome is reported, not when the cohort entered. It answers
"what do high-completion institutions look like?" — it is NOT an
early-warning system and makes no causal claims.
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
import shap  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CHART_DIR, OUT_DIR, PROCESSED_DIR  # noqa: E402

TARGET = "grad_rate_150"
NUM_FEATURES = [
    "admission_rate", "sat_total_mid", "act_mid", "retention_ft",
    "pell_share_ug", "log_enrollment", "log_instruction_per_fte",
    "student_faculty_ratio", "ug_share_black", "ug_share_hispanic",
    "grad_share_of_total", "hbcu",
]
CAT_FEATURES = ["control", "carnegie_group"]
MIN_COHORT = 50


def prep(df: pd.DataFrame, medians: pd.Series | None) -> tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d["log_enrollment"] = np.log10(d["total_enrollment"].clip(lower=1))
    d["log_instruction_per_fte"] = np.log10(d["instruction_per_fte"].clip(lower=1))
    X = d[NUM_FEATURES + CAT_FEATURES]
    X = pd.get_dummies(X, columns=CAT_FEATURES, dummy_na=False)
    if medians is None:
        medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    return X, medians


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "r2": round(float(r2_score(y, pred)), 3),
        "mae_pp": round(float(mean_absolute_error(y, pred)), 2),
        "rmse_pp": round(float(np.sqrt(mean_squared_error(y, pred))), 2),
    }


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "ipeds_panel.parquet")
    d = panel[(panel.four_year == 1) & (panel.gr_cohort_adj >= MIN_COHORT)
              & panel[TARGET].notna()].copy()
    train = d[d.year <= 2019]
    test = d[d.year >= 2021]
    Xtr, medians = prep(train, None)
    Xte, _ = prep(test, medians)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    ytr, yte = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    print(f"train n={len(train)} (2015-2019) | test n={len(test)} (2021-2023)")

    # Baselines
    naive = np.full_like(yte, np.median(ytr), dtype=float)
    sector_mean = train.groupby("control")[TARGET].mean()
    base_sector = test["control"].map(sector_mean).fillna(np.median(ytr)).to_numpy()

    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    # Grouped CV on train (institutions disjoint across folds)
    gkf = GroupKFold(n_splits=5)
    cv_r2 = []
    for tr_idx, va_idx in gkf.split(Xtr, ytr, groups=train["unitid"]):
        m = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)
        m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        cv_r2.append(r2_score(ytr[va_idx], m.predict(Xtr.iloc[va_idx])))

    # SHAP — TreeExplainer on HistGB; fallback to classic GBM refit
    shap_model, shap_note = model, "HistGradientBoostingRegressor"
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xte)
    except Exception as exc:  # pragma: no cover - backend dependent
        print(f"  SHAP on HistGB failed ({exc.__class__.__name__}); refitting GradientBoostingRegressor")
        shap_model = GradientBoostingRegressor(random_state=42)
        shap_model.fit(Xtr, ytr)
        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(Xte)
        shap_note = "GradientBoostingRegressor (fallback for SHAP backend)"
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (pd.DataFrame({"feature": Xte.columns, "mean_abs_shap": mean_abs})
                  .sort_values("mean_abs_shap", ascending=False))
    importance.to_csv(OUT_DIR / "shap_importance.csv", index=False)

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, Xte, show=False, max_display=14)
    plt.savefig(CHART_DIR / "shap_beeswarm.png", bbox_inches="tight", facecolor="white")
    plt.close("all")
    shap.summary_plot(shap_values, Xte, plot_type="bar", show=False, max_display=14)
    plt.savefig(CHART_DIR / "shap_bar.png", bbox_inches="tight", facecolor="white")
    plt.close("all")
    print("  charts: shap_beeswarm.png, shap_bar.png")

    # Predictions + error analysis
    preds = test[["unitid", "institution_name", "state", "control", "year"]].copy()
    preds["actual"] = yte
    preds["predicted"] = np.round(pred, 1)
    preds["error_pp"] = np.round(pred - yte, 1)
    preds.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # Fairness slices on the test set
    fairness = {}
    for col in ["control"]:
        fairness[col] = {
            str(k): {"n": int(len(g)), "mae_pp": round(float(mean_absolute_error(g.actual, g.predicted)), 2)}
            for k, g in preds.groupby(test[col])}
    terc = pd.qcut(test["pell_share_ug"], 3, labels=["low Pell", "mid Pell", "high Pell"])
    fairness["pell_tercile"] = {
        str(k): {"n": int(len(g)), "mae_pp": round(float(mean_absolute_error(g.actual, g.predicted)), 2)}
        for k, g in preds.groupby(terc, observed=True)}

    results = {
        "target": TARGET, "min_cohort": MIN_COHORT,
        "train_years": "2015-2019", "test_years": "2021-2023 (2020 excluded)",
        "n_train": int(len(train)), "n_test": int(len(test)),
        "model": "HistGradientBoostingRegressor(500 iters, lr=0.06)",
        "shap_model": shap_note,
        "test_metrics": score(yte, pred),
        "baseline_median": score(yte, naive),
        "baseline_sector_mean": score(yte, base_sector),
        "grouped_cv_r2_train": [round(float(x), 3) for x in cv_r2],
        "grouped_cv_r2_mean": round(float(np.mean(cv_r2)), 3),
        "top_shap_features": importance.head(8).assign(
            mean_abs_shap=lambda x: x.mean_abs_shap.round(2)).to_dict("records"),
        "fairness_slices": fairness,
        "largest_overpredictions": preds.nlargest(5, "error_pp")[
            ["institution_name", "state", "year", "actual", "predicted"]].to_dict("records"),
        "largest_underpredictions": preds.nsmallest(5, "error_pp")[
            ["institution_name", "state", "year", "actual", "predicted"]].to_dict("records"),
    }
    metrics_path = OUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["model"] = results
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(results, indent=2)[:1600])


if __name__ == "__main__":
    main()
