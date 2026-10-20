"""04 — Peer-group clustering of four-year institutions (fall 2022).

Population: four-year, degree-granting institutions WITH undergraduate
programs (undergrad_enrollment > 0) that report admissions selectivity
(admission_rate present). Two deliberate exclusions:
  * Graduate-only professional schools (law, medicine) — their
    undergraduate profile variables are not missing at random, they are
    meaningless, and imputing them would manufacture a fake peer group.
  * Institutions that do not report admissions (mostly open-enrollment) —
    selectivity is a defining peer dimension, and median-imputing a
    variable with ~39% missingness would plant a fake "average" spike in
    the middle of the feature space. Peer groups therefore cover the
    selective-admissions sector, and the README says exactly that.

K-means over standardized institutional-profile features; k chosen by
silhouette score over k=3..8. Within this population the remaining
inputs are nearly complete (retention 96%, Pell share 99%); the one soft
spot is instructional spending (~79% reported), median-imputed with the
imputed share reported in metrics.json.
Outputs:
  data/processed/cluster_assignments_2022.csv  (unitid, cluster, profile vars)
  outputs/cluster_profiles.csv                 (per-cluster medians/counts)
  outputs/charts/cluster_pca.png               (PCA projection, by cluster)
Cluster *names* are assigned editorially in outputs/cluster_labels.json
(written after inspecting the profiles) — this script stays label-free.
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
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import CHART_DIR, OUT_DIR, PROCESSED_DIR  # noqa: E402

YEAR = 2022
FEATURES = [
    "log_enrollment", "admission_rate", "retention_ft", "pell_share_ug",
    "log_instruction_per_fte", "student_faculty_ratio", "grad_share_of_total",
    "ug_share_black", "ug_share_hispanic",
]
RANDOM_STATE = 42


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "ipeds_panel.parquet")
    cs = panel[(panel.year == YEAR) & (panel.four_year == 1)
               & (panel.undergrad_enrollment > 0)
               & panel.admission_rate.notna()].copy()
    cs["log_enrollment"] = np.log10(cs["total_enrollment"].clip(lower=1))
    cs["log_instruction_per_fte"] = np.log10(cs["instruction_per_fte"].clip(lower=1))
    n_start = len(cs)
    # Median imputation for profile features: dropping incomplete reporters
    # would discard ~half of all four-year institutions (small colleges
    # report finance/enrollment items spottily) and bias the peer groups
    # toward large, well-resourced institutions. Imputation shares are
    # reported per feature in metrics.json and the data dictionary.
    coverage = {f: round(float(cs[f].notna().mean()), 3) for f in FEATURES}
    medians = cs[FEATURES].median(numeric_only=True)
    complete = cs.copy()
    complete[FEATURES] = complete[FEATURES].fillna(medians)
    complete = complete.dropna(subset=["total_enrollment"])
    X = complete[FEATURES].to_numpy()
    Xs = StandardScaler().fit_transform(X)

    sil = {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xs)
        sil[k] = round(float(silhouette_score(Xs, km.labels_)), 3)
    best_k = max(sil, key=sil.get)
    km = KMeans(n_clusters=best_k, n_init=25, random_state=RANDOM_STATE).fit(Xs)
    complete = complete.assign(cluster=km.labels_)
    print(f"k selection (silhouette): {sil} -> k={best_k}, n={len(complete)} of {n_start}")

    # Profiles: medians of interpretable (unscaled) variables per cluster
    prof_vars = ["total_enrollment", "admission_rate", "sat_total_mid", "retention_ft",
                 "grad_rate_150", "pell_share_ug", "instruction_per_fte",
                 "student_faculty_ratio", "grad_share_of_total"]
    profiles = complete.groupby("cluster")[prof_vars].median().round(1)
    profiles["n_institutions"] = complete.groupby("cluster").size()
    profiles["share_public_pct"] = (
        complete.assign(is_pub=(complete.control == "Public").astype(int))
        .groupby("cluster")["is_pub"].mean().mul(100).round(1))
    # Exemplars = institutions closest to their cluster centroid (most
    # typical members), not the largest — size would misrepresent clusters
    # whose median member is small.
    centers = km.cluster_centers_
    dists = np.linalg.norm(Xs - centers[km.labels_], axis=1)
    complete = complete.assign(centroid_dist=dists)
    profiles["representative_members"] = (
        complete.sort_values("centroid_dist")
        .groupby("cluster")["institution_name"].apply(lambda s: "; ".join(s.head(4))))
    profiles.to_csv(OUT_DIR / "cluster_profiles.csv")

    complete[["unitid", "institution_name", "state", "control", "carnegie_group", "cluster"]
             + prof_vars].to_csv(PROCESSED_DIR / "cluster_assignments_2022.csv", index=False)

    # PCA projection chart
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Xs)
    coords = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    for c in range(best_k):
        m = complete["cluster"].to_numpy() == c
        ax.scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.65,
                   color=cmap(c), label=f"Cluster {c} (n={m.sum()})", edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.set_title(f"Peer groups among four-year institutions, {YEAR} (k-means, k={best_k})")
    ax.legend(frameon=False, fontsize=9, markerscale=1.6)
    ax.spines[["top", "right"]].set_visible(False)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_DIR / "cluster_pca.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  chart: cluster_pca.png")

    metrics_path = OUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["clustering"] = {
        "year": YEAR, "features": FEATURES, "n_institutions_clustered": int(len(complete)),
        "n_four_year_total": int(n_start), "feature_coverage": coverage,
        "silhouette_by_k": {str(k): v for k, v in sil.items()}, "chosen_k": int(best_k),
        "chosen_silhouette": sil[best_k],
        "cluster_sizes": {str(int(k)): int(v) for k, v in complete["cluster"].value_counts().sort_index().items()},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(profiles.to_string())


if __name__ == "__main__":
    main()
