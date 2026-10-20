"""IPEDS Outcomes Lab — interactive explorer.

Run:  streamlit run app.py
Reads the tidy panel, peer-group assignments and Outcome Measures equity
table produced by src/02, src/04 and src/06. No live downloads.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
NAVY, TEAL, GOLD, TEAL_DARK, GREY = "#0A2A4A", "#0E7C7B", "#D9A441", "#0B5F5E", "#5A6B7B"

st.set_page_config(page_title="IPEDS Outcomes Lab", layout="wide")
st.markdown(
    f"<style>.stApp h1,.stApp h2,.stApp h3{{color:{NAVY}}}"
    f"div[data-testid='stMetricValue']{{color:{TEAL_DARK}}}</style>",
    unsafe_allow_html=True)


@st.cache_data
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(BASE / "data" / "processed" / "ipeds_panel.parquet")


@st.cache_data
def load_clusters() -> pd.DataFrame:
    p = BASE / "data" / "processed" / "cluster_assignments_2022.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_om() -> pd.DataFrame:
    p = BASE / "data" / "processed" / "om_equity.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def cluster_labels() -> dict:
    p = BASE / "outputs" / "cluster_labels.json"
    if p.exists():
        raw = json.loads(p.read_text())["clusters"]
        return {int(k): v["label"] for k, v in raw.items()}
    return {}


panel = load_panel()
p4 = panel[panel["four_year"] == 1].copy()

st.title("IPEDS Outcomes Lab")
st.caption("Real federal data (NCES IPEDS), fall 2015–2023 · admissions, outcomes, "
           "peer groups and the Pell completion gap for U.S. four-year institutions.")

with st.sidebar:
    st.header("Filters")
    year = st.select_slider("Panel year", options=sorted(p4.year.unique()),
                            value=int(p4.year.max()))
    states = st.multiselect("State", sorted(p4.state.dropna().unique()))
    controls = st.multiselect("Control", ["Public", "Private nonprofit", "Private for-profit"],
                              default=["Public", "Private nonprofit", "Private for-profit"])
    carnegie = st.multiselect("Carnegie group",
                              sorted(p4.carnegie_group.dropna().unique()))

cs = p4[(p4.year == year) & (p4.control.isin(controls))]
if states:
    cs = cs[cs.state.isin(states)]
if carnegie:
    cs = cs[cs.carnegie_group.isin(carnegie)]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore", "Peer groups", "Equity: the Pell gap"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Institutions", f"{cs.unitid.nunique():,}")
    c2.metric("Median graduation rate (150%)", f"{cs.grad_rate_150.median():.1f}%")
    c3.metric("Median retention", f"{cs.retention_ft.median():.0f}%")
    c4.metric("Median Pell share (UG)", f"{cs.pell_share_ug.median():.0f}%")
    trend = (p4[p4.control.isin(controls)]
             .groupby("year")["grad_rate_150"].median().reset_index())
    fig = px.line(trend, x="year", y="grad_rate_150", markers=True,
                  title="Median graduation rate by panel year",
                  labels={"grad_rate_150": "Median grad rate (%)", "year": "Fall year"},
                  color_discrete_sequence=[NAVY])
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        cs[["institution_name", "city", "state", "control", "total_enrollment",
            "admission_rate", "retention_ft", "grad_rate_150", "pell_share_ug"]]
        .sort_values("grad_rate_150", ascending=False),
        use_container_width=True, hide_index=True)

with tab2:
    plot_df = cs.dropna(subset=["admission_rate", "grad_rate_150", "total_enrollment"])
    fig = px.scatter(
        plot_df, x="admission_rate", y="grad_rate_150", size="total_enrollment",
        color="control", hover_name="institution_name",
        color_discrete_map={"Public": NAVY, "Private nonprofit": TEAL,
                            "Private for-profit": GOLD},
        labels={"admission_rate": "Admission rate (%)",
                "grad_rate_150": "Graduation rate, 150% (%)"},
        title=f"Selectivity vs completion — {year}")
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    clusters = load_clusters()
    labels = cluster_labels()
    if clusters.empty:
        st.info("Run src/04_clustering.py to build peer groups.")
    else:
        clusters["peer_group"] = clusters["cluster"].map(labels).fillna(
            clusters["cluster"].astype(str))
        prof = (clusters.groupby("peer_group")
                .agg(institutions=("unitid", "count"),
                     median_enrollment=("total_enrollment", "median"),
                     median_admission=("admission_rate", "median"),
                     median_grad_rate=("grad_rate_150", "median"),
                     median_pell=("pell_share_ug", "median"),
                     median_spend_per_fte=("instruction_per_fte", "median"))
                .round(1).reset_index())
        st.dataframe(prof, use_container_width=True, hide_index=True)
        fig = px.scatter(
            clusters, x="admission_rate", y="grad_rate_150", color="peer_group",
            size="total_enrollment", hover_name="institution_name",
            labels={"admission_rate": "Admission rate (%)",
                    "grad_rate_150": "Graduation rate, 150% (%)"},
            title="Peer groups in selectivity–completion space (2022)")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    om = load_om()
    if om.empty:
        st.info("Run src/06_equity.py to build the Outcome Measures equity table.")
    else:
        valid = om.dropna(subset=["omachrt_pell", "omachrt_nonpell"])
        nat = (valid.groupby("survey_year")
               .apply(lambda g: pd.Series({
                   "Pell": (g.omawdn8_pell.sum() / g.omachrt_pell.sum() * 100),
                   "Non-Pell": (g.omawdn8_nonpell.sum() / g.omachrt_nonpell.sum() * 100)}),
                      include_groups=False).reset_index())
        long = nat.melt("survey_year", var_name="group", value_name="award_rate_8")
        fig = px.line(long, x="survey_year", y="award_rate_8", color="group", markers=True,
                      color_discrete_map={"Pell": GOLD, "Non-Pell": NAVY},
                      labels={"survey_year": "OM survey year (cohort entered 8 yrs earlier)",
                              "award_rate_8": "Earned an award within 8 years (%)"},
                      title="National Pell completion gap (full-time, first-time cohorts)")
        fig.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        name = st.selectbox("Institution", sorted(om.institution_name.dropna().unique()))
        sub = om[om.institution_name == name].sort_values("survey_year")
        st.dataframe(sub[["survey_year", "cohort_entry_year", "award_rate_8_pell",
                          "award_rate_8_nonpell", "gap_8_pp"]],
                     use_container_width=True, hide_index=True)

st.divider()
name = st.selectbox("Institution lookup — outcomes over time",
                    sorted(p4.institution_name.dropna().unique()))
one = p4[p4.institution_name == name].sort_values("year")
if len(one):
    fig = px.line(one, x="year", y=["grad_rate_150", "retention_ft"], markers=True,
                  labels={"value": "%", "year": "Fall year", "variable": "Measure"},
                  color_discrete_sequence=[NAVY, TEAL],
                  title=name)
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
st.caption("Source: NCES IPEDS via the IPEDS Data Center bulk files. "
           "See data_dictionary.md for variable definitions and known breaks.")
