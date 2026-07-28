"""
MovieIQ — Predictive Analytics on Film Success
================================================
Interactive Streamlit dashboard: EDA, statistical tests, and a
Random Forest model that predicts whether a movie will be a
financial success (revenue > budget).

Run locally:
    streamlit run MovieIQ.py
"""

import ast
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MovieIQ — Predictive Analytics on Film Success",
    page_icon="🎬",
    layout="wide",
)

MODEL_DIR = "model"
DATA_PATH = "movies_clean.csv" if os.path.exists("movies_clean.csv") else "movies.csv"


# ---------------------------------------------------------------------------
# DATA LOADING & CLEANING (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    # Drop rows with zero/invalid budget or revenue — these can't be used
    # to judge success and likely indicate missing/unreported data.
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

    # Target variable
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    # Parse genres (TMDB-style stringified list of dicts)
    def parse_genres(g):
        try:
            parsed = ast.literal_eval(g)
            return [item["name"] for item in parsed] if parsed else ["Unknown"]
        except (ValueError, SyntaxError):
            return ["Unknown"]

    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda x: x[0])
    return df


@st.cache_resource
def load_model():
    model_path = f"{MODEL_DIR}/movieiq_rf_model.pkl"
    encoder_path = f"{MODEL_DIR}/genre_encoder.pkl"
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        clf = joblib.load(model_path)
        le = joblib.load(encoder_path)
        return clf, le
    return None, None


@st.cache_data
def load_json_report(name):
    path = f"{MODEL_DIR}/{name}"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


df = load_data()
clf, genre_encoder = load_model()
model_report = load_json_report("model_report.json")
stats_report_cached = load_json_report("stats_report.json")

ALL_GENRES = sorted({g for glist in df["genre_list"] for g in glist})

# ---------------------------------------------------------------------------
# SIDEBAR — FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("🎬 MovieIQ")
st.sidebar.caption("Predictive Analytics on Film Success")
st.sidebar.markdown("---")

st.sidebar.header("Filters")
selected_genres = st.sidebar.multiselect(
    "Genre",
    options=ALL_GENRES,
    default=ALL_GENRES,
    help="Filter movies by genre. A movie is included if it matches any selected genre.",
)
min_vote = st.sidebar.slider(
    "Minimum vote average",
    min_value=float(df["vote_average"].min()),
    max_value=float(df["vote_average"].max()),
    value=float(df["vote_average"].min()),
    step=0.1,
)

# Apply filters
mask_genre = df["genre_list"].apply(lambda glist: any(g in selected_genres for g in glist)) if selected_genres else pd.Series([True] * len(df), index=df.index)
mask_vote = df["vote_average"] >= min_vote
filtered = df[mask_genre & mask_vote].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Movies matching filters", len(filtered))
st.sidebar.caption(f"Out of {len(df)} total movies in the dataset.")

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.markdown(
    "A movie is labeled **successful** when its **revenue exceeds its budget** "
    "(`success = 1` if `revenue > budget`, else `0`). Use the sidebar to filter "
    "by genre and minimum audience rating."
)

tab_overview, tab_eda, tab_stats, tab_model, tab_predict = st.tabs(
    ["📊 Overview", "🔍 Exploratory Analysis", "🧪 Statistical Tests", "🤖 Model Performance", "🎯 Predict Success"]
)

# ---------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset at a Glance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies (filtered)", f"{len(filtered):,}")
    col2.metric("Success Rate", f"{filtered['success'].mean():.1%}" if len(filtered) else "—")
    col3.metric("Avg. Budget", f"${filtered['budget'].mean()/1e6:.1f}M" if len(filtered) else "—")
    col4.metric("Avg. Revenue", f"${filtered['revenue'].mean()/1e6:.1f}M" if len(filtered) else "—")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Success vs Failure (filtered data)**")
        counts = filtered["success"].value_counts().reindex([0, 1]).fillna(0)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Failure", "Success"], counts.values, color=["#C0392B", "#2E8B57"])
        ax.set_ylabel("Number of Movies")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(int(v)), ha="center")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.markdown("**Top Genres (filtered data)**")
        genre_counts = pd.Series([g for glist in filtered["genre_list"] for g in glist]).value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        genre_counts.plot(kind="barh", ax=ax, color="steelblue")
        ax.invert_yaxis()
        ax.set_xlabel("Number of Movies")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.markdown("**Sample of filtered movies**")
    st.dataframe(
        filtered[["title", "primary_genre", "budget", "revenue", "popularity", "runtime", "vote_average", "success"]]
        .sort_values("revenue", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# TAB 2 — EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")

    st.markdown("#### 1. Budget vs Revenue")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = filtered["success"].map({1: "#2E8B57", 0: "#C0392B"})
    ax.scatter(filtered["budget"] / 1e6, filtered["revenue"] / 1e6, c=colors, alpha=0.5, s=25)
    if len(filtered):
        max_val = max(filtered["budget"].max(), filtered["revenue"].max()) / 1e6
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Revenue = Budget")
    ax.set_xlabel("Budget ($M)")
    ax.set_ylabel("Revenue ($M)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    corr_val = filtered["budget"].corr(filtered["revenue"]) if len(filtered) > 1 else float("nan")
    st.caption(
        f"Correlation between budget and revenue in the filtered data: **{corr_val:.3f}**. "
        "A value close to 0 means bigger budgets do *not* reliably translate into bigger box-office returns here."
    )

    st.markdown("#### 2. Genre Trends")
    genre_rows = []
    for _, row in filtered.iterrows():
        for g in row["genre_list"]:
            genre_rows.append({"genre": g, "success": row["success"]})
    if genre_rows:
        genre_df = pd.DataFrame(genre_rows)
        genre_summary = genre_df.groupby("genre").agg(count=("success", "size"), success_rate=("success", "mean")).sort_values("count", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 5))
            genre_summary["count"].plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title("Movie Count by Genre")
            ax.set_ylabel("Number of Movies")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 5))
            genre_summary.sort_values("success_rate", ascending=False)["success_rate"].plot(kind="bar", ax=ax, color="seagreen")
            ax.axhline(filtered["success"].mean(), color="red", linestyle="--", label="Overall avg")
            ax.set_title("Success Rate by Genre")
            ax.set_ylabel("Success Rate")
            ax.legend()
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No data matches the current filters.")

    st.markdown("#### 3. Popularity, Runtime & Vote Average vs Success")
    if len(filtered):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
            sns.boxplot(x="success", y=col, data=filtered, ax=ax)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Failure", "Success"])
            ax.set_title(f"{col} by Success")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### 4. Correlation Heatmap")
    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    if len(filtered) > 1:
        corr = filtered[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Note: `revenue` is used only to *derive* the `success` label — it is excluded from the "
            "predictive model itself to avoid leaking the answer directly into the features."
        )

# ---------------------------------------------------------------------------
# TAB 3 — STATISTICAL TESTS
# ---------------------------------------------------------------------------
with tab_stats:
    st.subheader("Statistical Testing")
    st.markdown(
        "These tests are computed **live on the currently filtered data**, so results will update "
        "as you change the sidebar filters."
    )

    if len(filtered) > 5 and filtered["success"].nunique() == 2:
        # T-test
        success_votes = filtered[filtered["success"] == 1]["vote_average"]
        fail_votes = filtered[filtered["success"] == 0]["vote_average"]
        if len(success_votes) > 1 and len(fail_votes) > 1:
            t_stat, p_val_t = stats.ttest_ind(success_votes, fail_votes, equal_var=False)

            st.markdown("#### T-Test — vote_average by Success")
            st.markdown("**Null hypothesis (H₀):** There is no difference in mean `vote_average` between successful and unsuccessful movies.")
            c1, c2, c3 = st.columns(3)
            c1.metric("t-statistic", f"{t_stat:.3f}")
            c2.metric("p-value", f"{p_val_t:.4f}")
            c3.metric("Significant at α=0.05?", "Yes" if p_val_t < 0.05 else "No")
            st.write(
                f"Mean vote_average — Success: **{success_votes.mean():.2f}**, "
                f"Failure: **{fail_votes.mean():.2f}**"
            )
            if p_val_t < 0.05:
                st.success("We reject the null hypothesis — vote_average differs significantly between successful and unsuccessful movies.")
            else:
                st.warning("We fail to reject the null hypothesis — there is not enough evidence that vote_average differs significantly between successful and unsuccessful movies.")

        st.markdown("---")

        # Chi-square
        st.markdown("#### Chi-Square Test — Genre vs Success")
        st.markdown("**Null hypothesis (H₀):** Genre is independent of (not associated with) movie success.")
        contingency = pd.crosstab(filtered["primary_genre"], filtered["success"])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency)
            c1, c2, c3 = st.columns(3)
            c1.metric("Chi² statistic", f"{chi2:.3f}")
            c2.metric("p-value", f"{p_val_chi:.4f}")
            c3.metric("Significant at α=0.05?", "Yes" if p_val_chi < 0.05 else "No")
            if p_val_chi < 0.05:
                st.success("We reject the null hypothesis — genre is significantly associated with movie success.")
            else:
                st.warning("We fail to reject the null hypothesis — there is not enough evidence that genre is associated with movie success in this data.")
            with st.expander("View contingency table"):
                st.dataframe(contingency)
        else:
            st.info("Not enough genre variety in the current filter selection to run a chi-square test.")
    else:
        st.info("Not enough data in the current filter selection to run statistical tests (need both success and failure cases).")

    st.markdown("---")
    st.markdown(
        "**What is a p-value?** It's the probability of observing a result at least as extreme as the one "
        "found, *assuming the null hypothesis is true*. A small p-value suggests the observed pattern is "
        "unlikely to be due to chance alone. We use the common threshold of **α = 0.05**: if p < 0.05, we "
        "treat the result as statistically significant."
    )

# ---------------------------------------------------------------------------
# TAB 4 — MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Random Forest Model Performance")

    if model_report is None:
        st.error("No trained model found. Run `python train_analysis.py` first to train the model and generate reports.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{model_report['accuracy']:.1%}")
        c2.metric("Precision", f"{model_report['precision']:.1%}")
        c3.metric("Recall", f"{model_report['recall']:.1%}")

        st.caption(
            f"Trained on {model_report['train_size']} movies, tested on {model_report['test_size']} "
            f"({model_report['test_split_ratio']:.0%} held out as a test set)."
        )

        st.markdown("**Features used:** " + ", ".join(model_report["features_used"]))
        with st.expander("Why were some columns excluded?"):
            for reason in model_report["excluded_features"]:
                st.markdown(f"- {reason}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion Matrix")
            if os.path.exists("assets/05_confusion_matrix.png"):
                st.image("assets/05_confusion_matrix.png")
        with c2:
            st.markdown("#### Feature Importance")
            if os.path.exists("assets/06_feature_importance.png"):
                st.image("assets/06_feature_importance.png")

        st.caption(
            "Feature importance shows which inputs the Random Forest relied on most heavily when making "
            "its predictions — compare this against the EDA and statistical test results above."
        )

# ---------------------------------------------------------------------------
# TAB 5 — PREDICT SUCCESS
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict Whether a Movie Will Succeed")

    if clf is None or genre_encoder is None:
        st.error("No trained model found. Run `python train_analysis.py` first to train and save the model.")
    else:
        st.markdown("Enter a movie's details below to get a prediction from the trained Random Forest model.")

        c1, c2 = st.columns(2)
        with c1:
            input_budget = st.number_input("Budget ($)", min_value=1000.0, max_value=500_000_000.0, value=50_000_000.0, step=1_000_000.0, format="%.0f")
            input_popularity = st.slider("Popularity", min_value=0.0, max_value=150.0, value=50.0, step=0.5)
            input_runtime = st.slider("Runtime (minutes)", min_value=60, max_value=240, value=120, step=1)
        with c2:
            input_vote_average = st.slider("Vote Average (0–10)", min_value=0.0, max_value=10.0, value=6.5, step=0.1)
            input_genre = st.selectbox("Primary Genre", options=list(genre_encoder.classes_))

        if st.button("🎯 Predict Success", type="primary"):
            genre_encoded = genre_encoder.transform([input_genre])[0]
            X_new = pd.DataFrame(
                [[input_budget, input_popularity, input_runtime, input_vote_average, genre_encoded]],
                columns=["budget", "popularity", "runtime", "vote_average", "genre_encoded"],
            )
            pred = clf.predict(X_new)[0]
            proba = clf.predict_proba(X_new)[0]

            if pred == 1:
                st.success(f"✅ Predicted: **SUCCESS** (revenue expected to exceed budget)")
            else:
                st.error(f"❌ Predicted: **NOT SUCCESSFUL** (revenue not expected to exceed budget)")

            st.metric("Model confidence (success probability)", f"{proba[1]:.1%}")
            st.progress(float(proba[1]))

st.markdown("---")
st.caption("MovieIQ — built with Streamlit, scikit-learn, and a lot of popcorn. 🍿")
