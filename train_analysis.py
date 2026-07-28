"""
MovieIQ - Data Preparation, EDA, Statistical Testing & Model Training
=======================================================================
Run this script once to:
  1. Clean the raw movies.csv
  2. Generate all EDA charts (saved to assets/)
  3. Run statistical tests (t-test, chi-square)
  4. Train a Random Forest classifier to predict movie success
  5. Save the trained model + metadata for the Streamlit app to use

Usage:
    python train_analysis.py
"""

import ast
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

ASSETS_DIR = "assets"
MODEL_DIR = "model"


# ---------------------------------------------------------------------------
# STAGE 1 — DATA PREPARATION
# ---------------------------------------------------------------------------
def load_and_clean_data(path="movies.csv"):
    df = pd.read_csv(path)

    report = {}
    report["raw_shape"] = df.shape
    report["summary_stats"] = df[["budget", "revenue", "popularity", "runtime", "vote_average"]].describe().to_dict()

    # Missing values
    report["missing_values"] = df.isnull().sum().to_dict()

    # Zero / invalid budget or revenue -> these rows can't be used to judge
    # success (division by / comparison against zero budget is meaningless,
    # and a $0 revenue or budget usually signals missing/unreported data
    # rather than an actual $0 spend or earning).
    zero_budget = (df["budget"] <= 0).sum()
    zero_revenue = (df["revenue"] <= 0).sum()
    report["zero_budget_rows"] = int(zero_budget)
    report["zero_revenue_rows"] = int(zero_revenue)

    before = len(df)
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    report["rows_dropped_zero_budget_revenue"] = before - len(df)

    # Target variable: success = 1 when revenue > budget
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    report["success_rate"] = float(df["success"].mean())
    report["class_counts"] = df["success"].value_counts().to_dict()

    # Parse genres (stored as a stringified list of dicts, TMDB-style)
    def parse_genres(g):
        try:
            parsed = ast.literal_eval(g)
            return [item["name"] for item in parsed] if parsed else ["Unknown"]
        except (ValueError, SyntaxError):
            return ["Unknown"]

    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda x: x[0])

    report["unique_genres"] = sorted({g for glist in df["genre_list"] for g in glist})

    df.to_csv("movies_clean.csv", index=False)
    return df, report


# ---------------------------------------------------------------------------
# STAGE 2 — EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
def run_eda(df):
    eda_report = {}

    # 1. Budget vs Revenue scatter
    fig, ax = plt.subplots()
    colors = df["success"].map({1: "#2E8B57", 0: "#C0392B"})
    ax.scatter(df["budget"] / 1e6, df["revenue"] / 1e6, c=colors, alpha=0.5, s=25)
    max_val = max(df["budget"].max(), df["revenue"].max()) / 1e6
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="Revenue = Budget")
    ax.set_xlabel("Budget ($M)")
    ax.set_ylabel("Revenue ($M)")
    ax.set_title("Budget vs Revenue (green = success, red = failure)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/01_budget_vs_revenue.png", dpi=120)
    plt.close()

    corr_budget_revenue = df["budget"].corr(df["revenue"])
    eda_report["budget_revenue_correlation"] = float(corr_budget_revenue)

    # 2. Genre trends — frequency and success rate
    genre_rows = []
    for _, row in df.iterrows():
        for g in row["genre_list"]:
            genre_rows.append({"genre": g, "success": row["success"]})
    genre_df = pd.DataFrame(genre_rows)
    genre_summary = genre_df.groupby("genre").agg(count=("success", "size"), success_rate=("success", "mean")).sort_values("count", ascending=False)
    eda_report["genre_summary"] = genre_summary.to_dict(orient="index")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    genre_summary["count"].plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Movie Count by Genre")
    axes[0].set_ylabel("Number of Movies")
    axes[0].set_xlabel("Genre")

    genre_summary.sort_values("success_rate", ascending=False)["success_rate"].plot(kind="bar", ax=axes[1], color="seagreen")
    axes[1].set_title("Success Rate by Genre")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_xlabel("Genre")
    axes[1].axhline(df["success"].mean(), color="red", linestyle="--", label="Overall avg")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/02_genre_trends.png", dpi=120)
    plt.close()

    # 3. Popularity / runtime / vote_average vs success
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["popularity", "runtime", "vote_average"]):
        sns.boxplot(x="success", y=col, data=df, ax=ax)
        ax.set_xticklabels(["Failure", "Success"])
        ax.set_title(f"{col} by Success")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/03_features_vs_success.png", dpi=120)
    plt.close()

    means = df.groupby("success")[["popularity", "runtime", "vote_average"]].mean()
    eda_report["feature_means_by_success"] = means.to_dict()

    # 4. Correlation heatmap
    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/04_correlation_heatmap.png", dpi=120)
    plt.close()

    eda_report["correlation_matrix"] = corr.to_dict()

    return eda_report


# ---------------------------------------------------------------------------
# STAGE 3 — STATISTICAL TESTING
# ---------------------------------------------------------------------------
def run_statistical_tests(df):
    stats_report = {}

    # T-test: vote_average between successful vs unsuccessful movies
    success_votes = df[df["success"] == 1]["vote_average"]
    fail_votes = df[df["success"] == 0]["vote_average"]
    t_stat, p_val_t = stats.ttest_ind(success_votes, fail_votes, equal_var=False)
    stats_report["ttest"] = {
        "feature": "vote_average",
        "null_hypothesis": "There is no difference in mean vote_average between successful and unsuccessful movies.",
        "t_statistic": float(t_stat),
        "p_value": float(p_val_t),
        "mean_success": float(success_votes.mean()),
        "mean_failure": float(fail_votes.mean()),
        "significant_at_0.05": bool(p_val_t < 0.05),
    }

    # Chi-square: primary_genre vs success
    contingency = pd.crosstab(df["primary_genre"], df["success"])
    chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency)
    stats_report["chi_square"] = {
        "feature": "primary_genre",
        "null_hypothesis": "Genre is independent of (not associated with) movie success.",
        "chi2_statistic": float(chi2),
        "p_value": float(p_val_chi),
        "degrees_of_freedom": int(dof),
        "significant_at_0.05": bool(p_val_chi < 0.05),
    }

    with open(f"{MODEL_DIR}/stats_report.json", "w") as f:
        json.dump(stats_report, f, indent=2)

    return stats_report


# ---------------------------------------------------------------------------
# STAGE 4 — PREDICTIVE MODELING (RANDOM FOREST)
# ---------------------------------------------------------------------------
def train_model(df):
    feature_cols = ["budget", "popularity", "runtime", "vote_average", "primary_genre"]
    target_col = "success"

    model_df = df[feature_cols + [target_col]].copy()

    le = LabelEncoder()
    model_df["genre_encoded"] = le.fit_transform(model_df["primary_genre"])

    X = model_df[["budget", "popularity", "runtime", "vote_average", "genre_encoded"]]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    model_report = {
        "features_used": list(X.columns),
        "excluded_features": ["title (identifier, no predictive value)", "revenue (used to derive the target itself — including it would leak the answer)", "genres (raw string; replaced by encoded primary_genre)"],
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_split_ratio": 0.2,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    cm = confusion_matrix(y_test, y_pred)
    model_report["confusion_matrix"] = cm.tolist()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Failure", "Success"], yticklabels=["Failure", "Success"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/05_confusion_matrix.png", dpi=120)
    plt.close()

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    model_report["feature_importance"] = importances.to_dict()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="darkorange")
    ax.invert_yaxis()
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{ASSETS_DIR}/06_feature_importance.png", dpi=120)
    plt.close()

    # Save model + encoder + genre list for the Streamlit app
    joblib.dump(clf, f"{MODEL_DIR}/movieiq_rf_model.pkl")
    joblib.dump(le, f"{MODEL_DIR}/genre_encoder.pkl")

    with open(f"{MODEL_DIR}/model_report.json", "w") as f:
        json.dump(model_report, f, indent=2)

    return model_report


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    os.makedirs(ASSETS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading and cleaning data...")
    df, data_report = load_and_clean_data("movies.csv")
    print(f"  Shape after cleaning: {df.shape}")
    print(f"  Success rate: {data_report['success_rate']:.1%}")

    with open(f"{MODEL_DIR}/data_report.json", "w") as f:
        json.dump(data_report, f, indent=2, default=str)

    print("Running EDA...")
    eda_report = run_eda(df)

    with open(f"{MODEL_DIR}/eda_report.json", "w") as f:
        json.dump(eda_report, f, indent=2, default=str)

    print("Running statistical tests...")
    stats_report = run_statistical_tests(df)
    print(f"  T-test p-value: {stats_report['ttest']['p_value']:.6f}")
    print(f"  Chi-square p-value: {stats_report['chi_square']['p_value']:.6f}")

    print("Training Random Forest model...")
    model_report = train_model(df)
    print(f"  Accuracy: {model_report['accuracy']:.3f}")
    print(f"  Precision: {model_report['precision']:.3f}")
    print(f"  Recall: {model_report['recall']:.3f}")

    print("\nAll done. Charts saved to assets/, model + reports saved to model/.")
