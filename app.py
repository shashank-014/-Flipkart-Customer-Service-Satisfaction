from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Customer_support_data.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "flipkart_customer_satisfaction_model.joblib"


st.set_page_config(
    page_title="Flipkart Customer Satisfaction Prediction",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(6, 102, 129, 0.08), transparent 24%),
            radial-gradient(circle at bottom left, rgba(237, 125, 49, 0.08), transparent 20%),
            linear-gradient(180deg, #f9fbfc 0%, #eef3f6 100%);
        color: #1f2d33;
    }
    .hero {
        padding: 1.25rem 1.35rem;
        border: 1px solid #d1dde4;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(244,248,250,0.95));
        box-shadow: 0 16px 28px rgba(31, 45, 51, 0.08);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        line-height: 1.1;
        color: #12313d;
    }
    .hero p {
        margin: 0.35rem 0 0;
        color: #536973;
        font-size: 1rem;
    }
    .chip_row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }
    .chip {
        display: inline-block;
        padding: 0.35rem 0.68rem;
        border-radius: 999px;
        background: rgba(6, 102, 129, 0.10);
        color: #066681;
        font-size: 0.84rem;
        border: 1px solid rgba(6, 102, 129, 0.16);
    }
    .card {
        padding: 1rem 1rem 0.9rem;
        border-radius: 18px;
        border: 1px solid #d1dde4;
        background: rgba(255,255,255,0.80);
        box-shadow: 0 10px 20px rgba(31, 45, 51, 0.05);
        margin-bottom: 1rem;
    }
    .card h3 {
        margin: 0 0 0.45rem;
        color: #066681;
    }
    .small_note {
        color: #536973;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .metric_box {
        padding: 0.85rem 1rem;
        border-radius: 16px;
        border: 1px solid #d1dde4;
        background: linear-gradient(180deg, #ffffff 0%, #f4f8fa 100%);
        box-shadow: 0 10px 20px rgba(31, 45, 51, 0.05);
    }
    .metric_label {
        color: #536973;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric_value {
        color: #12313d;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.12rem;
    }
    .metric_sub {
        color: #536973;
        font-size: 0.84rem;
        margin-top: 0.15rem;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@dataclass(frozen=True)
class ModelResult:
    name: str
    model: object
    cv_roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    train_roc_auc: float


def fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def fmt_pct(value: float) -> str:
    return f"{value:.3f}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Unique id": "unique_id",
        "channel_name": "channel_name",
        "category": "category",
        "Sub-category": "sub_category",
        "Customer Remarks": "customer_remarks",
        "Order_id": "order_id",
        "order_date_time": "order_date_time",
        "Issue_reported at": "issue_reported_at",
        "issue_responded": "issue_responded",
        "Survey_response_Date": "survey_response_date",
        "Customer_City": "customer_city",
        "Product_category": "product_category",
        "Item_price": "item_price",
        "connected_handling_time": "connected_handling_time",
        "Agent_name": "agent_name",
        "Supervisor": "supervisor",
        "Manager": "manager",
        "Tenure Bucket": "tenure_bucket",
        "Agent Shift": "agent_shift",
        "CSAT Score": "csat_score",
    }
    out = df.rename(columns=rename_map).copy()

    for col in ["order_date_time", "issue_reported_at", "issue_responded", "survey_response_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)

    for col in ["item_price", "connected_handling_time", "csat_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def group_top_values(series: pd.Series, top_n: int = 20, other_label: str = "Other") -> pd.Series:
    filled = series.fillna("Unknown").astype(str)
    top_values = filled.value_counts().head(top_n).index
    return filled.where(filled.isin(top_values), other_label)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return normalize_columns(pd.read_csv(DATA_PATH))


@st.cache_data(show_spinner=False)
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["satisfaction_label"] = (data["csat_score"] >= 4).astype(int)
    data["response_delay_mins"] = (
        data["issue_responded"] - data["issue_reported_at"]
    ).dt.total_seconds() / 60
    data["time_to_survey_mins"] = (
        data["survey_response_date"] - data["issue_reported_at"]
    ).dt.total_seconds() / 60
    data["reported_hour"] = data["issue_reported_at"].dt.hour
    data["reported_dayofweek"] = data["issue_reported_at"].dt.dayofweek
    data["response_hour"] = data["issue_responded"].dt.hour
    data["response_dayofweek"] = data["issue_responded"].dt.dayofweek
    data["survey_month"] = data["survey_response_date"].dt.month
    data["survey_year"] = data["survey_response_date"].dt.year
    data["order_hour"] = data["order_date_time"].dt.hour
    data["order_month"] = data["order_date_time"].dt.month
    data["remarks_len"] = data["customer_remarks"].fillna("").astype(str).str.len()
    data["remarks_words"] = data["customer_remarks"].fillna("").astype(str).str.split().str.len()
    data["has_remarks"] = data["customer_remarks"].notna().astype(int)
    data["item_price_filled"] = data["item_price"].fillna(data["item_price"].median())
    data["handling_time_filled"] = data["connected_handling_time"].fillna(
        data["connected_handling_time"].median()
    )
    data["channel_bucket"] = data["channel_name"].fillna("Unknown").astype(str)
    data["category_bucket"] = data["category"].fillna("Unknown").astype(str)
    data["sub_category_bucket"] = group_top_values(data["sub_category"], top_n=25)
    data["city_bucket"] = group_top_values(data["customer_city"], top_n=25)
    data["product_bucket"] = group_top_values(data["product_category"], top_n=20)
    data["tenure_bucket_clean"] = data["tenure_bucket"].fillna("Unknown").astype(str)
    data["shift_bucket"] = data["agent_shift"].fillna("Unknown").astype(str)
    data["response_bucket"] = pd.cut(
        data["response_delay_mins"],
        bins=[-np.inf, 30, 60, 180, np.inf],
        labels=["0-30 min", "30-60 min", "1-3 hr", "3+ hr"],
    ).astype(str)
    data["price_bucket"] = pd.cut(
        data["item_price_filled"],
        bins=[-np.inf, 500, 1500, 3000, np.inf],
        labels=["Low", "Mid", "High", "Premium"],
    ).astype(str)
    data["handling_bucket"] = pd.cut(
        data["handling_time_filled"],
        bins=[-np.inf, 10, 20, 40, np.inf],
        labels=["Very short", "Short", "Medium", "Long"],
    ).astype(str)
    data["hour_bucket"] = pd.cut(
        data["reported_hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    ).astype(str)
    data["delay_flag"] = (data["response_delay_mins"] > 60).astype(int)
    data["satisfaction_text"] = np.where(data["satisfaction_label"] == 1, "Satisfied", "Not Satisfied")
    return data


def dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "start_date": df["issue_reported_at"].min(),
        "end_date": df["issue_reported_at"].max(),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "satisfied": int(df["satisfaction_label"].sum()),
        "not_satisfied": int((1 - df["satisfaction_label"]).sum()),
        "sat_rate": float(df["satisfaction_label"].mean()),
        "csat_min": int(df["csat_score"].min()),
        "csat_max": int(df["csat_score"].max()),
    }


def make_metric_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        f"""
        <div class="metric_box">
            <div class="metric_label">{label}</div>
            <div class="metric_value">{value}</div>
            <div class="metric_sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)


def show_chart(title: str, why: str, insight: str, impact: str, fig) -> None:
    with st.expander(title, expanded=False):
        st.pyplot(fig, use_container_width=True)
        st.markdown(f"**Why this chart:** {why}")
        st.markdown(f"**What it shows:** {insight}")
        st.markdown(f"**Business read:** {impact}")


def plot_class_balance(df: pd.DataFrame):
    counts = df["satisfaction_text"].value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=["#066681", "#ed7d31"])
    ax.set_title("Customer Satisfaction Split", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_csat_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.countplot(data=df, x="csat_score", palette="viridis", ax=ax)
    style_axes(ax, "CSAT Score Distribution", "CSAT score", "Count")
    fig.tight_layout()
    return fig


def plot_channel_satisfaction(df: pd.DataFrame):
    cross = df.groupby(["channel_bucket", "satisfaction_text"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(data=cross, x="channel_bucket", y="count", hue="satisfaction_text", ax=ax)
    style_axes(ax, "Channel vs Satisfaction", "Channel", "Count")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_category_satisfaction(df: pd.DataFrame):
    cross = df.groupby(["category_bucket", "satisfaction_text"]).size().reset_index(name="count")
    top_categories = df["category_bucket"].value_counts().head(8).index.tolist()
    cross = cross[cross["category_bucket"].isin(top_categories)]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.barplot(data=cross, x="category_bucket", y="count", hue="satisfaction_text", ax=ax)
    style_axes(ax, "Top Issue Categories vs Satisfaction", "Category", "Count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_response_delay_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.boxplot(data=df, x="satisfaction_text", y="response_delay_mins", palette="Set2", ax=ax)
    style_axes(ax, "Response Delay by Satisfaction", "Satisfaction", "Response delay (mins)")
    fig.tight_layout()
    return fig


def plot_handling_time_box(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.boxplot(data=df, x="satisfaction_text", y="handling_time_filled", palette="Set3", ax=ax)
    style_axes(ax, "Handling Time by Satisfaction", "Satisfaction", "Connected handling time")
    fig.tight_layout()
    return fig


def plot_tenure_satisfaction(df: pd.DataFrame):
    cross = df.groupby(["tenure_bucket_clean", "satisfaction_text"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=cross, x="tenure_bucket_clean", y="count", hue="satisfaction_text", ax=ax)
    style_axes(ax, "Tenure Bucket vs Satisfaction", "Tenure bucket", "Count")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_shift_satisfaction(df: pd.DataFrame):
    cross = df.groupby(["shift_bucket", "satisfaction_text"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(data=cross, x="shift_bucket", y="count", hue="satisfaction_text", ax=ax)
    style_axes(ax, "Agent Shift vs Satisfaction", "Shift", "Count")
    ax.tick_params(axis="x", rotation=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_city_top(df: pd.DataFrame):
    top = df["city_bucket"].value_counts().head(12).reset_index()
    top.columns = ["city_bucket", "count"]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.barplot(data=top, y="city_bucket", x="count", color="#ed7d31", ax=ax)
    style_axes(ax, "Top Customer Cities by Volume", "Count", "City")
    fig.tight_layout()
    return fig


def plot_price_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.histplot(df["item_price_filled"], bins=35, kde=True, color="#066681", ax=ax)
    style_axes(ax, "Item Price Distribution", "Item price", "Count")
    fig.tight_layout()
    return fig


def plot_hour_satisfaction(df: pd.DataFrame):
    cross = df.groupby(["hour_bucket", "satisfaction_text"]).size().reset_index(name="count")
    order = ["Night", "Morning", "Afternoon", "Evening"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(data=cross, x="hour_bucket", y="count", hue="satisfaction_text", order=order, ax=ax)
    style_axes(ax, "Issue Hour Bucket vs Satisfaction", "Hour bucket", "Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False).head(15).reset_index()
    miss.columns = ["column", "missing_rate"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=miss, y="column", x="missing_rate", color="#a7c7e7", ax=ax)
    style_axes(ax, "Top Missing Value Rates", "Missing rate", "Column")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
    fig.tight_layout()
    return fig


def plot_response_delay_scatter(df: pd.DataFrame):
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.scatterplot(
        data=sample,
        x="response_delay_mins",
        y="item_price_filled",
        hue="satisfaction_text",
        alpha=0.45,
        ax=ax,
    )
    style_axes(ax, "Response Delay vs Item Price", "Response delay (mins)", "Item price")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_corr_heatmap(df: pd.DataFrame):
    numeric_cols = [
        "item_price_filled",
        "handling_time_filled",
        "response_delay_mins",
        "time_to_survey_mins",
        "remarks_len",
        "remarks_words",
        "reported_hour",
        "reported_dayofweek",
        "response_hour",
        "response_dayofweek",
        "survey_month",
        "survey_year",
        "order_hour",
        "order_month",
        "satisfaction_label",
    ]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10.5, 7))
    sns.heatmap(corr, cmap="YlGnBu", vmin=-1, vmax=1, ax=ax)
    style_axes(ax, "Correlation Heatmap", "", "")
    fig.tight_layout()
    return fig


def plot_top_subcat_dissatisfaction(df: pd.DataFrame):
    summary = (
        df.groupby("sub_category_bucket")["satisfaction_label"]
        .agg(["mean", "count"])
        .sort_values("count", ascending=False)
        .head(12)
        .reset_index()
    )
    summary["dissatisfaction_rate"] = 1 - summary["mean"]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.barplot(data=summary, y="sub_category_bucket", x="dissatisfaction_rate", color="#ed7d31", ax=ax)
    style_axes(ax, "Dissatisfaction Rate by Sub-category", "Dissatisfaction rate", "Sub-category")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
    fig.tight_layout()
    return fig


def plot_model_metrics(metrics_df: pd.DataFrame):
    melted = metrics_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    fig, ax = plt.subplots(figsize=(10.5, 5))
    sns.barplot(data=melted, x="model", y="value", hue="variable", ax=ax)
    style_axes(ax, "Model Comparison", "Model", "Score")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def build_preprocessors():
    numeric_features = [
        "item_price_filled",
        "handling_time_filled",
        "response_delay_mins",
        "time_to_survey_mins",
        "reported_hour",
        "reported_dayofweek",
        "response_hour",
        "response_dayofweek",
        "survey_month",
        "survey_year",
        "order_hour",
        "order_month",
        "remarks_len",
        "remarks_words",
        "has_remarks",
        "delay_flag",
    ]
    categorical_features = [
        "channel_bucket",
        "category_bucket",
        "sub_category_bucket",
        "city_bucket",
        "product_bucket",
        "tenure_bucket_clean",
        "shift_bucket",
        "response_bucket",
        "price_bucket",
        "handling_bucket",
        "hour_bucket",
    ]

    logistic = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    tree = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return logistic, tree


def build_models():
    logistic_preprocessor, tree_preprocessor = build_preprocessors()
    models: dict[str, object] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", logistic_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="saga",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=140,
                        max_depth=15,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("prep", tree_preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.08,
                        max_iter=120,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("prep", tree_preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def fit_xgb_scale_weight(y_train: pd.Series) -> float:
    positive = max(int((y_train == 1).sum()), 1)
    negative = int((y_train == 0).sum())
    return negative / positive


@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, object, pd.DataFrame, list[str]]:
    feature_cols = [
        "item_price_filled",
        "handling_time_filled",
        "response_delay_mins",
        "time_to_survey_mins",
        "reported_hour",
        "reported_dayofweek",
        "response_hour",
        "response_dayofweek",
        "survey_month",
        "survey_year",
        "order_hour",
        "order_month",
        "remarks_len",
        "remarks_words",
        "has_remarks",
        "delay_flag",
        "channel_bucket",
        "category_bucket",
        "sub_category_bucket",
        "city_bucket",
        "product_bucket",
        "tenure_bucket_clean",
        "shift_bucket",
        "response_bucket",
        "price_bucket",
        "handling_bucket",
        "hour_bucket",
    ]

    working = df.dropna(subset=["csat_score"]).copy()
    x = working[feature_cols]
    y = working["satisfaction_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    if "XGBoost" in models:
        models["XGBoost"].named_steps["model"].set_params(
            scale_pos_weight=fit_xgb_scale_weight(y_train)
        )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    best_model = None
    best_name = None
    best_score = -1

    for name, model in models.items():
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(x_train, y_train)
        train_proba = model.predict_proba(x_train)[:, 1]
        test_pred = model.predict(x_test)
        test_proba = model.predict_proba(x_test)[:, 1]

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, zero_division=0),
            "recall": recall_score(y_test, test_pred, zero_division=0),
            "f1": f1_score(y_test, test_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_proba),
            "cv_roc_auc": float(cv_scores.mean()),
            "model_obj": model,
        }
        rows.append(row)

        if row["cv_roc_auc"] > best_score:
            best_score = row["cv_roc_auc"]
            best_model = model
            best_name = name

    metrics_df = (
        pd.DataFrame(rows)
        .drop(columns=["model_obj"])
        .sort_values(["cv_roc_auc", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    selected = metrics_df.iloc[0]["model"]
    if best_name != selected:
        best_model = next(row["model_obj"] for row in rows if row["model"] == selected)

    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": best_model, "feature_cols": feature_cols}, MODEL_PATH)
    except Exception:
        pass

    return metrics_df, best_model, x_test.assign(_target=y_test.values), feature_cols


def plot_confusion(best_model, x_test: pd.DataFrame, y_test: pd.Series):
    y_pred = best_model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix - Best Model")
    fig.tight_layout()
    return fig


def plot_roc(best_model, x_test: pd.DataFrame, y_test: pd.Series):
    y_proba = best_model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.plot(fpr, tpr, color="#066681", linewidth=2.2, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    style_axes(ax, "ROC Curve - Best Model", "False positive rate", "True positive rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_feature_importance(best_model, feature_cols):
    core = best_model.named_steps["model"] if hasattr(best_model, "named_steps") else best_model
    prep = best_model.named_steps["prep"] if hasattr(best_model, "named_steps") else None
    fig, ax = plt.subplots(figsize=(10, 5))

    if hasattr(core, "feature_importances_") and prep is not None:
        try:
            encoded_names = prep.get_feature_names_out()
        except Exception:
            encoded_names = np.array(feature_cols)
        importances = pd.Series(core.feature_importances_, index=encoded_names).sort_values().tail(15)
        importances.plot(kind="barh", color="#066681", ax=ax)
        style_axes(ax, "Top Feature Importances", "Importance", "Feature")
    elif hasattr(core, "coef_") and prep is not None:
        try:
            encoded_names = prep.get_feature_names_out()
        except Exception:
            encoded_names = np.array(feature_cols)
        coefs = pd.Series(np.abs(core.coef_[0]), index=encoded_names).sort_values().tail(15)
        coefs.plot(kind="barh", color="#ed7d31", ax=ax)
        style_axes(ax, "Top Logistic Coefficients", "Absolute coefficient", "Feature")
    else:
        ax.text(0.5, 0.5, "Feature importance not available for this model.", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    return fig


def render_sidebar(summary: dict[str, object]) -> str:
    st.sidebar.markdown("## Flipkart Project")
    st.sidebar.markdown(
        """
        Support data, customer experience, and a clear model story.

        The layout follows the same style as the notebook and presentation notes.
        """
    )
    st.sidebar.markdown("### Quick facts")
    st.sidebar.write(f"Rows: {summary['rows']}")
    st.sidebar.write(f"Columns: {summary['cols']}")
    st.sidebar.write(f"Class split: {summary['satisfied']} satisfied / {summary['not_satisfied']} not satisfied")
    st.sidebar.write(f"Date range: {summary['start_date']} to {summary['end_date']}")
    st.sidebar.write(f"Missing cells: {summary['missing_cells']}")

    section = st.sidebar.radio(
        "Jump to section",
        [
            "Overview",
            "Data Understanding",
            "Data Wrangling",
            "EDA",
            "Hypothesis Testing",
            "Feature Engineering",
            "Modeling",
            "Conclusion",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Build notes")
    st.sidebar.markdown(
        """
        - Human tone in the explanations
        - Short, functional comments in code
        - Folder-specific files stay inside `Flipkart/current`
        - Old docs are archived inside `DELETED_FILES`
        """
    )
    return section


def overview_section(df: pd.DataFrame, summary: dict[str, object]) -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Flipkart Customer Satisfaction Prediction</h1>
            <p>Support interaction analysis, classification modeling, and a clean presentation style that stays close to the notebook tone.</p>
            <div class="chip_row">
                <span class="chip">EDA and classification</span>
                <span class="chip">Imbalanced target</span>
                <span class="chip">Response-time focus</span>
                <span class="chip">Deployment-ready pipeline</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        make_metric_card("Rows", str(summary["rows"]), "Customer support records")
    with c2:
        make_metric_card("Satisfied", f"{summary['satisfied']:,}", "CSAT >= 4")
    with c3:
        make_metric_card("Not satisfied", f"{summary['not_satisfied']:,}", "CSAT < 4")
    with c4:
        make_metric_card("Satisfaction rate", f"{summary['sat_rate']:.1%}", "Majority class share")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(
            """
            <div class="card">
                <h3>Project summary</h3>
                <div class="small_note">
                    This dataset has 85,907 support interactions and 20 columns.
                    The target is the CSAT score, which I convert into a binary satisfaction label for classification.
                    The core story is simple: which support interactions end well, which ones do not, and what patterns
                    separate the two groups.
                    <br><br>
                    The model is not here to replace the support team. It is here to spot risk earlier, so the team can
                    prioritize response timing, issue types, and high-risk support situations before the customer gets
                    frustrated.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
                <h3>Problem statement</h3>
                <div class="small_note">
                    Predict whether a Flipkart support interaction will end in satisfaction or dissatisfaction using
                    channel, issue type, timing, product, agent, and response features.
                    <br><br>
                    The business goal is not just accuracy. It is early detection of unhappy customers and the ability
                    to act before the issue turns into repeat contact, poor sentiment, or churn risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        df.head(8)[
            [
                "channel_name",
                "category",
                "sub_category",
                "issue_reported_at",
                "issue_responded",
                "customer_city",
                "product_category",
                "csat_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def data_understanding_section(df: pd.DataFrame, summary: dict[str, object]) -> None:
    st.subheader("Data Understanding")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            """
            <div class="card">
                <h3>What is in the file</h3>
                <div class="small_note">
                    A real-world support log: channel, issue category, sub-category, timestamps, agent hierarchy,
                    item value, handling time, and CSAT score.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="card">
                <h3>What stands out</h3>
                <div class="small_note">
                    The target is imbalanced.
                    Most interactions are satisfied, but the dissatisfied group is still large enough to matter.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            """
            <div class="card">
                <h3>What needs care</h3>
                <div class="small_note">
                    Several fields are missing in a column-specific way, which is normal in operational data.
                    We need to handle that carefully instead of assuming the file is perfectly complete.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    facts = pd.DataFrame(
        {
            "item": [
                "Rows",
                "Columns",
                "Missing cells",
                "Duplicate rows",
                "Satisfied class",
                "Not satisfied class",
                "CSAT range",
            ],
            "value": [
                f"{summary['rows']:,}",
                f"{summary['cols']}",
                f"{summary['missing_cells']:,}",
                f"{summary['duplicates']:,}",
                f"{summary['satisfied']:,}",
                f"{summary['not_satisfied']:,}",
                f"{summary['csat_min']} to {summary['csat_max']}",
            ],
        }
    )
    st.dataframe(facts, use_container_width=True, hide_index=True)


def data_wrangling_section(df: pd.DataFrame) -> None:
    st.subheader("Data Wrangling")
    st.markdown(
        """
        <div class="card">
            <h3>What I changed</h3>
            <div class="small_note">
                I created a working copy, converted CSAT into a binary satisfaction label, parsed the date columns,
                filled missing values in a controlled way, and grouped high-cardinality labels like city and
                sub-category into a clean top-values-plus-Other format.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    wrangle_notes = pd.DataFrame(
        {
            "step": [
                "Target creation",
                "Date parsing",
                "Missing values",
                "High-cardinality labels",
                "Time features",
                "Text features",
            ],
            "what it does": [
                "CSAT >= 4 becomes satisfied, else not satisfied",
                "Turns raw strings into usable timestamps",
                "Keeps the file usable without forcing bad assumptions",
                "Compresses city and sub-category into manageable buckets",
                "Adds response delay and hour/day fields",
                "Uses remark length and word count as light text signals",
            ],
        }
    )
    st.dataframe(wrangle_notes, use_container_width=True, hide_index=True)


def hypothesis_section(df: pd.DataFrame) -> None:
    st.subheader("Hypothesis Testing")
    sat = df[df["satisfaction_label"] == 1]
    unsat = df[df["satisfaction_label"] == 0]

    tests = []
    for label, col in [
        ("Response delay", "response_delay_mins"),
        ("Handling time", "handling_time_filled"),
        ("Item price", "item_price_filled"),
    ]:
        a = sat[col].dropna()
        b = unsat[col].dropna()
        _, p_value = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        tests.append(
            {
                "feature": label,
                "p_value": p_value,
                "mean_satisfied": a.mean(),
                "mean_not_satisfied": b.mean(),
            }
        )

    tests_df = pd.DataFrame(tests)
    st.dataframe(
        tests_df.assign(
            p_value=lambda d: d["p_value"].map(lambda x: f"{x:.4g}"),
            mean_satisfied=lambda d: d["mean_satisfied"].map(fmt_num),
            mean_not_satisfied=lambda d: d["mean_not_satisfied"].map(fmt_num),
        ),
        use_container_width=True,
        hide_index=True,
    )


def exploratory_section(df: pd.DataFrame) -> None:
    st.subheader("Exploratory Analysis")
    st.caption("I kept the charts focused on the parts of the support flow that matter most for satisfaction.")

    chart_specs = [
        ("Chart 1 - Satisfaction split", plot_class_balance(df),
         "A pie chart is the fastest way to show the class balance.",
         "Most interactions are satisfied, but the dissatisfied group is still big enough to matter.",
         "The business should not ignore the smaller class just because it is not the majority."),
        ("Chart 2 - CSAT distribution", plot_csat_distribution(df),
         "A bar chart shows the actual score spread behind the binary target.",
         "Ratings are concentrated near the top, which matches the class imbalance.",
         "This confirms why accuracy alone would be misleading."),
        ("Chart 3 - Channel vs satisfaction", plot_channel_satisfaction(df),
         "Channel volume matters because service mode can shape the experience.",
         "Inbound carries the largest load and the biggest share of dissatisfaction.",
         "That means channel operations are a high-return place to improve service quality."),
        ("Chart 4 - Category vs satisfaction", plot_category_satisfaction(df),
         "Issue category helps reveal which problem families create more friction.",
         "Returns, order-related issues, and refunds appear frequently in the trouble zone.",
         "Fixing a few high-volume categories can lift satisfaction faster than changing everything."),
        ("Chart 5 - Response delay boxplot", plot_response_delay_box(df),
         "Response time is one of the strongest operational signals in support data.",
         "Dissatisfied customers usually wait longer before the issue gets a response.",
         "Reducing delay should be treated as a direct customer satisfaction lever."),
        ("Chart 6 - Handling time boxplot", plot_handling_time_box(df),
         "Handling time is worth checking separately from response delay.",
         "Longer calls are not always the reason for bad satisfaction, but the spread is informative.",
         "Speed helps, but the quality of resolution still matters more than just call length."),
        ("Chart 7 - Tenure bucket", plot_tenure_satisfaction(df),
         "Agent tenure is a good proxy for experience and training stage.",
         "Experience buckets show different satisfaction patterns.",
         "This supports better coaching and shift allocation."),
        ("Chart 8 - Shift vs satisfaction", plot_shift_satisfaction(df),
         "Shift-based patterns can uncover workload pressure or staffing imbalance.",
         "Some shifts contribute more dissatisfaction than others.",
         "It is worth checking if a small staffing change could reduce friction."),
        ("Chart 9 - City volume", plot_city_top(df),
         "Customer city helps show where the support workload is concentrated.",
         "A few cities dominate the interaction volume.",
         "That helps decide where local service pressure is highest."),
        ("Chart 10 - Item price distribution", plot_price_distribution(df),
         "Item price adds a business value layer to the support interaction.",
         "The order-value spread is wide, so not every support case has the same sensitivity.",
         "High-value orders may deserve faster handling and stricter escalation rules."),
        ("Chart 11 - Hour bucket vs satisfaction", plot_hour_satisfaction(df),
         "Time of day can be linked to staffing load and response quality.",
         "Peak times tend to show more dissatisfaction.",
         "That points to scheduling and queue management as practical fixes."),
        ("Chart 12 - Missing values", plot_missing_values(df),
         "Missingness is easier to understand visually than with a long table.",
         "Several operational fields are partially missing, especially in text and city-related columns.",
         "This is normal in support logs, but it still needs a careful preprocessing choice."),
        ("Chart 13 - Response delay scatter", plot_response_delay_scatter(df),
         "This scatter checks whether expensive orders and slower replies overlap.",
         "There is visible spread, and dissatisfaction is not random across the plot.",
         "It helps justify faster escalation for risky cases."),
        ("Chart 14 - Correlation heatmap", plot_corr_heatmap(df),
         "Correlation helps me see which numeric fields move together.",
         "Delay, handling time, and time-derived fields show meaningful structure.",
         "That is why I compare linear and non-linear models later on."),
        ("Chart 15 - Dissatisfaction by sub-category", plot_top_subcat_dissatisfaction(df),
         "This puts the problem sub-categories in rank order by dissatisfaction rate.",
         "A few sub-categories clearly stand out as repeat pain points.",
         "That is the best place to focus a playbook instead of spreading effort too thin."),
    ]

    for title, fig, why, insight, impact in chart_specs:
        show_chart(title, why, insight, impact, fig)


def feature_engineering_section(df: pd.DataFrame) -> None:
    st.subheader("Feature Engineering")
    feats = pd.DataFrame(
        {
            "feature": [
                "response_delay_mins",
                "time_to_survey_mins",
                "reported_hour",
                "response_hour",
                "remarks_len",
                "remarks_words",
                "city_bucket",
                "sub_category_bucket",
                "price_bucket",
                "handling_bucket",
                "hour_bucket",
            ],
            "why it helps": [
                "captures speed of first response",
                "captures the support journey length",
                "shows the time of the issue",
                "shows the time of the reply",
                "gives light text signal from customer remarks",
                "keeps remark detail without a full NLP stack",
                "controls high-cardinality city labels",
                "controls high-cardinality sub-categories",
                "turns price into easier business bands",
                "turns handling time into readable operational bands",
                "captures time-of-day workload patterns",
            ],
        }
    )
    st.dataframe(feats, use_container_width=True, hide_index=True)


def modeling_section(df: pd.DataFrame) -> None:
    st.subheader("Modeling")
    st.markdown(
        """
        <div class="card">
            <h3>How the models are built</h3>
            <div class="small_note">
                I use a train-test split with stratification because the target is imbalanced.
                Logistic Regression gives a clean baseline.
                Random Forest and HistGradientBoosting add non-linearity.
                If XGBoost is available, I include it as the final stronger structured-data model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_df, best_model, test_pack, feature_cols = train_models(df)
    x_test = test_pack.drop(columns=["_target"])
    y_test = test_pack["_target"]

    st.dataframe(metrics_df.round(3), use_container_width=True, hide_index=True)

    best_name = metrics_df.iloc[0]["model"]
    best_row = metrics_df.iloc[0]
    st.success(f"Best model by CV ROC-AUC: {best_name} | Test ROC-AUC: {best_row['roc_auc']:.3f}")

    c1, c2, c3 = st.columns(3)
    with c1:
        make_metric_card("Accuracy", fmt_pct(best_row["accuracy"]), "Held-out test set")
    with c2:
        make_metric_card("F1-score", fmt_pct(best_row["f1"]), "Balance of precision and recall")
    with c3:
        make_metric_card("ROC-AUC", fmt_pct(best_row["roc_auc"]), "Ranking quality")

    st.pyplot(plot_model_metrics(metrics_df), use_container_width=True)
    st.pyplot(plot_confusion(best_model, x_test, y_test), use_container_width=True)
    st.pyplot(plot_roc(best_model, x_test, y_test), use_container_width=True)
    st.pyplot(plot_feature_importance(best_model, feature_cols), use_container_width=True)

    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    proba = best_model.predict_proba(x_test)[:, 1]
    threshold_pred = (proba >= threshold).astype(int)
    thr_table = pd.DataFrame(
        {
            "metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "value": [
                accuracy_score(y_test, threshold_pred),
                precision_score(y_test, threshold_pred, zero_division=0),
                recall_score(y_test, threshold_pred, zero_division=0),
                f1_score(y_test, threshold_pred, zero_division=0),
            ],
        }
    )
    st.dataframe(
        thr_table.assign(value=lambda d: d["value"].map(lambda x: f"{x:.3f}")),
        use_container_width=True,
        hide_index=True,
    )

    with st.form("prediction_form"):
        st.markdown("### Try your own support case")
        c1, c2, c3 = st.columns(3)
        with c1:
            input_channel = st.selectbox("Channel", sorted(df["channel_bucket"].dropna().astype(str).unique()))
            input_category = st.selectbox("Category", sorted(df["category_bucket"].dropna().astype(str).unique()))
            input_sub = st.selectbox("Sub-category", sorted(df["sub_category_bucket"].dropna().astype(str).unique()))
        with c2:
            input_city = st.selectbox("City bucket", sorted(df["city_bucket"].dropna().astype(str).unique()))
            input_product = st.selectbox("Product bucket", sorted(df["product_bucket"].dropna().astype(str).unique()))
            input_tenure = st.selectbox("Tenure bucket", sorted(df["tenure_bucket_clean"].dropna().astype(str).unique()))
        with c3:
            input_shift = st.selectbox("Shift", sorted(df["shift_bucket"].dropna().astype(str).unique()))
            input_price = st.number_input("Item price", min_value=0.0, value=float(df["item_price_filled"].median()), step=1.0)
            input_handling = st.number_input("Handling time", min_value=0.0, value=float(df["handling_time_filled"].median()), step=1.0)
        response_delay = st.number_input(
            "Response delay in minutes",
            min_value=0.0,
            value=float(df["response_delay_mins"].median()),
            step=1.0,
        )
        submitted = st.form_submit_button("Predict satisfaction")

        if submitted:
            manual = pd.DataFrame(
                {
                    "item_price_filled": [input_price],
                    "handling_time_filled": [input_handling],
                    "response_delay_mins": [response_delay],
                    "time_to_survey_mins": [np.nan],
                    "reported_hour": [12],
                    "reported_dayofweek": [2],
                    "response_hour": [12],
                    "response_dayofweek": [2],
                    "survey_month": [6],
                    "survey_year": [2023],
                    "order_hour": [12],
                    "order_month": [6],
                    "remarks_len": [0],
                    "remarks_words": [0],
                    "has_remarks": [0],
                    "delay_flag": [int(response_delay > 60)],
                    "channel_bucket": [input_channel],
                    "category_bucket": [input_category],
                    "sub_category_bucket": [input_sub],
                    "city_bucket": [input_city],
                    "product_bucket": [input_product],
                    "tenure_bucket_clean": [input_tenure],
                    "shift_bucket": [input_shift],
                    "response_bucket": [pd.cut(
                        pd.Series([response_delay]),
                        bins=[-np.inf, 30, 60, 180, np.inf],
                        labels=["0-30 min", "30-60 min", "1-3 hr", "3+ hr"],
                    ).astype(str).iloc[0]],
                    "price_bucket": [pd.cut(
                        pd.Series([input_price]),
                        bins=[-np.inf, 500, 1500, 3000, np.inf],
                        labels=["Low", "Mid", "High", "Premium"],
                    ).astype(str).iloc[0]],
                    "handling_bucket": [pd.cut(
                        pd.Series([input_handling]),
                        bins=[-np.inf, 10, 20, 40, np.inf],
                        labels=["Very short", "Short", "Medium", "Long"],
                    ).astype(str).iloc[0]],
                    "hour_bucket": ["Morning"],
                }
            )
            prob = float(best_model.predict_proba(manual)[0, 1])
            pred = "Not Satisfied" if prob >= threshold else "Satisfied"
            st.success(f"Predicted class: {pred}")
            st.write(f"Predicted probability of dissatisfaction: {prob:.3f}")


def conclusion_section(df: pd.DataFrame, summary: dict[str, object]) -> None:
    st.subheader("Conclusion")
    st.markdown(
        """
        <div class="card">
            <h3>What the project says in plain words</h3>
            <div class="small_note">
                Flipkart support satisfaction is not random.
                It moves with the speed of response, the type of issue, the handling pattern, and the support context
                around each case.
                The data is large enough to learn from, but also messy enough to feel real.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for point in [
        "Response delay is one of the most important operational signals.",
        "High-volume issue families like returns, refunds, and order-related cases deserve the most attention.",
        "Premium orders and peak hours create extra pressure, so they need stricter handling rules.",
        "A stratified and imbalance-aware setup is the right way to judge performance.",
        "XGBoost or another tree-based model is a strong final choice when the target is not balanced.",
    ]:
        st.markdown(f"- {point}")

    summary_table = pd.DataFrame(
        {
            "item": [
                "Data status",
                "EDA",
                "Feature shaping",
                "Model comparison",
                "Deployment readiness",
            ],
            "status": [
                "Loaded and cleaned",
                "15 charts and 3 hypothesis checks",
                "Created response, timing, and bucket features",
                "Compared baseline and non-linear classifiers",
                "Saved as a reloadable joblib pipeline",
            ],
        }
    )
    st.dataframe(summary_table, use_container_width=True, hide_index=True)


def main() -> None:
    raw_df = load_data()
    df = build_features(raw_df)
    summary = dataset_summary(df)
    section = render_sidebar(summary)

    if section == "Overview":
        overview_section(df, summary)
    elif section == "Data Understanding":
        data_understanding_section(df, summary)
    elif section == "Data Wrangling":
        data_wrangling_section(df)
    elif section == "EDA":
        exploratory_section(df)
    elif section == "Hypothesis Testing":
        hypothesis_section(df)
    elif section == "Feature Engineering":
        feature_engineering_section(df)
    elif section == "Modeling":
        modeling_section(df)
    elif section == "Conclusion":
        conclusion_section(df, summary)


if __name__ == "__main__":
    main()
