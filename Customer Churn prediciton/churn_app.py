import os
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
import joblib

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Constants & defaults
# ------------------------------
ARTIFACTS = {
    "preprocess": Path("preprocess.joblib"),
    "model": Path("modelxgb.joblib"),
    "config": Path("confi.joblib"),  # {"threshold": 0.35}
}

YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "Churn",
]

MULTICAT_COLS = ["Contract", "InternetService", "PaymentMethod"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ------------------------------
# Cache helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    df = pd.read_csv(file)
    return df

@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Optional[ColumnTransformer], Optional[object], float]:
    preprocess = model = None
    threshold = 0.5
    if ARTIFACTS["preprocess"].exists():
        preprocess = joblib.load(ARTIFACTS["preprocess"])
    if ARTIFACTS["model"].exists():
        model = joblib.load(ARTIFACTS["model"])
    if ARTIFACTS["config"].exists():
        cfg = joblib.load(ARTIFACTS["config"])
        threshold = float(cfg.get("threshold", 0.5))
    return preprocess, model, threshold

# ------------------------------
# Utilities
# ------------------------------

def clean_churn_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Map Yes/No columns where present
    for col in YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].map({
                "Yes": 1, "No": 0,
                "No internet service": 0,
                "No phone service": 0
            }).astype("float64")
    # TotalCharges numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


def fit_or_use_preprocess(train_df: pd.DataFrame) -> ColumnTransformer:
    """Return a fitted ColumnTransformer. If artifact exists, load it; else fit and save."""
    preprocess, _, _ = load_artifacts()
    if preprocess is not None:
        return preprocess

    missing_cols = [c for c in MULTICAT_COLS + NUMERIC_COLS if c not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for preprocessing: {missing_cols}")

    ct = ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(drop="first", handle_unknown="ignore"), MULTICAT_COLS),
            ("scale", StandardScaler(), NUMERIC_COLS),
        ]
    )
    ct.fit(train_df[MULTICAT_COLS + NUMERIC_COLS])
    joblib.dump(ct, ARTIFACTS["preprocess"])
    return ct


def train_or_use_model(x_train_proc, y_train) -> object:
    """Return a trained XGBClassifier. If artifact exists, load it; else fit and save."""
    _, model, _ = load_artifacts()
    if model is not None:
        return model

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    xgb.fit(x_train_proc, y_train)
    joblib.dump(xgb, ARTIFACTS["model"])
    return xgb


def make_confusion_fig(y_true, y_pred) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], texttemplate="%{z}", showscale=False))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Prediction", yaxis_title="Actual")
    return fig


def pr_auc_score(y_true, y_scores) -> float:
    p, r, _ = precision_recall_curve(y_true, y_scores)
    return auc(r, p)

# ------------------------------
# Sidebar navigation
# ------------------------------
with st.sidebar:
    st.title("📉 Churn Studio")
    page = st.radio(
        "Go to",
        [
            "🏁 Get Started",
            "📦 Data & EDA",
            "🧠 Train & Evaluate",
            "🔮 Predict",
            "📊 Feature Insights",
            "⚙️ Settings",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("Tip: You can upload your CSV or use pre-trained artifacts.")

# ------------------------------
# Get Started
# ------------------------------
if page == "🏁 Get Started":
    st.title("Customer Churn Prediction")
    st.markdown(
        """
        Welcome! This app helps you **analyze churn**, **train a model**, and **predict churn risk** for customers.

        **Workflow:**
        1. Upload your Telco-style dataset on **Data & EDA**
        2. Train or use existing model on **Train & Evaluate**
        3. Make single/batch predictions in **Predict**
        4. Explore drivers of churn in **Feature Insights**
        """
    )
    with st.expander("Required columns (minimum)"):
        st.write({
            "Binary/Yes-No": [c for c in YES_NO_COLS if c != "Churn"],
            "Categorical": MULTICAT_COLS,
            "Numeric": NUMERIC_COLS,
            "Target": ["Churn"],
        })

# ------------------------------
# Data & EDA
# ------------------------------
if page == "📦 Data & EDA":
    st.header("Upload data")
    data_file = st.file_uploader("Upload customer CSV (incl. 'Churn' if you want EDA by target)", type=["csv"]) 
    df = load_csv(data_file) if data_file else pd.DataFrame()

    if df.empty:
        st.info("Upload a CSV to continue. Example columns: Contract, InternetService, PaymentMethod, tenure, MonthlyCharges, TotalCharges, Churn…")
        st.stop()

    st.success(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(30), use_container_width=True)

    # Clean up copy for EDA
    eda_df = clean_churn_df(df)

    st.subheader("Overall Churn Rate")
    if "Churn" in eda_df.columns:
        churn_counts = eda_df["Churn"].value_counts().rename({0: "Not Churned", 1: "Churned"})
        pie_df = pd.DataFrame({"Status": churn_counts.index, "Count": churn_counts.values})
        pie_fig = px.pie(pie_df, names="Status", values="Count", hole=0.35)
        pie_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.warning("Column 'Churn' not found — skipping target-based charts.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Contract Type")
        if set(["Contract","Churn"]).issubset(eda_df.columns):
            rate = eda_df.groupby("Contract")["Churn"].mean().reset_index()
            rate["Churn %"] = (rate["Churn"] * 100).round(2)
            fig = px.bar(rate, x="Contract", y="Churn %", text="Churn %")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(yaxis_range=[0, max(50, rate["Churn %"].max() * 1.2)])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 'Contract' and 'Churn' columns for this chart.")

    with col2:
        st.subheader("Average Monthly Charges: Churn vs Non-Churn")
        if set(["MonthlyCharges","Churn"]).issubset(eda_df.columns):
            avg_ch = eda_df.groupby("Churn")["MonthlyCharges"].mean().reset_index()
            avg_ch["Label"] = avg_ch["Churn"].map({0:"Non-Churn",1:"Churn"})
            fig2 = px.bar(avg_ch, x="Label", y="MonthlyCharges", text="MonthlyCharges")
            fig2.update_traces(texttemplate="%{text:.2f}")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Need 'MonthlyCharges' and 'Churn' for this chart.")

    # 2D/3D SVD visualization (unsupervised view of features)
    st.subheader("Dimensionality Reduction (SVD)")
    try:
        req_cols = list(set(MULTICAT_COLS + NUMERIC_COLS) & set(eda_df.columns))
        if len(req_cols) >= 4:  # basic sanity
            ct = ColumnTransformer([
                ("one_hot", OneHotEncoder(drop="first", handle_unknown="ignore"), [c for c in MULTICAT_COLS if c in eda_df.columns]),
                ("scale", StandardScaler(), [c for c in NUMERIC_COLS if c in eda_df.columns]),
            ])
            X = ct.fit_transform(eda_df[[c for c in MULTICAT_COLS + NUMERIC_COLS if c in eda_df.columns]])
            if X.shape[1] >= 3:
                svd3 = TruncatedSVD(n_components=3, random_state=42)
                X3 = svd3.fit_transform(X)
                df3 = pd.DataFrame(X3, columns=["PC1","PC2","PC3"]) 
                if "Churn" in eda_df.columns:
                    df3["Churn"] = eda_df["Churn"].values
                    fig3 = px.scatter_3d(df3, x="PC1", y="PC2", z="PC3", color="Churn", opacity=0.6)
                else:
                    fig3 = px.scatter_3d(df3, x="PC1", y="PC2", z="PC3", opacity=0.6)
                st.plotly_chart(fig3, use_container_width=True)

                svd2 = TruncatedSVD(n_components=2, random_state=42)
                X2 = svd2.fit_transform(X)
                df2 = pd.DataFrame(X2, columns=["PC1","PC2"]) 
                if "Churn" in eda_df.columns:
                    df2["Churn"] = eda_df["Churn"].values
                    fig2d = px.scatter(df2, x="PC1", y="PC2", color="Churn", opacity=0.6,
                                       title="Customer Churn — 2D SVD Visualization")
                else:
                    fig2d = px.scatter(df2, x="PC1", y="PC2", opacity=0.6,
                                       title="Customer Churn — 2D SVD Visualization")
                fig2d.update_traces(marker=dict(size=5))
                st.plotly_chart(fig2d, use_container_width=True)
        else:
            st.info("Need more feature columns to create SVD plots.")
    except Exception as e:
        st.warning(f"SVD visualization skipped: {e}")

# ------------------------------
# Train & Evaluate
# ------------------------------
if page == "🧠 Train & Evaluate":
    st.header("Model training & evaluation")
    data_file = st.file_uploader("Upload labeled CSV for training (must include 'Churn')", type=["csv"], key="train_upload")
    df = load_csv(data_file) if data_file else pd.DataFrame()

    if df.empty:
        st.info("Upload a dataset to train/evaluate or use existing artifacts on the Predict page.")
        st.stop()

    df = clean_churn_df(df)
    if "Churn" not in df.columns:
        st.error("'Churn' column is required for training.")
        st.stop()

    X = df.drop("Churn", axis=1)
    y = df["Churn"].astype(int)

    st.write("**Train / test split (70/30)**")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    with st.spinner("Preparing features…"):
        ct = fit_or_use_preprocess(x_train)
        xtr = ct.transform(x_train[MULTICAT_COLS + NUMERIC_COLS])
        xte = ct.transform(x_test[MULTICAT_COLS + NUMERIC_COLS])

    algo = st.selectbox("Algorithm", ["XGBoost (recommended)", "Logistic Regression"], index=0)

    with st.spinner("Training model… this is usually quick"):
        if algo.startswith("XGBoost"):
            model = train_or_use_model(xtr, y_train)
        else:
            lr = LogisticRegression(solver="saga", max_iter=1000, class_weight="balanced", n_jobs=-1)
            lr.fit(xtr, y_train)
            model = lr

    # Evaluate
    y_proba = model.predict_proba(xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(xte)
    default_threshold = 0.5
    if ARTIFACTS["config"].exists():
        cfg = joblib.load(ARTIFACTS["config"])
        default_threshold = float(cfg.get("threshold", 0.5))

    thr = st.slider("Decision threshold", 0.05, 0.95, float(default_threshold), 0.01)
    y_pred = (y_proba >= thr).astype(int)

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    with colB:
        st.metric("PR AUC", f"{pr_auc_score(y_test, y_proba):.3f}")
    with colC:
        st.metric("Threshold", f"{thr:.2f}")

    st.plotly_chart(make_confusion_fig(y_test, y_pred), use_container_width=True)

    rep = classification_report(y_test, y_pred, output_dict=True, digits=3)
    rep_df = pd.DataFrame(rep).T
    st.dataframe(rep_df, use_container_width=True)

    save_cfg = st.checkbox("Save this threshold as default", value=False)
    if save_cfg:
        joblib.dump({"threshold": float(thr)}, ARTIFACTS["config"])
        st.success("Saved default threshold.")

# ------------------------------
# Predict
# ------------------------------
if page == "🔮 Predict":
    st.header("Predict churn risk")
    preprocess, model, default_thr = load_artifacts()

    if preprocess is None or model is None:
        st.warning("Pre-trained artifacts not found. Go to 'Train & Evaluate' to fit and save them, or upload a labeled dataset there.")

    tab1, tab2 = st.tabs(["Single customer", "Batch CSV"])

    with tab1:
        st.subheader("Single-customer form")
        col1, col2 = st.columns(2)

        with col1:
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]) 
            InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=1)
            PaymentMethod = st.selectbox("PaymentMethod", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            tenure = st.number_input("tenure (months)", min_value=0, max_value=120, value=12)
        with col2:
            MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=75.0)
            TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=900.0)
            PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"], index=0)

        # Optional binary services
        bin_cols = [
            "Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity",
            "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
        ]
        bin_vals = {}
        with st.expander("More services (optional)"):
            colA, colB = st.columns(2)
            for i, c in enumerate(bin_cols):
                with (colA if i % 2 == 0 else colB):
                    bin_vals[c] = st.selectbox(c, ["Yes", "No"], key=c)

        if st.button("Predict churn risk", use_container_width=True):
            if preprocess is None or model is None:
                st.error("Artifacts missing. Train a model first.")
            else:
                row = {
                    "Contract": Contract,
                    "InternetService": InternetService,
                    "PaymentMethod": PaymentMethod,
                    "tenure": tenure,
                    "MonthlyCharges": MonthlyCharges,
                    "TotalCharges": TotalCharges,
                    "PaperlessBilling": bin_vals.get("PaperlessBilling", PaperlessBilling),
                }
                # include optional binaries if provided
                row.update(bin_vals)
                df_row = pd.DataFrame([row])

                # ensure mapping for binaries
                for c in YES_NO_COLS:
                    if c in df_row.columns:
                        df_row[c] = df_row[c].map({"Yes":1, "No":0, "No internet service":0, "No phone service":0})

                Xrow = preprocess.transform(df_row[MULTICAT_COLS + NUMERIC_COLS])
                proba = model.predict_proba(Xrow)[0,1]
                pred = int(proba >= default_thr)
                st.metric("Churn probability", f"{proba:.3f}")
                st.metric("Prediction", "Churn" if pred==1 else "Not Churn")

    with tab2:
        st.subheader("Batch predictions from CSV")
        up = st.file_uploader("Upload customer CSV (no 'Churn' column required)", type=["csv"], key="batch")
        if up and preprocess is not None and model is not None:
            bdf = pd.read_csv(up)
            raw = bdf.copy()
            # map binaries if present
            for c in YES_NO_COLS:
                if c in bdf.columns:
                    bdf[c] = bdf[c].map({"Yes":1, "No":0, "No internet service":0, "No phone service":0})
            Xb = preprocess.transform(bdf[MULTICAT_COLS + NUMERIC_COLS])
            proba = model.predict_proba(Xb)[:,1]
            pred = (proba >= default_thr).astype(int)
            out = raw.copy()
            out["churn_proba"] = proba
            out["prediction"] = pred
            st.dataframe(out.head(50), use_container_width=True)
            csv = out.to_csv(index=False).encode()
            st.download_button("Download predictions CSV", csv, file_name="churn_predictions.csv")
        elif up and (preprocess is None or model is None):
            st.error("Artifacts missing. Train a model on the previous tab first.")

# ------------------------------
# Feature Insights
# ------------------------------
if page == "📊 Feature Insights":
    st.header("Feature importance (XGBoost)")
    preprocess, model, _ = load_artifacts()
    if preprocess is None or model is None:
        st.warning("Artifacts not found. Train a model to view feature importances.")
        st.stop()

    # Build feature names for OHE + numeric
    try:
        ohe = preprocess.named_transformers_["one_hot"]
        ohe_names = ohe.get_feature_names_out([c for c in MULTICAT_COLS])
        feature_names = np.r_[ohe_names, NUMERIC_COLS]
    except Exception:
        feature_names = [f"f{i}" for i in range(model.n_features_in_)]

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(imp)[::-1]
        topk = st.slider("Show top K features", 5, min(50, len(feature_names)), 20)
        df_imp = pd.DataFrame({"Feature": np.array(feature_names)[order][:topk], "Importance": imp[order][:topk]})
        fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_imp, use_container_width=True)
    else:
        st.info("Current model does not expose feature_importances_. Try XGBoost.")

# ------------------------------
# Settings
# ------------------------------
if page == "⚙️ Settings":
    st.header("App settings & artifacts")
    st.write({k: str(v) for k, v in ARTIFACTS.items()})
    if st.button("Clear cached data", type="secondary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared Streamlit caches.")

    # Theme hint
    st.caption("This app follows your Streamlit theme (light/dark). Adjust in Settings → Theme if needed.")
