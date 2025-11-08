# app.py - Diabetes Prediction UI (dark, readable, animated)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
import pickle
import requests
import time

# try optional lottie
try:
    from streamlit_lottie import st_lottie  # type: ignore
    LOTTIE_OK = True
except Exception:
    LOTTIE_OK = False

# ------------- Page config -------------
st.set_page_config(page_title="Diabetes Predictor — Sleek UI", page_icon="🩺", layout="wide")

# ------------- Robust CSS: dark + readable -------------
# This CSS keeps a modern dark theme while ensuring text is readable anywhere.
st.markdown(
    """
    <style>
    /* page */
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg,#07111a 0%, #0b1220 100%) !important; color: #e6eef8; }
    [data-testid="stSidebar"] { background: #07111a !important; color: #e6eef8; border-right:1px solid rgba(255,255,255,0.03); }

    /* enforce readable text everywhere */
    h1,h2,h3,h4,h5,h6, p, span, label, .stText, .stMarkdown { color: #e6eef8 !important; text-shadow:none !important; }

    /* card style for sections */
    .card { background: rgba(255,255,255,0.02) !important; padding: 12px; border-radius: 12px; border:1px solid rgba(255,255,255,0.03); }

    /* inputs */
    input, textarea, select, .stNumberInput input, .stTextInput input { background:#0f1720 !important; color:#e6eef8 !important; border:1px solid rgba(255,255,255,0.03) !important; border-radius:8px !important; }
    input::placeholder, textarea::placeholder { color:#9aa7bf !important; }

    /* buttons */
    .stButton>button { background: linear-gradient(135deg,#00B4DB,#0083B0) !important; color:#fff !important; border:none !important; border-radius:10px !important; padding:8px 14px !important; font-weight:700 !important; }
    .stButton>button:hover { transform: scale(1.02); }

    /* plotly */
    .plotly-graph-div { background: transparent !important; }

    /* dataframes */
    div[data-testid="stDataFrame"], .stDataFrameContainer { background: rgba(255,255,255,0.01) !important; color:#e6eef8 !important; border-radius:8px !important; }

    /* remove huge shadows/glows that might make text unreadable */
    * { box-shadow: none !important; filter: none !important; backdrop-filter: none !important; }

    /* fallback ensure contrast on lighter areas */
    .force-dark-text { color: #07111a !important; font-weight:700 !important; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------- helper: load lottie (safe) -------------
def load_lottie_url(url: str, timeout=3):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# ------------- Data & model loading helpers -------------
DATA_FILE = Path(__file__).resolve().parent / "diabetes.csv"
MODEL_CANDIDATES = [
    "best_rf_pipeline.joblib",
    "best_rf_pipeline.pkl",
    "diabetes_model.joblib",
    "diabetes_model.pkl",
    "best_model.joblib",
    "best_model.pkl",
    "model.joblib",
    "model.pkl",
]

@st.cache_data
def load_dataset():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return None

@st.cache_data
def try_load_model():
    base = Path(__file__).resolve().parent
    for name in MODEL_CANDIDATES:
        p = base / name
        if p.exists():
            try:
                if p.suffix == ".joblib":
                    m = joblib.load(p)
                else:
                    try:
                        m = joblib.load(p)
                    except Exception:
                        with open(p, "rb") as f:
                            m = pickle.load(f)
                return m, str(p)
            except Exception as e:
                st.session_state.setdefault("model_errors", []).append(f"{name}: {e}")
    return None, None

# ------------- Load data & model -------------
data_df = load_dataset()
model, model_path = try_load_model()

# ------------- Header -------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<h1 style='margin:0'>🩺 Diabetes Prediction — Sleek Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Interactive demo — probability, ROC, feature importance, and EDA</div>", unsafe_allow_html=True)
with col2:
    if LOTTIE_OK:
        lottie = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_5ngs2ksb.json")
        if lottie:
            try:
                st_lottie(lottie, height=120, key="l1")
            except Exception:
                st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Diabetes_Mellitus_Icon.png", width=120)
        else:
            st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Diabetes_Mellitus_Icon.png", width=120)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Diabetes_Mellitus_Icon.png", width=120)

st.markdown("---")

# ------------- If no model, train (fast demo) -------------
# If you have a saved model, prefer loading it; otherwise train quickly for demo.
if model is None:
    if data_df is None:
        st.error("Missing dataset: place 'diabetes.csv' next to this app.py (or provide a saved model).")
        st.stop()
    # quick train (fast-ish)
    X_all = data_df.iloc[:, :-1]
    y_all = data_df.iloc[:, -1]
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    with st.spinner("Training demo RandomForest model..."):
        model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, class_weight="balanced_subsample")
        model.fit(X_train, y_train)
        time.sleep(0.6)
    trained_from = "trained_on_spot"
else:
    # if loaded model, we will still need X_test for diagnostics (if data available)
    trained_from = model_path

# ------------- Sidebar inputs -------------
st.sidebar.markdown("## 🧬 Patient inputs")
Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
Glucose = st.sidebar.slider("Glucose", min_value=0, max_value=250, value=120)
BloodPressure = st.sidebar.slider("Blood Pressure", min_value=0, max_value=140, value=70)
SkinThickness = st.sidebar.slider("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.sidebar.slider("Insulin", min_value=0, max_value=900, value=79)
BMI = st.sidebar.slider("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
Age = st.sidebar.slider("Age", min_value=10, max_value=100, value=30)
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("💉 Predict")

# ------------- Prediction function -------------
def predict_single(inp):
    X = pd.DataFrame([inp], columns=list(inp.keys()))
    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[:,1][0])
        else:
            pr = int(model.predict(X)[0])
            prob = 0.99 if pr==1 else 0.01
        used_model = True if model is not None else False
    except Exception:
        # fallback heuristic (safe)
        g = inp["Glucose"]; b = inp["BMI"]; a = inp["Age"]; p = inp["Pregnancies"]
        score = max(0, g-100)*0.008 + max(0, b-25)*0.01 + max(0, a-40)*0.006 + p*0.02
        prob = 1/(1+np.exp(-(score-0.5)))
        prob = float(np.clip(prob, 0, 0.99))
        used_model = False
    pred = 1 if prob >= 0.3745 else 0
    return prob, pred, used_model

# ------------- Main layout: left (controls/result) & right (diagnostics) -------------
left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Patient Overview")
    st.write("Adjust the inputs in the sidebar and press **Predict** to generate a probability and classification.")
    # show a neat table summary
    in_dict = {
        "Pregnancies": Pregnancies, "Glucose": Glucose, "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness, "Insulin": Insulin, "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age
    }
    st.dataframe(pd.DataFrame([in_dict]).T.rename(columns={0:"Value"}), width=380)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        prob, pred, used_model_flag = predict_single(in_dict)
        st.session_state["last_pred"] = {"prob": prob, "pred": pred, "used_model": used_model_flag}
    if "last_pred" not in st.session_state:
        st.session_state["last_pred"] = None

    if st.session_state["last_pred"]:
        res = st.session_state["last_pred"]
        prob = res["prob"]; pred = res["pred"]; used_model_flag = res["used_model"]

        # Result card
        st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Prediction Result")
        st.markdown(f"**Model source:** `{trained_from}`")
        st.markdown(f"**Probability:** **{prob*100:.2f}%**")
        if pred == 1:
            st.error("🩸 Diabetes likely (positive)")
        else:
            st.success("💚 Low risk (negative)")

        # Animated gauge (plotly)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            number={"suffix":"%"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"#06B6D4"},
                "steps":[{"range":[0,37.45],"color":"#10b981"},
                         {"range":[37.45,70],"color":"#f59e0b"},
                         {"range":[70,100],"color":"#ef4444"}],
                "threshold":{"line":{"color":"red","width":4},"value":37.45}
            },
            title={"text":"Predicted Risk"}
        ))
        gauge.update_layout(template="plotly_dark", margin=dict(t=10,b=10,l=10,r=10), height=300)
        st.plotly_chart(gauge, use_container_width=True)

        # Progress (animated)
        with st.container():
            prog = int(prob*100)
            my_bar = st.progress(0)
            for i in range(prog+1):
                my_bar.progress(i)
                time.sleep(0.003)
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model & Diagnostics")
    if model is not None:
        st.success(f"Model ready — source: `{trained_from}`")
        # Show ROC if dataset exists
        if data_df is not None:
            try:
                from sklearn.metrics import roc_curve, auc
                X_all = data_df.iloc[:, :-1]
                y_all = data_df.iloc[:, -1]
                if hasattr(model, "predict_proba"):
                    y_scores = model.predict_proba(X_all)[:,1]
                else:
                    y_scores = model.predict(X_all)
                fpr, tpr, thr = roc_curve(y_all, y_scores)
                roc_auc = auc(fpr, tpr)
                # annotate threshold points (every N)
                n = max(1, len(thr)//12)
                indic = np.arange(len(thr)) % n == 0
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.2f})", line=dict(color="#21D4FD")))
                roc_fig.add_trace(go.Scatter(x=fpr[indic], y=tpr[indic], mode="markers+text",
                                             text=[f"{t:.2f}" for t in thr[indic]],
                                             textposition="top center", marker=dict(size=8, color="#f59e0b"), name="Thresholds"))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False))
                roc_fig.update_layout(template="plotly_dark", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=360)
                st.plotly_chart(roc_fig, use_container_width=True)
            except Exception as e:
                st.warning("Could not compute ROC: " + str(e))
        # feature importance if available
        try:
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                cols = data_df.columns[:-1].tolist() if data_df is not None else [f"f{i}" for i in range(len(fi))]
                fi_df = pd.DataFrame({"feature": cols, "importance": fi}).sort_values("importance", ascending=True)
                fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=px.colors.sequential.Viridis, title="Feature importance")
                fig_fi.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            pass
    else:
        st.warning("Model not available and dataset missing; prediction will not be accurate.")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------- Bottom: EDA & data sample -------------
st.markdown("## Dataset & EDA")
if data_df is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("Sample rows:")
    st.dataframe(data_df.head(10))
    # histogram of Glucose
    fig_h = px.histogram(data_df, x="Glucose", nbins=40, title="Glucose distribution", marginal="box")
    fig_h.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_h, use_container_width=True)
    # correlation heatmap
    corr = data_df.corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale=px.colors.sequential.Viridis, title="Feature Correlation")
    fig_corr.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Place 'diabetes.csv' next to app.py to enable dataset preview & diagnostics.")

# ------------- Footer -------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#9aa7bf;'>Made with ❤️</div>", unsafe_allow_html=True)
