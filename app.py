import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.pkl"
CONFIG_PATH = BASE_DIR / "models" / "credit_risk_config.pkl"

MODEL_METRICS = {
    "ROC-AUC": 0.8644,
    "PR-AUC": 0.3859,
    "F1 Score": 0.3416,
    "Precision": 0.2203,
    "Recall": 0.7601,
}

MODEL_NAME = "Tuned Random Forest Classifier"
DATASET_NAME = "Give Me Some Credit"
TARGET_NAME = "SeriousDlqin2yrs"
DEFAULT_THRESHOLD = 0.50
DEFAULT_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


# ============================================================
# PROFESSIONAL UI
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at top left, #172554 0, #0b1220 34%, #070b14 72%); color:#e5e7eb; }
    .block-container { max-width:1450px; padding-top:1.25rem; padding-bottom:3rem; }
    #MainMenu, header, footer { visibility:hidden; }
    .app-header{display:flex;justify-content:space-between;align-items:center;background:rgba(15,23,42,.86);border:1px solid rgba(148,163,184,.16);border-radius:18px;padding:16px 20px;margin-bottom:14px;box-shadow:0 14px 40px rgba(0,0,0,.24);backdrop-filter:blur(12px)}
    .brand{display:flex;align-items:center;gap:12px}.brand-icon{width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,#1d4ed8,#7c3aed);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 8px 24px rgba(59,130,246,.30)}
    .brand-title{font-size:1.22rem;font-weight:800;color:#f8fafc}.brand-subtitle{font-size:.8rem;color:#94a3b8;margin-top:3px}
    .status{display:flex;align-items:center;gap:7px;background:rgba(20,83,45,.28);border:1px solid rgba(74,222,128,.26);color:#86efac;border-radius:999px;padding:7px 12px;font-size:.78rem;font-weight:700}.dot{width:8px;height:8px;border-radius:50%;background:#4ade80;box-shadow:0 0 10px rgba(74,222,128,.75)}
    .page-title{font-size:2rem;font-weight:850;color:#f8fafc;margin-bottom:2px;letter-spacing:-.025em}.page-description{font-size:.96rem;color:#94a3b8;margin-bottom:18px}.section-title{font-size:1.12rem;font-weight:780;color:#f1f5f9;margin-top:10px;margin-bottom:3px}.section-subtitle{font-size:.84rem;color:#7f8ea3;margin-bottom:12px}
    .card{background:linear-gradient(145deg,rgba(17,24,39,.96),rgba(15,23,42,.84));border:1px solid rgba(148,163,184,.13);border-radius:16px;padding:17px;box-shadow:0 12px 30px rgba(0,0,0,.20);margin-bottom:14px;position:relative;overflow:hidden}.card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#38bdf8,#6366f1)}
    .card-title{font-size:1rem;font-weight:760;color:#f8fafc;margin-bottom:6px}.card-text{font-size:.87rem;color:#a7b2c3;line-height:1.7}.card-text b{color:#dbeafe}
    .kpi{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(17,24,39,.98),rgba(15,23,42,.90));border:1px solid rgba(148,163,184,.14);border-radius:16px;padding:16px;min-height:116px;box-shadow:0 12px 30px rgba(0,0,0,.20)}.kpi::after{content:"";position:absolute;right:-28px;top:-28px;width:90px;height:90px;border-radius:50%;opacity:.13}
    .kpi-blue{border-top:2px solid #38bdf8}.kpi-blue::after{background:#38bdf8}.kpi-violet{border-top:2px solid #a78bfa}.kpi-violet::after{background:#a78bfa}.kpi-emerald{border-top:2px solid #34d399}.kpi-emerald::after{background:#34d399}.kpi-amber{border-top:2px solid #fbbf24}.kpi-amber::after{background:#fbbf24}.kpi-rose{border-top:2px solid #fb7185}.kpi-rose::after{background:#fb7185}.kpi-cyan{border-top:2px solid #22d3ee}.kpi-cyan::after{background:#22d3ee}
    .kpi-label{font-size:.74rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}.kpi-value{font-size:1.72rem;font-weight:850;color:#f8fafc;margin-top:8px}.kpi-sub{font-size:.74rem;color:#718096;margin-top:5px}
    .risk-card{background:linear-gradient(145deg,rgba(17,24,39,.99),rgba(15,23,42,.94));border:1px solid rgba(148,163,184,.16);border-radius:18px;padding:24px;text-align:center;box-shadow:0 14px 34px rgba(0,0,0,.26);margin:10px 0 14px}.risk-label{font-size:.76rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#94a3b8}.risk-value{font-size:2rem;font-weight:850;margin:5px 0 7px}.risk-prob{font-size:.92rem;color:#cbd5e1}.risk-low .risk-value{color:#34d399;text-shadow:0 0 24px rgba(52,211,153,.18)}.risk-medium .risk-value{color:#fbbf24;text-shadow:0 0 24px rgba(251,191,36,.18)}.risk-high .risk-value{color:#fb7185;text-shadow:0 0 24px rgba(251,113,133,.18)}
    .risk-framework-card.risk-low{border-left:3px solid #34d399}.risk-framework-card.risk-medium{border-left:3px solid #fbbf24}.risk-framework-card.risk-high{border-left:3px solid #fb7185}
    .dist{background:linear-gradient(145deg,rgba(17,24,39,.97),rgba(15,23,42,.88));border:1px solid rgba(148,163,184,.14);border-radius:16px;padding:15px;min-height:112px;box-shadow:0 10px 26px rgba(0,0,0,.20)}.dist.low{border-left:3px solid #34d399}.dist.medium{border-left:3px solid #fbbf24}.dist.high{border-left:3px solid #fb7185}.dist-label{font-size:.76rem;font-weight:700;color:#cbd5e1;text-transform:uppercase}.dist-value{font-size:1.5rem;font-weight:850;color:#f8fafc;margin-top:7px}.dist-sub{font-size:.75rem;color:#718096;margin-top:4px}
    .importance{background:rgba(15,23,42,.80);border:1px solid rgba(148,163,184,.11);border-radius:14px;padding:12px 15px;margin-bottom:9px}.imp-head{display:flex;justify-content:space-between;gap:12px;margin-bottom:7px}.imp-name{font-size:.84rem;font-weight:700;color:#e2e8f0}.imp-val{font-size:.76rem;color:#94a3b8;font-weight:650}.track{height:9px;background:#1e293b;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#38bdf8,#6366f1,#a78bfa);border-radius:999px;box-shadow:0 0 14px rgba(99,102,241,.35)}
    .stTabs [data-baseweb="tab-list"]{gap:6px;background:rgba(15,23,42,.75);border:1px solid rgba(148,163,184,.10);padding:6px;border-radius:14px}.stTabs [data-baseweb="tab"]{color:#94a3b8;border-radius:10px;padding:8px 14px;font-weight:700;font-size:.88rem}.stTabs [aria-selected="true"]{color:#f8fafc !important;background:linear-gradient(135deg,#1e3a8a,#4c1d95);box-shadow:0 6px 18px rgba(59,130,246,.18)}
    div[data-testid="stNumberInput"] label,div[data-testid="stTextInput"] label,div[data-testid="stFileUploader"] label{color:#cbd5e1 !important}div[data-testid="stNumberInput"] input,div[data-testid="stTextInput"] input{background:#0f172a !important;color:#e5e7eb !important;border-color:#334155 !important}div[data-testid="stFileUploader"] section{background:#0f172a !important;border:1px dashed #475569 !important;color:#cbd5e1 !important}
    .stProgress > div > div > div > div{background:linear-gradient(90deg,#22d3ee,#6366f1,#a78bfa)}
    .stButton > button{border-radius:8px !important;font-weight:700 !important;min-height:44px !important;background:#1e293b !important;color:#93c5fd !important;border:1px solid #475569 !important}
    .stButton > button:hover{background:#334155 !important;color:#ffffff !important}
    div[data-testid="stFormSubmitButton"] button{background:#1e293b !important;color:#93c5fd !important;border:1px solid #475569 !important;border-radius:8px !important;font-weight:700 !important}
    div[data-testid="stFormSubmitButton"] button:hover{background:#334155 !important;color:#ffffff !important}
    div[data-testid="stFileUploader"] button{background:#1e293b !important;color:#93c5fd !important;border:1px solid #475569 !important;border-radius:8px !important;font-weight:700 !important}
    div[data-testid="stFileUploader"] button:hover{background:#334155 !important;color:#ffffff !important}
    .stAlert{background:rgba(15,23,42,.84) !important;border:1px solid rgba(148,163,184,.14) !important;color:#dbeafe !important}.footer{text-align:center;color:#64748b;font-size:.76rem;border-top:1px solid rgba(148,163,184,.12);margin-top:35px;padding-top:14px}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD TRAINED ASSETS
# ============================================================
@st.cache_resource

def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    return model, config


try:
    model, config = load_assets()
except Exception as exc:
    st.error("Unable to load the trained model.")
    st.exception(exc)
    st.stop()

FEATURES = config.get("features_list", DEFAULT_FEATURES)
BEST_THRESHOLD = float(config.get("best_threshold", DEFAULT_THRESHOLD))


# ============================================================
# HELPERS
# ============================================================
def feature_label(name: str) -> str:
    labels = {
        "RevolvingUtilizationOfUnsecuredLines": "Revolving Utilization",
        "age": "Age",
        "NumberOfTime30-59DaysPastDueNotWorse": "30–59 Days Past Due",
        "DebtRatio": "Debt Ratio",
        "MonthlyIncome": "Monthly Income",
        "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines / Loans",
        "NumberOfTimes90DaysLate": "90+ Days Late",
        "NumberRealEstateLoansOrLines": "Real Estate Loans / Lines",
        "NumberOfTime60-89DaysPastDueNotWorse": "60–89 Days Past Due",
        "NumberOfDependents": "Number of Dependents",
    }
    return labels.get(name, name.replace("_", " ").title())


def get_risk_level(probability: float) -> str:
    low_boundary = BEST_THRESHOLD * 0.70
    if probability < low_boundary:
        return "Low Risk"
    if probability < BEST_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


def get_risk_emoji(risk: str) -> str:
    return {"Low Risk": "🟢", "Medium Risk": "🟠", "High Risk": "🔴"}.get(risk, "⚪")


def get_risk_description(risk: str) -> str:
    low_boundary = BEST_THRESHOLD * 0.70
    return {
        "Low Risk": f"Predicted probability is below the lower application boundary of {low_boundary:.2f}.",
        "Medium Risk": f"Predicted probability is between {low_boundary:.2f} and the model threshold of {BEST_THRESHOLD:.2f}.",
        "High Risk": f"Predicted probability is at or above the model threshold of {BEST_THRESHOLD:.2f}.",
    }[risk]


def feature_importance_df() -> pd.DataFrame:
    try:
        classifier = model.named_steps["classifier"]
        return (
            pd.DataFrame({"Feature": FEATURES, "Importance": classifier.feature_importances_})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


IMPORTANCE_DF = feature_importance_df()


def render_importance_bars(df: pd.DataFrame, top_n: int = 10) -> None:
    if df.empty:
        st.info("Feature importance is unavailable.")
        return
    view = df.head(top_n).copy()
    max_value = max(float(view["Importance"].max()), 1e-12)
    chunks = []
    for _, row in view.iterrows():
        name = html.escape(feature_label(str(row["Feature"])))
        value = float(row["Importance"])
        width = max(2.0, value / max_value * 100)
        chunks.append(
            f'<div class="importance"><div class="imp-head"><span class="imp-name">{name}</span>'
            f'<span class="imp-val">{value:.4f}</span></div><div class="track">'
            f'<div class="fill" style="width:{width:.2f}%"></div></div></div>'
        )
    st.markdown("".join(chunks), unsafe_allow_html=True)


def predict_single(df: pd.DataFrame):
    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = int(probability >= BEST_THRESHOLD)
    return probability, prediction, get_risk_level(probability)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(df)[:, 1]
    result = df.copy()
    result["Risk Probability"] = probabilities.round(6)
    result["Risk Probability (%)"] = (probabilities * 100).round(2)
    result["Prediction Class"] = (probabilities >= BEST_THRESHOLD).astype(int)
    result["Risk Level"] = [get_risk_level(float(p)) for p in probabilities]
    return result


def validate_csv(df: pd.DataFrame):
    return [feature for feature in FEATURES if feature not in df.columns]


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="app-header">
        <div class="brand">
            <div class="brand-icon">💳</div>
            <div>
                <div class="brand-title">AI Credit Risk Predictor</div>
                <div class="brand-subtitle">Machine Learning powered credit risk assessment</div>
            </div>
        </div>
        <div class="status"><span class="dot"></span>Model Online</div>
    </div>
    """,
    unsafe_allow_html=True,
)


dashboard, assessment, batch, performance, about = st.tabs(
    ["📊 Dashboard", "🧮 Assessment", "📁 Batch Prediction", "📈 Performance", "ℹ️ About"]
)


# ============================================================
# DASHBOARD
# ============================================================
with dashboard:
    st.markdown('<div class="page-title">Risk Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-description">A concise overview of model quality, risk drivers, and system capabilities.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    kpis = [
        ("ROC-AUC", f"{MODEL_METRICS['ROC-AUC']:.4f}", "Ranking performance", "kpi-blue"),
        ("PR-AUC", f"{MODEL_METRICS['PR-AUC']:.4f}", "Imbalanced-class performance", "kpi-violet"),
        ("Recall", f"{MODEL_METRICS['Recall'] * 100:.2f}%", "Positive cases detected", "kpi-emerald"),
        ("Decision Threshold", f"{BEST_THRESHOLD:.2f}", "High-risk boundary", "kpi-amber"),
    ]
    for col, (label, value, sub, accent) in zip(cols, kpis):
        with col:
            st.markdown(
                f'<div class="kpi {accent}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card"><div class="card-title">What does this application do?</div>'
            f'<div class="card-text">The system estimates the probability that a customer may experience serious delinquency within the prediction horizon represented by the training dataset. It supports individual assessment and CSV-based batch prediction using the saved preprocessing and Random Forest pipeline.</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card"><div class="card-title">{MODEL_NAME}</div><div class="card-text">'
            f'<b>Dataset:</b> {DATASET_NAME}<br><b>Input Features:</b> {len(FEATURES)}<br>'
            f'<b>Target:</b> {TARGET_NAME}<br><b>Threshold:</b> {BEST_THRESHOLD:.2f}<br>'
            f'<b>PR-AUC:</b> {MODEL_METRICS["PR-AUC"]:.4f}<br><b>Recall:</b> {MODEL_METRICS["Recall"] * 100:.2f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Top Risk Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Global Random Forest feature importance from the trained model.</div>', unsafe_allow_html=True)
    top = IMPORTANCE_DF.head(3)
    top_cols = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        accent = ["kpi-blue", "kpi-violet", "kpi-rose"][i % 3]
        with top_cols[i]:
            st.markdown(
                f'<div class="kpi {accent}"><div class="kpi-label">#{i + 1} Feature</div>'
                f'<div class="kpi-value" style="font-size:1.1rem">{html.escape(feature_label(str(row["Feature"])))}</div>'
                f'<div class="kpi-sub">Importance: {float(row["Importance"]):.4f}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Risk Interpretation Framework</div>', unsafe_allow_html=True)
    low = BEST_THRESHOLD * 0.70
    rc = st.columns(3)
    risk_info = [
        ("🟢", "Low Risk", f"Probability below {low:.2f}"),
        ("🟠", "Medium Risk", f"Probability from {low:.2f} to below {BEST_THRESHOLD:.2f}"),
        ("🔴", "High Risk", f"Probability at or above {BEST_THRESHOLD:.2f}"),
    ]
    for col, (emoji, label, desc) in zip(rc, risk_info):
        with col:
            st.markdown(
                f'<div class="card risk-framework-card risk-{label.split()[0].lower()}"><div class="card-title">{emoji} {label}</div><div class="card-text">{desc}.</div></div>',
                unsafe_allow_html=True,
            )
    st.info(
        f"The Low / Medium / High labels are an application-level presentation layer. The underlying binary model decision uses the configured threshold of {BEST_THRESHOLD:.2f}."
    )


# ============================================================
# ASSESSMENT
# ============================================================
with assessment:
    st.markdown('<div class="page-title">Customer Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Enter customer financial information to estimate delinquency risk.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Customer Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter the customer profile used by the trained model.</div>', unsafe_allow_html=True)

    with st.form("assessment_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            revolving_utilization = st.number_input("Revolving Utilization", min_value=0.0, value=0.50, step=0.01)
        with c2:
            age = st.number_input("Age", min_value=18, max_value=120, value=35, step=1)
        with c3:
            debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.50, step=0.01)

        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0, step=500.0)
        with c2:
            open_credit_lines = st.number_input("Open Credit Lines / Loans", min_value=0, max_value=200, value=5, step=1)
        with c3:
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=30, value=0, step=1)

        c1, c2, c3 = st.columns(3)
        with c1:
            late_30_59 = st.number_input("30–59 Days Past Due", min_value=0, max_value=100, value=0, step=1)
        with c2:
            late_90 = st.number_input("90+ Days Late", min_value=0, max_value=100, value=0, step=1)
        with c3:
            late_60_89 = st.number_input("60–89 Days Past Due", min_value=0, max_value=100, value=0, step=1)

        c1, c2 = st.columns(2)
        with c1:
            real_estate_loans = st.number_input("Real Estate Loans / Lines", min_value=0, max_value=100, value=0, step=1)
        with c2:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents,
        }])[FEATURES]

        probability, prediction, risk = predict_single(input_df)
        probability_pct = probability * 100

        st.markdown('<div class="section-title">Assessment Result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="risk-card risk-{risk.split()[0].lower()}"><div class="risk-label">Predicted Risk Category</div>'
            f'<div class="risk-value">{get_risk_emoji(risk)} {risk}</div>'
            f'<div class="risk-probability">Estimated delinquency probability: <b>{probability_pct:.2f}%</b></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("**Risk Probability**")
        st.progress(min(max(probability, 0.0), 1.0))
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Probability", f"{probability_pct:.2f}%")
        with p2:
            st.metric("Model Decision", "High Risk" if prediction else "Lower Risk")
        with p3:
            st.metric("Threshold", f"{BEST_THRESHOLD:.2f}")

        st.markdown('<div class="section-title">Interpretation</div>', unsafe_allow_html=True)
        st.info(get_risk_description(risk))

        with st.expander("View submitted customer data"):
            display = input_df.copy()
            display.columns = [feature_label(c) for c in display.columns]
            st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Global Model Risk Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Global model-level drivers are not a personalized causal explanation of this individual prediction.</div>', unsafe_allow_html=True)
        render_importance_bars(IMPORTANCE_DF, 10)

        st.warning(
            "Educational and portfolio use only. This prediction should not be treated as the sole basis for real-world lending or financial decisions."
        )


# ============================================================
# BATCH PREDICTION
# ============================================================
with batch:
    st.markdown('<div class="page-title">Batch Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Upload a CSV containing multiple customer records and generate predictions in one operation.</div>', unsafe_allow_html=True)

    with st.expander("View required CSV columns"):
        st.code("\n".join(FEATURES), language="text")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error("The uploaded file could not be read as CSV.")
            st.exception(exc)
            st.stop()

        missing = validate_csv(uploaded_df)
        if missing:
            st.error("The uploaded CSV is missing required columns.")
            st.write(missing)
            st.stop()

        batch_input = uploaded_df[FEATURES].copy()
        conversion_errors = []
        for column in FEATURES:
            original = batch_input[column]
            converted = pd.to_numeric(original, errors="coerce")
            bad = original.notna() & converted.isna()
            if bad.any():
                conversion_errors.append(column)
            batch_input[column] = converted

        if conversion_errors:
            st.error("Non-numeric values were found in these columns:")
            st.write(conversion_errors)
            st.stop()

        st.markdown('<div class="section-title">Uploaded Dataset</div>', unsafe_allow_html=True)
        st.dataframe(uploaded_df.head(10), use_container_width=True, hide_index=True)
        st.caption(f"Showing first 10 rows of {len(uploaded_df):,} uploaded records.")

        if st.button("🚀 Generate Batch Predictions", use_container_width=True):
            with st.spinner("Running the credit-risk model..."):
                results = predict_batch(batch_input)

            total = len(results)
            high = int((results["Risk Level"] == "High Risk").sum())
            medium = int((results["Risk Level"] == "Medium Risk").sum())
            low_count = int((results["Risk Level"] == "Low Risk").sum())
            high_pct = high / total * 100 if total else 0

            st.success(f"Predictions generated successfully for {total:,} customers.")

            cols = st.columns(4)
            summary = [
                ("Total Customers", f"{total:,}", "", "kpi-cyan"),
                ("High Risk", f"{high:,}", "", "kpi-rose"),
                ("Medium Risk", f"{medium:,}", "", "kpi-amber"),
                ("High Risk %", f"{high_pct:.2f}%", "", "kpi-violet"),
            ]
            for col, (label, value, _, accent) in zip(cols, summary):
                with col:
                    st.markdown(
                        f'<div class="kpi {accent}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
            dc = st.columns(3)
            dist = [("🟢 Low Risk", low_count, "low"), ("🟠 Medium Risk", medium, "medium"), ("🔴 High Risk", high, "high")]
            for col, (label, count, dist_class) in zip(dc, dist):
                pct = count / total * 100 if total else 0
                with col:
                    st.markdown(
                        f'<div class="dist {dist_class}"><div class="dist-label">{label}</div><div class="dist-value">{count:,}</div><div class="dist-sub">{pct:.2f}% of uploaded customers</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download Prediction Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="credit_risk_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# PERFORMANCE
# ============================================================
with performance:
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Evaluation metrics and model-level information from the final out-of-sample evaluation.</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        ("ROC-AUC", MODEL_METRICS["ROC-AUC"], "kpi-blue"),
        ("PR-AUC", MODEL_METRICS["PR-AUC"], "kpi-violet"),
        ("F1 Score", MODEL_METRICS["F1 Score"], "kpi-cyan"),
        ("Precision", MODEL_METRICS["Precision"], "kpi-amber"),
        ("Recall", MODEL_METRICS["Recall"], "kpi-emerald"),
    ]
    for col, (label, value, accent) in zip(cols, metrics):
        formatted = f"{value * 100:.2f}%" if label == "Recall" else f"{value:.4f}"
        with col:
            st.markdown(
                f'<div class="kpi {accent}"><div class="kpi-label">{label}</div><div class="kpi-value">{formatted}</div></div>',
                unsafe_allow_html=True,
            )

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            '<div class="card"><div class="card-title">Why these metrics?</div><div class="card-text">The target is imbalanced, so accuracy alone can be misleading. ROC-AUC evaluates ranking ability across thresholds, while PR-AUC is especially useful for the relatively rare positive class.</div></div>',
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            '<div class="card"><div class="card-title">Recall vs Precision</div><div class="card-text">Recall measures how many actual positive-risk cases are detected. Precision measures how many predicted positive cases are actually positive. Both should be considered together.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Model Details</div>', unsafe_allow_html=True)
    # FIX for Streamlit Arrow warning: every Value entry is a string.
    details = pd.DataFrame({
        "Property": ["Algorithm", "Dataset", "Input Features", "Target", "Decision Threshold", "Positive Class"],
        "Value": [MODEL_NAME, DATASET_NAME, str(len(FEATURES)), TARGET_NAME, f"{BEST_THRESHOLD:.2f}", "Serious delinquency within prediction horizon"],
    }).astype(str)
    st.dataframe(details, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Relative feature importance extracted from the trained Random Forest classifier.</div>', unsafe_allow_html=True)
    render_importance_bars(IMPORTANCE_DF, 10)
    importance_table = IMPORTANCE_DF.copy()
    importance_table["Feature"] = importance_table["Feature"].map(feature_label).astype(str)
    importance_table["Importance"] = importance_table["Importance"].astype(float).round(4)
    st.dataframe(importance_table, use_container_width=True, hide_index=True)


# ============================================================
# ABOUT
# ============================================================
with about:
    st.markdown('<div class="page-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-description">Project overview, methodology, technology stack, and limitations.</div>', unsafe_allow_html=True)

    sections = [
        ("🎯 Problem Statement", "Financial institutions need ways to identify customers who may be at higher risk of serious delinquency. A machine-learning approach can learn relationships among credit behaviour, debt ratios, income, and payment history."),
        ("🤖 Solution", f"This project uses a {MODEL_NAME} to estimate delinquency probability from customer credit and financial attributes. The saved preprocessing and classification pipeline is reused directly by the application."),
        ("🗂 Dataset", f"<b>Dataset:</b> {DATASET_NAME}<br><b>Target:</b> {TARGET_NAME}<br><b>Features used:</b> {len(FEATURES)}<br><b>Task:</b> Binary classification"),
    ]
    for title, body in sections:
        st.markdown(
            f'<div class="card"><div class="card-title">{title}</div><div class="card-text">{body}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Technology Stack</div>', unsafe_allow_html=True)
    tech = pd.DataFrame({
        "Layer": ["Programming", "Machine Learning", "Preprocessing", "Model Persistence", "Frontend"],
        "Technology": ["Python", "Scikit-learn / Random Forest", "SimpleImputer", "Joblib", "Streamlit"],
    }).astype(str)
    st.dataframe(tech, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Application Capabilities</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="card"><div class="card-text">✓ Individual customer prediction<br>✓ Probability-based risk estimation<br>✓ Low / Medium / High presentation<br>✓ Batch CSV prediction<br>✓ CSV results download</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-text">✓ Model performance overview<br>✓ Feature importance analysis<br>✓ Input validation<br>✓ Reusable trained pipeline<br>✓ Production-style Streamlit interface</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Future Improvements</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-text">• SHAP-based individual explanations<br>• Probability calibration<br>• Model monitoring and drift detection<br>• FastAPI inference service<br>• Docker + CI/CD<br>• Cloud deployment<br>• Role-based access and audit logging</div></div>', unsafe_allow_html=True)
    st.warning("Educational and portfolio project only. Predictions should not be used as the sole basis for real-world lending or financial decisions.")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">AI Credit Risk Predictor · Machine Learning Portfolio Project<br>Built with Python · Scikit-learn · Streamlit</div>',
    unsafe_allow_html=True,
)
