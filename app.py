"""
app.py
──────
Streamlit web application entry point.
All ML logic lives in src/model.py — this file handles UI only.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

from model import StudentModel, load_dataset_from_bytes
from utils import (
    build_sample_dataframe,
    GRADE_CONFIG,
    FEATURE_META,
    feature_icon,
    feature_hint,
    get_grade_config,
)

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size:2.2rem; font-weight:700;
    background:linear-gradient(90deg,#4F8EF7,#A259FF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.subtitle { color:#888; font-size:1rem; margin-top:0; }
.metric-card {
    background:#f8f9ff; border:1px solid #e0e4ff;
    border-radius:12px; padding:1.2rem 1.5rem; text-align:center;
}
.metric-value  { font-size:2rem; font-weight:700; color:#4F8EF7; }
.metric-label  { font-size:0.8rem; color:#888; text-transform:uppercase; letter-spacing:0.05em; }
.result-box    { border-radius:14px; padding:1.5rem 2rem; margin:1rem 0; border:1.5px solid #e0e4ff; }
div[data-testid="stSidebarContent"] { background:#f4f6ff; }
</style>
""", unsafe_allow_html=True)


# ── session state ─────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = StudentModel()


def model() -> StudentModel:
    return st.session_state.model


# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.markdown("---")
    st.markdown("### 📂 Load Dataset")

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Any file with numeric columns works.",
    )

    if st.button("▶ Use Sample Data", use_container_width=True):
        df_sample = build_sample_dataframe()
        target    = "final_score"
        features  = ["study_hours", "previous_score", "attendance",
                     "sleep_hours", "assignments_done"]
        model().train(df_sample, target, features)
        st.success("✓ Sample data loaded & model trained!")

    if uploaded:
        try:
            df_raw = load_dataset_from_bytes(uploaded, uploaded.name)
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns.")
            else:
                st.success(f"✓ {len(df_raw)} rows × {len(df_raw.columns)} columns")

                from model import auto_detect_target, auto_detect_features
                def_target = auto_detect_target(numeric_cols)
                target_col = st.selectbox(
                    "🎯 Target column (score to predict)",
                    numeric_cols,
                    index=numeric_cols.index(def_target) if def_target else 0,
                )
                feat_opts  = [c for c in numeric_cols if c != target_col]
                def_feats  = auto_detect_features(feat_opts, target_col)
                feat_cols  = st.multiselect(
                    "📥 Feature columns (inputs)",
                    feat_opts, default=def_feats,
                )

                if st.button("🚀 Train Model", use_container_width=True) and feat_cols:
                    model().train(df_raw, target_col, feat_cols)
                    st.success("✓ Model trained!")

        except Exception as e:
            st.error(f"Error: {e}")

    # model stats in sidebar
    if model().is_trained:
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        m = model().metrics_dict()
        st.metric("R² Score",  f"{m['r2']:.4f}")
        st.metric("MAE",       f"{m['mae']:.2f} marks")
        st.metric("RMSE",      f"{m['rmse']:.2f} marks")
        st.metric("Rows used", m["n_rows"])


# ── main area ─────────────────────────────────────────────────
st.markdown('<p class="main-title">🎓 Student Performance Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a dataset → auto-train → predict scores, grades & pass/fail</p>', unsafe_allow_html=True)
st.markdown("---")

if not model().is_trained:
    st.info("👈  Load a dataset or click **Use Sample Data** in the sidebar to get started.")
    with st.expander("📋 What columns should my CSV have?"):
        st.markdown("""
| Column | Description |
|---|---|
| `study_hours` | Daily study hours |
| `previous_score` | Last exam score (0–100) |
| `attendance` | Attendance % |
| `sleep_hours` | Nightly sleep hours |
| `assignments_done` | Assignments completed |
| `final_score` | **Target** — the score to predict |
        """)
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Quick Predict",
    "📋 Full Predict",
    "📦 Batch Predict",
    "📈 Model Insights",
])


# ── Tab 1: Quick Predict ──────────────────────────────────────
with tab1:
    st.subheader("⚡ Quick Predict by Study Hours")
    st.caption("Adjust study hours — other features use dataset averages.")

    col_sl, col_res = st.columns([1, 1], gap="large")
    with col_sl:
        study_hours = st.slider("📚 Study Hours per Day", 0.0, 12.0, 5.0, 0.5)
        st.markdown(f"**Selected: {study_hours} hrs / day**")

    means = model().feature_means()
    row   = {f: means.get(f, 0) for f in model().features}
    row["study_hours"] = study_hours
    score = model().predict(row)

    from model import score_to_grade, score_to_passfail
    grade = score_to_grade(score)
    cfg   = get_grade_config(grade)

    with col_res:
        st.markdown(f"""
<div class="result-box" style="border-color:{cfg['color']}44;background:{cfg['color']}11;">
  <div style="font-size:.9rem;color:#888;margin-bottom:.5rem;">Predicted Result</div>
  <div style="font-size:3rem;font-weight:800;color:{cfg['color']}">{score}
    <span style="font-size:1.2rem;color:#aaa">/100</span></div>
  <div style="margin:.8rem 0;">
    <span style="background:{cfg['color']}22;color:{cfg['color']};
                 padding:.4rem 1.2rem;border-radius:999px;font-size:1.4rem;font-weight:700">
      {cfg['emoji']} Grade {grade}
    </span>
  </div>
  <div style="font-size:1.1rem;font-weight:600;color:{'#22c55e' if score>=40 else '#ef4444'}">
    {'✅ PASS' if score >= 40 else '❌ FAIL'}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Grade scale: A+ ≥90 · A ≥80 · B ≥70 · C ≥60 · D ≥40 · F <40 · Pass threshold = 40")


# ── Tab 2: Full Predict ───────────────────────────────────────
with tab2:
    st.subheader("📋 Full Prediction — Enter All Features")
    st.caption("Fill in every feature for a precise individual prediction.")

    features  = model().features
    means     = model().feature_means()
    df_train  = model().df

    cols_ui = st.columns(min(len(features), 3))
    inputs  = {}
    for i, feat in enumerate(features):
        lo = float(df_train[feat].min()) if df_train is not None else 0.0
        hi = float(df_train[feat].max()) if df_train is not None else 100.0
        mn = float(means.get(feat, (lo + hi) / 2))
        with cols_ui[i % len(cols_ui)]:
            inputs[feat] = st.number_input(
                f"{feature_icon(feat)} {feat.replace('_', ' ').title()}",
                min_value=0.0, max_value=hi * 1.5,
                value=round(mn, 1), step=0.5,
                key=f"full_{feat}",
                help=feature_hint(feat),
            )

    st.markdown("")
    if st.button("🔮 Predict Score", type="primary", use_container_width=True):
        score = model().predict(inputs)
        grade = score_to_grade(score)
        cfg   = get_grade_config(grade)

        c1, c2, c3 = st.columns(3)
        pf_color = "#22c55e" if score >= 40 else "#ef4444"
        pf_text  = "PASS ✅" if score >= 40 else "FAIL ❌"

        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{cfg['color']}">{score}</div>
                <div class="metric-label">Predicted Score / 100</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{cfg['color']}">{cfg['emoji']} {grade}</div>
                <div class="metric-label">Grade</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{pf_color};font-size:1.5rem">{pf_text}</div>
                <div class="metric-label">Result</div></div>""",
                unsafe_allow_html=True)


# ── Tab 3: Batch Predict ──────────────────────────────────────
with tab3:
    st.subheader("📦 Batch Prediction — Upload a File")
    st.caption("Predict scores for every student in a CSV or Excel file.")

    batch_file = st.file_uploader(
        "Upload student data", type=["csv", "xlsx"], key="batch_upload"
    )
    if batch_file:
        try:
            bdf = load_dataset_from_bytes(batch_file, batch_file.name)
            missing = [f for f in model().features if f not in bdf.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                st.info(f"Required: {model().features}")
            else:
                result_df = model().predict_dataframe(bdf)
                st.success(f"✓ Predicted {len(result_df)} students!")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Score",  f"{result_df['predicted_score'].mean():.1f}")
                c2.metric("Highest",    f"{result_df['predicted_score'].max():.1f}")
                c3.metric("Lowest",     f"{result_df['predicted_score'].min():.1f}")
                c4.metric("Pass Rate",  f"{(result_df['pass_fail']=='PASS').mean()*100:.1f}%")

                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    result_df.to_csv(index=False).encode(),
                    "predictions.csv", "text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error: {e}")


# ── Tab 4: Model Insights ─────────────────────────────────────
with tab4:
    st.subheader("📈 Model Insights")

    m = model().metrics_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score",  f"{m['r2']:.4f}",   help="1.0 = perfect")
    c2.metric("MAE",       f"{m['mae']:.2f} marks")
    c3.metric("RMSE",      f"{m['rmse']:.2f} marks")
    c4.metric("Rows used", m["n_rows"])

    st.markdown("---")
    st.markdown("#### Feature Coefficients")
    st.caption("Higher absolute value = stronger influence on the score.")
    st.dataframe(model().coef_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Dataset Preview (first 20 rows)")
    st.dataframe(model().df.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Model Summary")
    st.code(model().summary(), language=None)
