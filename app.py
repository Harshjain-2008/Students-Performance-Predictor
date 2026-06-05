"""
app.py
──────
Streamlit web app — 100% pure Python, zero CSS, zero HTML.
All ML logic lives in src/model.py

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.model import (
    StudentModel,
    load_dataset_from_bytes,
    auto_detect_target,
    auto_detect_features,
    score_to_grade,
    score_to_passfail,
)
from src.utils import (
    build_sample_dataframe,
    feature_icon,
    feature_hint,
)

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── grade helpers (pure Python) ───────────────────────────────
GRADE_EMOJI = {"A+": "🏆", "A": "⭐", "B": "👍", "C": "📘", "D": "⚠️", "F": "❌"}
GRADE_DESC  = {"A+": "Excellent", "A": "Great", "B": "Good",
               "C": "Average",   "D": "Below Average", "F": "Fail"}

def grade_line(score):
    grade = score_to_grade(score)
    emoji = GRADE_EMOJI[grade]
    desc  = GRADE_DESC[grade]
    pf    = "✅ PASS" if score >= 40 else "❌ FAIL"
    return grade, emoji, desc, pf

# ── session state ─────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = StudentModel()

def mdl() -> StudentModel:
    return st.session_state.model

# ══════════════════════════════════════════════════════════════
# SIDEBAR  (pure Python Streamlit only)
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎓 Student Predictor")
    st.caption("Train a model on your data, then predict student scores.")
    st.divider()

    # ── section 1: load data ──────────────────────────────────
    st.subheader("📂 Load Dataset")

    uploaded = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Must have numeric columns. Any column names work.",
    )

    st.caption("— or —")

    if st.button("▶️  Use Built-in Sample Data", use_container_width=True):
        df_s     = build_sample_dataframe()
        target   = "final_score"
        features = ["study_hours", "previous_score", "attendance",
                    "sleep_hours", "assignments_done"]
        mdl().train(df_s, target, features)
        st.session_state["trained_msg"] = "✅ Sample data loaded and model trained!"

    if "trained_msg" in st.session_state:
        st.success(st.session_state["trained_msg"])

    # ── handle uploaded file ──────────────────────────────────
    if uploaded:
        try:
            df_raw       = load_dataset_from_bytes(uploaded, uploaded.name)
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2:
                st.error("⚠️ Need at least 2 numeric columns in your file.")
            else:
                st.success(f"✅ File loaded: {len(df_raw)} rows × {len(df_raw.columns)} columns")
                st.divider()

                st.subheader("⚙️ Configure Columns")

                def_target = auto_detect_target(numeric_cols)
                target_col = st.selectbox(
                    "🎯 Which column is the score to PREDICT?",
                    numeric_cols,
                    index=numeric_cols.index(def_target) if def_target else 0,
                    help="This is the output column — what the model learns to predict.",
                )

                feat_opts = [c for c in numeric_cols if c != target_col]
                def_feats = auto_detect_features(feat_opts, target_col)
                feat_cols = st.multiselect(
                    "📥 Which columns are INPUTS (features)?",
                    feat_opts,
                    default=def_feats,
                    help="These are the student attributes used to make the prediction.",
                )

                if feat_cols:
                    if st.button("🚀 Train Model on This Data", use_container_width=True):
                        with st.spinner("Training model..."):
                            mdl().train(df_raw, target_col, feat_cols)
                        st.success("✅ Model trained successfully!")
                else:
                    st.warning("Please select at least one input feature.")

        except Exception as e:
            st.error(f"❌ Could not load file: {e}")

    # ── section 2: model stats ────────────────────────────────
    if mdl().is_trained:
        st.divider()
        st.subheader("📊 Model Performance")
        m = mdl().metrics_dict()

        col_a, col_b = st.columns(2)
        col_a.metric("R² Score",   f"{m['r2']:.3f}",  help="1.0 = perfect fit")
        col_b.metric("MAE",        f"{m['mae']:.2f}",  help="Average error in marks")
        col_a.metric("RMSE",       f"{m['rmse']:.2f}", help="Root mean squared error")
        col_b.metric("Rows used",  str(m["n_rows"]))

        st.divider()
        st.caption(f"🎯 Target: **{m['target']}**")
        st.caption(f"📥 Features: {m['features']} columns")
        st.caption("Pass threshold: **40 marks**")
        st.caption("Grade: A+ ≥90 · A ≥80 · B ≥70 · C ≥60 · D ≥40 · F <40")


# ══════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════
st.title("🎓 Student Performance Predictor")
st.caption("Upload your student dataset → model trains automatically → predict scores, grades & pass/fail")
st.divider()

# ── not trained yet ───────────────────────────────────────────
if not mdl().is_trained:
    st.info("👈  Use the **sidebar** to load a dataset and train the model first.", icon="ℹ️")

    with st.expander("📋 What should my CSV file look like?", expanded=True):
        st.write("Your file needs numeric columns. Recommended column names:")
        st.table(pd.DataFrame({
            "Column Name":  ["study_hours", "previous_score", "attendance",
                             "sleep_hours", "assignments_done", "final_score"],
            "Description":  ["Daily study hours", "Last exam score (0–100)",
                             "Attendance percentage", "Nightly sleep hours",
                             "Assignments completed", "✅ TARGET — score to predict"],
        }))
        st.caption("Any column names work — you choose target & features after uploading.")

    with st.expander("❓ How does the app work?"):
        st.write("""
1. **Upload** your CSV or Excel file in the sidebar (or click *Use Sample Data*)
2. **Select** which column is the target score and which are input features
3. **Train** — the model learns patterns from your data
4. **Predict** — enter a student's details and get their predicted score, grade, and pass/fail
        """)
    st.stop()

# ── tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡  Quick Predict",
    "📋  Full Predict",
    "📦  Batch Predict",
    "📈  Model Insights",
])


# ══════════════════════════════
# TAB 1 — QUICK PREDICT
# ══════════════════════════════
with tab1:
    st.subheader("⚡ Quick Predict by Study Hours")
    st.write("Move the slider to your daily study hours — the app instantly predicts your score.")
    st.write("*(All other features are filled with class averages from the training data.)*")
    st.divider()

    study_hours = st.slider(
        "📚 How many hours do you study per day?",
        min_value=0.0, max_value=12.0, value=5.0, step=0.5,
        help="Drag to change study hours",
    )

    # predict using averages for all other features
    means = mdl().feature_means()
    row   = {f: means.get(f, 0) for f in mdl().features}
    row["study_hours"] = study_hours
    score = mdl().predict(row)
    grade, emoji, desc, pf = grade_line(score)

    st.divider()
    st.subheader("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="🎯 Predicted Score",
        value=f"{score} / 100",
    )
    col2.metric(
        label="🏅 Grade",
        value=f"{emoji} {grade}",
        help=desc,
    )
    col3.metric(
        label="📋 Result",
        value=pf,
    )

    st.divider()

    # simple feedback message
    if score >= 90:
        st.success(f"🏆 Outstanding! {study_hours} hours/day puts you at the top of the class.")
    elif score >= 70:
        st.success(f"👍 Good work! Keep studying {study_hours} hours/day to stay on track.")
    elif score >= 40:
        st.warning(f"⚠️ You will pass, but try increasing your study hours above {study_hours}.")
    else:
        st.error(f"❌ At {study_hours} hours/day, you are at risk of failing. Aim for at least 4–5 hours.")


# ══════════════════════════════
# TAB 2 — FULL PREDICT
# ══════════════════════════════
with tab2:
    st.subheader("📋 Full Prediction — Enter All Features")
    st.write("Enter the exact details for a student to get the most accurate prediction.")
    st.divider()

    features = mdl().features
    means    = mdl().feature_means()
    df_train = mdl().df

    # layout: up to 3 inputs per row
    inputs = {}
    cols_per_row = 3
    rows = [features[i:i+cols_per_row] for i in range(0, len(features), cols_per_row)]

    for row_feats in rows:
        cols = st.columns(len(row_feats))
        for col, feat in zip(cols, row_feats):
            lo = float(df_train[feat].min()) if df_train is not None else 0.0
            hi = float(df_train[feat].max()) if df_train is not None else 100.0
            mn = float(means.get(feat, (lo + hi) / 2))
            with col:
                inputs[feat] = st.number_input(
                    label=f"{feature_icon(feat)}  {feat.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=round(hi * 1.5, 1),
                    value=round(mn, 1),
                    step=0.5,
                    help=feature_hint(feat),
                    key=f"full_{feat}",
                )

    st.divider()
    predict_clicked = st.button(
        "🔮  Predict This Student's Score",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        score = mdl().predict(inputs)
        grade, emoji, desc, pf = grade_line(score)

        st.divider()
        st.subheader("📊 Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Score",  f"{score} / 100")
        c2.metric("🏅 Grade",  f"{emoji} {grade} — {desc}")
        c3.metric("📋 Result", pf)

        st.divider()

        # breakdown table
        st.write("**Input Summary:**")
        summary_df = pd.DataFrame({
            "Feature": list(inputs.keys()),
            "Value":   [round(v, 2) for v in inputs.values()],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        if score >= 40:
            st.success(f"This student is predicted to **PASS** with {score}/100 (Grade {grade}).")
        else:
            st.error(f"This student is predicted to **FAIL** with {score}/100. Grade: {grade}.")


# ══════════════════════════════
# TAB 3 — BATCH PREDICT
# ══════════════════════════════
with tab3:
    st.subheader("📦 Batch Prediction — Predict for Many Students at Once")
    st.write("Upload a CSV or Excel file containing multiple students. The app predicts scores for all of them and lets you download the results.")
    st.divider()

    st.info(f"ℹ️  Your file must contain these columns: **{mdl().features}**", icon="ℹ️")

    batch_file = st.file_uploader(
        "Upload student data file",
        type=["csv", "xlsx"],
        key="batch_upload",
    )

    if batch_file:
        try:
            bdf     = load_dataset_from_bytes(batch_file, batch_file.name)
            missing = [f for f in mdl().features if f not in bdf.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.info(f"Your file has: {list(bdf.columns)}")
            else:
                with st.spinner(f"Predicting scores for {len(bdf)} students..."):
                    result_df = mdl().predict_dataframe(bdf)

                st.success(f"✅ Done! Predicted scores for {len(result_df)} students.")
                st.divider()

                # summary stats
                st.subheader("📊 Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Average Score", f"{result_df['predicted_score'].mean():.1f}")
                s2.metric("Highest Score", f"{result_df['predicted_score'].max():.1f}")
                s3.metric("Lowest Score",  f"{result_df['predicted_score'].min():.1f}")
                s4.metric("Pass Rate",     f"{(result_df['pass_fail']=='PASS').mean()*100:.1f}%")

                st.divider()

                # grade distribution
                st.subheader("🏅 Grade Distribution")
                grade_counts = result_df["predicted_grade"].value_counts().reset_index()
                grade_counts.columns = ["Grade", "Count"]
                st.dataframe(grade_counts, use_container_width=True, hide_index=True)

                st.divider()

                st.subheader("📋 Full Results Table")
                st.dataframe(result_df, use_container_width=True)

                st.divider()
                st.download_button(
                    label="⬇️  Download Results as CSV",
                    data=result_df.to_csv(index=False).encode(),
                    file_name="student_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


# ══════════════════════════════
# TAB 4 — MODEL INSIGHTS
# ══════════════════════════════
with tab4:
    st.subheader("📈 Model Insights")
    st.write("Understand how well the model performs and which features matter most.")
    st.divider()

    m = mdl().metrics_dict()

    # accuracy metrics
    st.subheader("🎯 Accuracy Metrics")
    st.caption("These numbers tell you how accurate the model is on unseen test data.")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("R² Score",   f"{m['r2']:.4f}",
              help="How much of the score variation the model explains. 1.0 = perfect.")
    a2.metric("MAE",        f"{m['mae']:.2f} marks",
              help="Average absolute error in predicted marks.")
    a3.metric("RMSE",       f"{m['rmse']:.2f} marks",
              help="Root mean squared error — penalises big mistakes more.")
    a4.metric("Training rows", str(m["n_rows"]))

    # R² interpretation
    r2 = m["r2"]
    if r2 >= 0.95:
        st.success(f"✅ R² = {r2:.4f} — Excellent fit. The model explains {r2*100:.1f}% of score variance.")
    elif r2 >= 0.80:
        st.info(f"ℹ️ R² = {r2:.4f} — Good fit. The model explains {r2*100:.1f}% of score variance.")
    else:
        st.warning(f"⚠️ R² = {r2:.4f} — Moderate fit. Try adding more features to improve accuracy.")

    st.divider()

    # feature importance
    st.subheader("🔑 Feature Importance")
    st.caption("Which inputs have the strongest influence on the predicted score?")
    st.write("A higher coefficient (positive or negative) means that feature has more impact.")
    st.dataframe(
        mdl().coef_df.rename(columns={
            "Feature":     "📥 Feature",
            "Coefficient": "📐 Coefficient",
            "Direction":   "📊 Impact Direction",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # dataset preview
    st.subheader("🗂️ Training Dataset Preview")
    st.caption(f"Showing first 20 rows of {m['n_rows']} total rows used for training.")
    st.dataframe(mdl().df.head(20), use_container_width=True)

    st.divider()

    # model details
    st.subheader("ℹ️ Model Details")
    detail_df = pd.DataFrame({
        "Property": ["Algorithm", "Target column", "Number of features",
                     "Feature columns", "Pass threshold", "Training rows"],
        "Value":    ["Linear Regression (scikit-learn)", m["target"],
                     str(m["features"]), ", ".join(mdl().features),
                     "40 marks", str(m["n_rows"])],
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
