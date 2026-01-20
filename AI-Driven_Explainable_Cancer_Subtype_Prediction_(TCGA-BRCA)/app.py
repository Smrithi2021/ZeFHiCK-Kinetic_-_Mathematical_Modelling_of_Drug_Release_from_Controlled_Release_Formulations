from rag_explainer import explain_subtype
from agents.agent_router import route_query

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_artifacts():
    MODELS = Path("models")
    DATA_PROC = Path("data/processed")

    model = joblib.load(MODELS / "xgb_brca.pkl")
    label_encoder = joblib.load(MODELS / "label_encoder.pkl")
    X = pd.read_parquet(DATA_PROC / "X_brca.parquet")

    return model, label_encoder, X


xgb_clf, le, X_all = load_artifacts()
feature_order = X_all.columns
feature_set = set(feature_order)


# =========================
# Prediction helper
# =========================
def predict_subtype_from_row(row: pd.Series):
    row = row.reindex(feature_order)
    row = pd.to_numeric(row, errors="coerce")

    if row.isna().all():
        raise ValueError("All values became NaN after numeric conversion.")

    row = row.fillna(0.0)

    probs = xgb_clf.predict_proba(row.values.reshape(1, -1))[0]
    pred_label = le.inverse_transform([np.argmax(probs)])[0]
    return pred_label, probs


# =========================
# Streamlit UI
# =========================
st.title("🧬 PAM50 Subtype Predictor – TCGA-BRCA (XGBoost)")

st.markdown(
    """
This app uses an **XGBoost model trained on TCGA-BRCA RNA-seq data**  
to predict **PAM50 molecular subtypes** and provide **AI-assisted biological explanations**.
"""
)

mode = st.sidebar.radio(
    "Choose input mode:",
    ("Use existing TCGA sample", "Upload your own expression row (CSV/TSV)"),
)


# =========================
# MODE 1 — TCGA sample
# =========================
if mode == "Use existing TCGA sample":
    st.subheader("Select a sample from TCGA-BRCA")

    sample_id = st.selectbox("Sample ID", X_all.index.sort_values())

    if st.button("🔬 Predict PAM50 subtype"):
        try:
            row = X_all.loc[sample_id]
            pred, probs = predict_subtype_from_row(row)

            st.session_state["predicted_subtype"] = pred
            st.session_state["prediction_probs"] = probs

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        st.success(f"✅ Predicted PAM50 subtype for **{sample_id}**: **{pred}**")

        prob_df = pd.DataFrame(
            probs, index=le.classes_, columns=["Probability"]
        )
        st.bar_chart(prob_df)
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))

        with st.spinner("Generating biological explanation (AI-assisted)..."):
            explanation = explain_subtype(pred)
            st.markdown("### 🧠 AI-generated biological explanation")
            st.write(explanation)


# =========================
# MODE 2 — Upload CSV / TSV
# =========================
elif mode == "Upload your own expression row (CSV/TSV)":
    st.subheader("Upload a single-sample expression file")

    uploaded_file = st.file_uploader("Upload CSV or TSV", type=["csv", "tsv"])

    if uploaded_file:
        sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
        df = pd.read_csv(uploaded_file, sep=sep, index_col=0)

        if df.shape[0] != 1:
            st.error("❌ Please upload exactly **one row** (one sample).")
            st.stop()

        uploaded_genes = set(df.columns)
        common_genes = uploaded_genes & feature_set

        if not common_genes:
            st.error("❌ No overlapping genes with model features.")
            st.stop()

        aligned = pd.Series(
            {g: df.iloc[0].get(g, np.nan) for g in feature_order},
            index=feature_order,
        )

        if st.button("🔬 Predict PAM50 subtype"):
            pred, probs = predict_subtype_from_row(aligned)

            st.session_state["predicted_subtype"] = pred
            st.session_state["prediction_probs"] = probs

            st.success(f"✅ Predicted PAM50 subtype: **{pred}**")

            prob_df = pd.DataFrame(
                probs, index=le.classes_, columns=["Probability"]
            )
            st.bar_chart(prob_df)
            st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))


# =========================
# AGENTIC AI EXPLANATION
# =========================
st.markdown("---")
st.markdown("## 🤖 Ask the AI about this prediction")

user_question = st.text_input(
    "Ask about subtype biology, prediction reasoning, or PAM50 classification",
    placeholder="Why was this classified as HER2-enriched?"
)

if st.button("🧠 Explain with AI"):
    predicted_subtype = st.session_state.get("predicted_subtype")

    if not predicted_subtype:
        st.warning("⚠ Please run a prediction first.")
    elif not user_question.strip():
        st.warning("⚠ Please enter a question.")
    else:
        with st.spinner("🔍 Retrieving knowledge and reasoning..."):
            answer = route_query(
                user_query=user_question,
                predicted_subtype=predicted_subtype
            )

        st.markdown("### 📘 AI-generated explanation")
        st.write(answer)
