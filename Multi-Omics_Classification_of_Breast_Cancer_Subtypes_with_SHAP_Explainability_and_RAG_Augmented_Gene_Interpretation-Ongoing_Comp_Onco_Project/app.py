import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------- Load artifacts ---------
@st.cache_resource
def load_artifacts():
    MODELS = Path("models")
    DATA_PROC = Path("data/processed")

    try:
        model = joblib.load(MODELS / "xgb_brca.pkl")
        label_encoder = joblib.load(MODELS / "label_encoder.pkl")
        X = pd.read_parquet(DATA_PROC / "X_brca.parquet")
    except Exception as e:
        st.error(f"❌ Failed to load model or data: {e}")
        raise

    return model, label_encoder, X


xgb_clf, le, X_all = load_artifacts()
feature_order = X_all.columns
feature_set = set(feature_order)

# --------- Helper for prediction ---------
def predict_subtype_from_row(row: pd.Series):
    """
    row: pandas Series with gene expression for one sample (must match feature_order)
    """
    # Make sure all genes are present in the right order
    row = row.reindex(feature_order)

    # Convert to numeric
    row = pd.to_numeric(row, errors="coerce")

    if row.isna().all():
        raise ValueError("All values became NaN after numeric conversion.")

    # Fill remaining NaNs with 0 (or you could use median if you prefer)
    row = row.fillna(0.0)

    probs = xgb_clf.predict_proba(row.values.reshape(1, -1))[0]
    pred_label = le.inverse_transform([np.argmax(probs)])[0]
    return pred_label, probs


# --------- Streamlit UI ---------
st.title("🧬 PAM50 Subtype Predictor – TCGA-BRCA (XGBoost)")

st.markdown(
    """
This app uses an XGBoost model trained on TCGA-BRCA RNA-seq data  
to predict **PAM50 molecular subtypes** (Luminal A/B, Basal-like, HER2-enriched, Normal-like).
"""
)

mode = st.sidebar.radio(
    "Choose input mode:",
    ("Use existing TCGA sample", "Upload your own expression row (CSV/TSV)"),
)

# =========================
# Mode 1: Existing TCGA sample
# =========================
if mode == "Use existing TCGA sample":
    st.subheader("Select a sample from TCGA-BRCA")

    sample_id = st.selectbox("Sample ID", X_all.index.sort_values())

    if st.button("Predict subtype"):
        try:
            row = X_all.loc[sample_id]
        except KeyError:
            st.error(f"❌ Sample ID `{sample_id}` not found in training data.")
            st.stop()

        try:
            pred, probs = predict_subtype_from_row(row)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        st.success(f"✅ Predicted PAM50 subtype for **{sample_id}**: **{pred}**")

        # Show probabilities as a bar chart
        prob_df = pd.DataFrame(
            probs,
            index=le.classes_,
            columns=["Probability"],
        )
        st.bar_chart(prob_df)

        st.markdown("### Raw probabilities")
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))

# =========================
# Mode 2: Upload your own expression row
# =========================
elif mode == "Upload your own expression row (CSV/TSV)":
    st.subheader("Upload a single-sample expression file")

    st.markdown(
        """
- File should contain **one row** of expression values  
- Columns must be **gene names matching the training set**  
- Index/first column can be a sample ID (optional)
"""
    )

    uploaded_file = st.file_uploader("Upload CSV or TSV", type=["csv", "tsv"])

    if uploaded_file is not None:
        # Try to read the file safely
        try:
            sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
            df = pd.read_csv(uploaded_file, sep=sep, index_col=0)
        except Exception as e:
            st.error(f"❌ Could not read the file: {e}")
            st.stop()

        # Basic shape check
        if df.shape[0] != 1:
            st.error(
                f"❌ Expected exactly 1 sample (1 row), but found {df.shape[0]} rows."
            )
            st.stop()

        st.write("Uploaded sample shape:", df.shape)
        st.markdown("Preview of first 10 genes:")
        st.dataframe(df.iloc[:, :10])

        # Gene overlap check
        uploaded_genes = set(df.columns)
        common_genes = uploaded_genes & feature_set
        overlap_fraction = len(common_genes) / len(feature_set)

        if len(common_genes) == 0:
            st.error(
                "❌ No overlapping genes found between uploaded file and model features.\n"
                "Please ensure the column names are gene symbols matching the training data."
            )
            st.stop()
        elif overlap_fraction < 0.6:
            st.warning(
                f"⚠ Only {len(common_genes)} / {len(feature_set)} genes "
                f"({overlap_fraction*100:.1f}%) overlap with the training set.\n\n"
                "Predictions may be unreliable."
            )

        if st.button("Predict subtype"):
            # Take the single row
            row = df.iloc[0]

            # Create a full row with all required genes: use uploaded values where present, 0 otherwise
            aligned = pd.Series(
                {g: row.get(g, np.nan) for g in feature_order},
                index=feature_order,
            )

            try:
                pred, probs = predict_subtype_from_row(aligned)
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.stop()

            st.success(f"✅ Predicted PAM50 subtype: **{pred}**")

            prob_df = pd.DataFrame(
                probs,
                index=le.classes_,
                columns=["Probability"],
            )
            st.bar_chart(prob_df)
            st.markdown("### Raw probabilities")
            st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}))
