import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="🔋 LSTM Energy Forecast", layout="wide")

# ------------------- LOAD MODEL & SCALER -------------------
@st.cache_resource
def load_lstm_model():
    # compile=False avoids: "Could not deserialize 'keras.metrics.mse'..."
    return load_model("lstm_energy_model.h5", compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load("lstm_scaler.pkl")

model = load_lstm_model()
scaler = load_scaler()

# ------------------- HELPERS -------------------
WINDOW_SIZE = 30

def detect_value_column(df: pd.DataFrame) -> str:
    """Pick the first numeric column that is not 'Datetime'."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric value column found in the CSV.")
    return numeric_cols[0]

def make_daily_series(df: pd.DataFrame, date_col: str = "Datetime", value_col: str = None) -> pd.Series:
    """Resample to daily mean and return a clean pd.Series indexed by date."""
    if date_col not in df.columns:
        raise ValueError("CSV must contain a 'Datetime' column.")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    if value_col is None:
        value_col = detect_value_column(df)
    s = df.set_index(date_col)[value_col].resample("D").mean().dropna()
    if len(s) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} daily points to forecast.")
    return s

def forecast_next_30_from_window(last_window_values: np.ndarray) -> (pd.DatetimeIndex, np.ndarray):
    """
    last_window_values: shape (30,) or (30, 1) raw MW values (NOT scaled).
    Returns: (future_dates, predictions_raw) where predictions_raw shape is (30, 1)
    """
    # Ensure 2D for scaler
    last_window_values = np.array(last_window_values).reshape(-1, 1)
    if last_window_values.shape[0] != WINDOW_SIZE:
        raise ValueError(f"Last window must have exactly {WINDOW_SIZE} values.")

    # Scale -> predict iteratively -> inverse-scale
    scaled_input = scaler.transform(last_window_values)
    preds_scaled = []
    fc_in = scaled_input.copy()  # (30,1)
    for _ in range(30):
        x = fc_in.reshape(1, WINDOW_SIZE, 1)
        y_hat_scaled = model.predict(x, verbose=0)        # (1,1)
        preds_scaled.append(y_hat_scaled[0])              # append (1,)
        fc_in = np.append(fc_in[1:], y_hat_scaled, axis=0)

    preds_scaled = np.vstack(preds_scaled)                # (30,1)
    preds_raw = scaler.inverse_transform(preds_scaled)    # back to MW
    return preds_raw

def plot_csv_forecast(history_series: pd.Series, forecast_dates: pd.DatetimeIndex, preds: np.ndarray, tail=100):
    plt.figure(figsize=(8, 4))
    plt.plot(history_series.index[-tail:], history_series.values[-tail:], label="Last 100 Days Actual")
    plt.plot(forecast_dates, preds.flatten(), label="30-Day LSTM Forecast", color="orange")
    plt.title("LSTM Energy Forecast")
    plt.xlabel("Date")
    plt.ylabel("Energy Consumption (MW)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def plot_manual_forecast(last_30: np.ndarray, start_date: datetime, preds: np.ndarray):
    """Show the provided last 30 days window + next 30-day forecast."""
    base_dates = pd.date_range(end=start_date, periods=WINDOW_SIZE, freq="D")
    future_dates = pd.date_range(start=start_date + timedelta(days=1), periods=30, freq="D")

    plt.figure(figsize=(8, 4))
    plt.plot(base_dates, np.array(last_30).flatten(), label="Provided Last 30 Days (Baseline)")
    plt.plot(future_dates, preds.flatten(), label="30-Day LSTM Forecast", color="orange")
    plt.title("Manual 30-Day Forecast (Baseline + Forecast)")
    plt.xlabel("Date")
    plt.ylabel("Energy Consumption (MW)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# ------------------- HEADER -------------------
st.title("🔋 LSTM Energy Consumption Forecast App")
st.markdown(
    "Forecast **PJM Energy Load (MW)** for the **next 30 days** using an **LSTM** model. "
    "Use either a **CSV upload** (recommended) or **individual input**."
)
st.markdown("---")

# ------------------- SIDEBAR MODE -------------------
st.sidebar.header("📥 Choose Mode")
mode = st.sidebar.radio(
    "Select:",
    ["📄 CSV Forecast (30 days)", "🧍 Individual Input (30 days)"]
)

# ================== MODE 1: CSV FORECAST ==================
if mode == "📄 CSV Forecast (30 days)":
    st.subheader("📄 Upload Historical CSV")
    uploaded = st.file_uploader("Upload a CSV with at least a 'Datetime' column and one numeric energy column.", type=["csv"])

    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded)
            # Let user pick the energy column if multiple numeric columns exist
            numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                st.error("No numeric columns found. Ensure your CSV has numeric energy values.")
            else:
                date_col = "Datetime"
                value_col = st.selectbox("📊 Select the energy column to forecast:", numeric_cols, index=0)

                # Build daily series
                daily_series = make_daily_series(raw_df, date_col=date_col, value_col=value_col)

                # Start point options
                st.markdown("### 🧭 Forecast Start")
                start_opt = st.radio(
                    "Start from:",
                    ["Day after last date in data (default)", "Custom date in history (requires ≥ 30 prior days)"],
                    index=0
                )

                if start_opt == "Day after last date in data (default)":
                    # Use the last 30 daily points
                    last_30_vals = daily_series.tail(WINDOW_SIZE).values
                    preds_raw = forecast_next_30_from_window(last_30_vals)
                    start_date = daily_series.index[-1]
                    future_dates = pd.date_range(start=start_date + timedelta(days=1), periods=30, freq="D")

                else:
                    min_allowed = daily_series.index.min() + timedelta(days=WINDOW_SIZE-1)
                    max_allowed = daily_series.index.max()
                    st.caption(f"Pick a date between **{min_allowed.date()}** and **{max_allowed.date()}**.")
                    custom_start = st.date_input("📅 Forecast starting the day after:", value=max_allowed.date())
                    custom_start = pd.to_datetime(custom_start)

                    # Ensure there are 30 days prior to the chosen start date
                    window_end = pd.to_datetime(custom_start)  # last observed day
                    prior = daily_series.loc[:window_end]
                    if len(prior) < WINDOW_SIZE:
                        st.error(f"Not enough prior days before {window_end.date()} (need ≥ {WINDOW_SIZE}). Pick a later date.")
                        st.stop()

                    last_30_vals = prior.tail(WINDOW_SIZE).values
                    preds_raw = forecast_next_30_from_window(last_30_vals)
                    start_date = window_end
                    future_dates = pd.date_range(start=start_date + timedelta(days=1), periods=30, freq="D")

                # Results table
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_MW": preds_raw.flatten()})
                st.success("✅ Forecast generated!")
                st.subheader("🔮 30-Day Forecast")
                st.dataframe(forecast_df)

                # Plot
                st.subheader("📊 Forecast Plot")
                plot_csv_forecast(daily_series, future_dates, preds_raw, tail=100)

                # Download
                st.download_button(
                    "📥 Download Forecast CSV",
                    data=forecast_df.to_csv(index=False).encode(),
                    file_name="lstm_forecast_30d.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    else:
        st.info("📌 Upload a CSV to begin. Tip: the app auto-resamples hourly data to daily mean before forecasting.")

# ================== MODE 2: INDIVIDUAL INPUT ==================
else:
    st.subheader("🧍 Individual Input → 30-Day Forecast")

    st.markdown(
        "You can either **upload a CSV** so the app auto-fills the last 30 daily averages, "
        "or **paste exactly 30 daily average values** (comma-separated)."
    )

    # Optional CSV to auto-fill last 30
    opt_csv = st.file_uploader("📂 (Optional) Upload a CSV to auto-fill last 30 days", type=["csv"], key="manual_csv")

    last_30_str_default = ""
    start_date_default = datetime.today().date()

    if opt_csv:
        try:
            raw_df2 = pd.read_csv(opt_csv)
            value_col2 = None
            if "Datetime" not in raw_df2.columns:
                st.error("CSV must include a 'Datetime' column.")
                st.stop()
            # auto-detect value column
            num_cols2 = raw_df2.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols2) == 0:
                st.error("No numeric columns found in the CSV.")
                st.stop()
            value_col2 = st.selectbox("📊 Select the energy column:", num_cols2, index=0, key="manual_value_col")
            series2 = make_daily_series(raw_df2, value_col=value_col2)
            if len(series2) >= WINDOW_SIZE:
                last_30_str_default = ", ".join([f"{v:.4f}" for v in series2.tail(WINDOW_SIZE).values])
                start_date_default = series2.index[-1].date()
                st.info("✅ Loaded last 30 daily averages from the CSV. You can still edit them below if needed.")
            else:
                st.warning(f"CSV did not have ≥ {WINDOW_SIZE} daily points. Please paste values manually.")
        except Exception as e:
            st.error(f"⚠️ Error reading optional CSV: {e}")

    # Inputs
    start_date = st.date_input("📅 Baseline window ends on (the day before forecasting starts):", value=start_date_default)
    last_30_text = st.text_area(
        "🧮 Paste exactly 30 daily average MW values (comma-separated)",
        value=last_30_str_default,
        height=120,
        placeholder="e.g., 3521, 3602, 3499, ... (30 values)"
    )

    if st.button("🔮 Forecast Next 30 Days"):
        try:
            # Parse values
            vals = [float(x.strip()) for x in last_30_text.split(",") if x.strip() != ""]
            if len(vals) != WINDOW_SIZE:
                st.error(f"⚠️ Please provide exactly {WINDOW_SIZE} values. You provided {len(vals)}.")
                st.stop()

            # Forecast
            preds_raw = forecast_next_30_from_window(np.array(vals))
            start_date_dt = pd.to_datetime(start_date)
            future_dates = pd.date_range(start=start_date_dt + timedelta(days=1), periods=30, freq="D")

            # Table
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted_MW": preds_raw.flatten().round().astype(int)
            })

            st.success("✅ Forecast generated!")
            st.subheader("🔮 30-Day Forecast")
            st.dataframe(forecast_df)

            # Plot baseline + forecast
            st.subheader("📊 Baseline + Forecast Plot")
            plot_manual_forecast(np.array(vals), start_date_dt, preds_raw)

            # Download
            st.download_button(
                "📥 Download Forecast CSV",
                data=forecast_df.to_csv(index=False).encode(),
                file_name="lstm_forecast_30d_manual.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ------------------- ABOUT -------------------
with st.expander("ℹ️ About this app"):
    st.markdown(
        "- Uses an **LSTM** trained on **daily mean** energy consumption (hourly data is resampled to daily).\n"
        "- Forecast horizon: **30 days**.\n"
        "- CSV mode lets you start forecasting **after the last date** or from a **custom historical date** (if ≥ 30 prior days exist).\n"
        "- Individual mode accepts **exactly 30 daily averages** as the baseline window, or auto-fills from a CSV."
    )
