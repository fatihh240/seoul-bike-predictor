import math
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Seoul Bike Demand Predictor", layout="centered")

# --- Cache: model and helper files stay in memory ---
@st.cache_resource
def load_assets():
    model = joblib.load("lgbm_model.pkl")
    cols = joblib.load("model_columns.pkl")
    stats = joblib.load("input_stats.pkl")

    imp_obj = joblib.load("feature_importance.pkl")
    imp = pd.Series(imp_obj["vals"], index=imp_obj["cols"]).sort_values(ascending=False)

    return model, cols, stats, imp


model, MODEL_COLS, STATS, FEAT_IMP = load_assets()

st.title("🚲 Seoul Bike Demand Predictor")
st.caption("Inference only. No training/cleaning. Consistent pipeline. Output is actual bike count.")

# --- Helper: safely get min/max ---
def mm(col, fallback_min, fallback_max):
    if col in STATS:
        return STATS[col]["min"], STATS[col]["max"]
    return fallback_min, fallback_max

# --- Sidebar inputs ---
st.sidebar.header("Input")

# Date (for Day_of_Week + Month)
day_name = st.sidebar.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
day_of_week = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_name)

month = st.sidebar.slider("Month", 1, 12, 6)
hour = st.sidebar.slider("Hour", 0, 23, 12)

# Numerical inputs: constrained by training bounds
tmin, tmax = mm("Temperature(°C)", -20, 40)
hmin, hmax = mm("Humidity(%)", 0, 100)
wmin, wmax = mm("Wind speed (m/s)", 0, 10)
vmin, vmax = mm("Visibility (10m)", 0, 2000)
smin, smax = mm("Solar Radiation (MJ/m2)", 0, 5)
rmin, rmax = mm("Rainfall(mm)", 0, 100)
snmin, snmax = mm("Snowfall (cm)", 0, 50)

temp = st.sidebar.slider("Temperature (°C)", float(tmin), float(tmax), float(max(min(15, tmax), tmin)))
hum  = st.sidebar.slider("Humidity (%)", float(hmin), float(hmax), float(max(min(50, hmax), hmin)))
wind = st.sidebar.slider("Wind speed (m/s)", float(wmin), float(wmax), float(max(min(2, wmax), wmin)))
vis  = st.sidebar.slider("Visibility (10m)", float(vmin), float(vmax), float(max(min(1000, vmax), vmin)))
solar= st.sidebar.slider("Solar Radiation (MJ/m2)", float(smin), float(smax), float(max(min(0.5, smax), smin)))
rain = st.sidebar.slider("Rainfall (mm)", float(rmin), float(rmax), 0.0)
snow = st.sidebar.slider("Snowfall (cm)", float(snmin), float(snmax), 0.0)

# Categorical
seasons = st.sidebar.selectbox("Seasons", ["Spring", "Summer", "Autumn", "Winter"])

# --- Feature engineering (dashboard side only transforms; no cleaning/training) ---
def build_features():
    # cyclical hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    # month cyclical
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    # precipitation flag
    is_precip = int((rain > 0) or (snow > 0))

    # Raw row (with names the model saw during training!)
    row = {
        "Temperature(°C)": temp,
        "Humidity(%)": hum,
        "Wind speed (m/s)": wind,
        "Visibility (10m)": vis,
        "Solar Radiation (MJ/m2)": solar,
        "Rainfall(mm)": rain,
        "Snowfall (cm)": snow,
        "Day_of_Week": day_of_week,
        "Month": month,
        "Month_sin": month_sin,
        "Month_cos": month_cos,
        "Hour_sin": hour_sin,
        "Hour_cos": hour_cos,
        "Is_Precipitation": is_precip,
        "Seasons": seasons,
    }

    df_raw = pd.DataFrame([row])

    X = pd.get_dummies(df_raw, drop_first=True)
    X = X.reindex(columns=MODEL_COLS, fill_value=0)

    return X

# --- Prediction ---
if st.button("Predict"):
    X_in = build_features()
    if X_in is not None:
        pred_log = float(model.predict(X_in)[0])
        pred = float(np.expm1(pred_log))  # ✅ inverse transform (critical)

        st.success(f"✅ Predicted Rented Bike Count: **{pred:.0f}**")
        st.caption(f"(Model output was log(y+1): {pred_log:.4f} → converted with expm1)")

        # Transparency: top-3 importance
        st.subheader("What Does the Model Base Its Decision On?")
        top3 = FEAT_IMP.head(3).copy()
        st.bar_chart(top3)

        # Show user inputs if desired
        with st.expander("Input Summary"):
            st.write({
                "Day_of_Week": day_name,
                "Month": month,
                "Hour": hour,
                "Temperature(°C)": temp,
                "Humidity(%)": hum,
                "Wind speed (m/s)": wind,
                "Visibility (10m)": vis,
                "Solar Radiation (MJ/m2)": solar,
                "Rainfall(mm)": rain,
                "Snowfall (cm)": snow,
                "Seasons": seasons
            })
