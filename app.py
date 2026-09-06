
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Farmer Crop Profit Predictor", page_icon="🌾", layout="wide")

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "profit_prediction_model.joblib"
DATA_PATH = BASE / "data" / "farmer_crop_profit_dataset.csv"

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

st.title("🌾 Farmer Crop Profit Prediction System")
st.caption("Machine Learning based decision-support prototype for crop profitability.")

st.sidebar.header("Farmer / Crop Inputs")
crop = st.sidebar.selectbox("Crop", sorted(data["Crop"].unique()))
district = st.sidebar.selectbox("District", sorted(data["District"].unique()))
season = st.sidebar.selectbox("Season", sorted(data["Season"].unique()))
soil = st.sidebar.selectbox("Soil Type", sorted(data["Soil_Type"].unique()))
irrigation = st.sidebar.selectbox("Irrigation Type", sorted(data["Irrigation_Type"].unique()))
year = st.sidebar.slider("Year", int(data.Year.min()), int(data.Year.max())+1, 2026)
land = st.sidebar.number_input("Land Area (acres)", 0.1, 100.0, 2.0, 0.1)
rain = st.sidebar.number_input("Rainfall (mm)", 100.0, 3000.0, 700.0, 10.0)
temp = st.sidebar.number_input("Temperature (°C)", 5.0, 45.0, 25.0, 0.5)
humidity = st.sidebar.number_input("Humidity (%)", 10.0, 100.0, 65.0, 1.0)
ph = st.sidebar.number_input("Soil pH", 4.0, 9.5, 6.7, 0.1)

st.sidebar.subheader("Production Costs")
seed = st.sidebar.number_input("Seed Cost (₹)", 0.0, 10000000.0, 8000.0, 500.0)
fert = st.sidebar.number_input("Fertilizer Cost (₹)", 0.0, 10000000.0, 12000.0, 500.0)
pest = st.sidebar.number_input("Pesticide Cost (₹)", 0.0, 10000000.0, 7000.0, 500.0)
lab = st.sidebar.number_input("Labour Cost (₹)", 0.0, 10000000.0, 20000.0, 500.0)
irr = st.sidebar.number_input("Irrigation Cost (₹)", 0.0, 10000000.0, 6000.0, 500.0)
other = st.sidebar.number_input("Other Cost (₹)", 0.0, 10000000.0, 5000.0, 500.0)
yield_kg = st.sidebar.number_input("Expected Yield (kg)", 0.0, 10000000.0, 12000.0, 100.0)
price = st.sidebar.number_input("Expected Market Price (₹/kg)", 0.1, 10000.0, 25.0, 0.5)

input_df = pd.DataFrame([{
    "Year": year, "District": district, "Crop": crop, "Season": season,
    "Soil_Type": soil, "Irrigation_Type": irrigation, "Land_Area_Acres": land,
    "Rainfall_mm": rain, "Temperature_C": temp, "Humidity_pct": humidity,
    "Soil_pH": ph, "Seed_Cost_INR": seed, "Fertilizer_Cost_INR": fert,
    "Pesticide_Cost_INR": pest, "Labour_Cost_INR": lab, "Irrigation_Cost_INR": irr,
    "Other_Cost_INR": other, "Expected_Yield_kg": yield_kg,
    "Market_Price_INR_per_kg": price
}])

total_cost = seed + fert + pest + lab + irr + other
revenue = yield_kg * price
calculated_profit = revenue - total_cost

predicted_profit = float(model.predict(input_df)[0])
profit_margin = (predicted_profit / revenue * 100) if revenue else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cost", f"₹{total_cost:,.0f}")
c2.metric("Expected Revenue", f"₹{revenue:,.0f}")
c3.metric("ML Predicted Profit", f"₹{predicted_profit:,.0f}")
c4.metric("Profit Margin", f"{profit_margin:.1f}%")

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("📊 Cost Breakdown")
    cost_df = pd.DataFrame({
        "Cost Type": ["Seed","Fertilizer","Pesticide","Labour","Irrigation","Other"],
        "Amount (₹)": [seed,fert,pest,lab,irr,other]
    }).set_index("Cost Type")
    st.bar_chart(cost_df)

with right:
    st.subheader("💡 Decision")
    if predicted_profit > 0:
        st.success("Potentially profitable under the entered assumptions.")
    else:
        st.error("Potential loss under the entered assumptions.")
    st.write(f"Calculator profit: **₹{calculated_profit:,.0f}**")
    st.write(f"ML prediction: **₹{predicted_profit:,.0f}**")
    st.info("Use the prediction as decision support, not as a guaranteed market return.")

st.subheader("🌱 Crop Comparison from Historical Dataset")
comparison = data.groupby("Crop").agg(
    Average_Profit_INR=("Profit_INR","mean"),
    Average_Yield_kg=("Expected_Yield_kg","mean"),
    Average_Price_INR_per_kg=("Market_Price_INR_per_kg","mean")
).sort_values("Average_Profit_INR", ascending=False)
st.dataframe(comparison.style.format({
    "Average_Profit_INR":"₹{:,.0f}",
    "Average_Yield_kg":"{:,.0f}",
    "Average_Price_INR_per_kg":"₹{:.2f}"
}), use_container_width=True)

st.subheader("📈 Historical Profit by Crop")
chart = data.groupby("Crop")["Profit_INR"].mean().sort_values(ascending=False)
st.bar_chart(chart)
