"""
app.py
------
The web app. Run this THIRD, after generate_dataset.py and train_model.py
have both been run once.

HOW TO RUN (type this in your terminal, inside this folder):
    streamlit run app.py
"""

import streamlit as st
import joblib
import pandas as pd

# ---- Load the saved pipeline (created by train_model.py) ----
saved = joblib.load("crop_yield_model.pkl")
pipeline = saved["pipeline"]
model_name = saved["model_name"]
area_options = saved["area_options"]
item_options = saved["item_options"]

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Crop Yield Prediction")
st.write(
    "Enter the details below and this app will predict the expected "
    f"crop yield using a trained **{model_name}** model."
)

st.divider()

area = st.selectbox("Country (Area)", area_options)
item = st.selectbox("Crop (Item)", item_options)
year = st.number_input("Year", min_value=1990, max_value=2030, value=2024, step=1)
rainfall = st.number_input("Average rainfall (mm/year)", min_value=0.0, value=1000.0, step=10.0)
pesticides = st.number_input("Pesticides used (tonnes)", min_value=0.0, value=50.0, step=1.0)
avg_temp = st.number_input("Average temperature (°C)", min_value=0.0, value=25.0, step=0.5)

if st.button("Predict Yield", type="primary"):
    # Build a one-row dataframe with the SAME column names used in training.
    # The saved pipeline does the OneHotEncoding + scaling internally --
    # we don't need to do any of that by hand here.
    input_data = pd.DataFrame([{
        "Area": area,
        "Item": item,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": avg_temp,
    }])

    prediction_hg_ha = pipeline.predict(input_data)[0]
    prediction_tons_ha = prediction_hg_ha / 10000  # hg/ha -> tons/hectare

    st.success(f"### Predicted Yield: {prediction_hg_ha:,.0f} hg/ha")
    st.info(f"That's about **{prediction_tons_ha:.2f} tons per hectare**.")

st.divider()
st.caption(
    f"Model used: {model_name}  |  "
    "Trained on a synthetic dataset shaped like the Kaggle "
    "'Crop Yield Prediction Dataset' (Area, Item, Year, rainfall, "
    "pesticides, avg temp)."
)
