import streamlit as st
import pandas as pd
import joblib
import os

# Load model and encoders
model = joblib.load("fertilizer_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

st.set_page_config(page_title="AI Fertilizer Recommendation Agent", page_icon="🌱")

st.title("🌱 AI Fertilizer Recommendation Agent")

temperature = st.number_input("Temperature", 0, 100, 25)
humidity = st.number_input("Humidity", 0, 100, 50)
moisture = st.number_input("Moisture", 0, 100, 40)

soil = st.selectbox("Soil Type", soil_encoder.classes_)
crop = st.selectbox("Crop Type", crop_encoder.classes_)

nitrogen = st.number_input("Nitrogen", 0, 150, 20)
potassium = st.number_input("Potassium", 0, 150, 20)
phosphorous = st.number_input("Phosphorous", 0, 150, 20)

if st.button("Recommend Fertilizer"):

    soil_value = soil_encoder.transform([soil])[0]
    crop_value = crop_encoder.transform([crop])[0]

    sample = pd.DataFrame(
        [[
            temperature,
            humidity,
            moisture,
            soil_value,
            crop_value,
            nitrogen,
            potassium,
            phosphorous
        ]],
        columns=[
            "Temparature",
            "Humidity",
            "Moisture",
            "Soil Type",
            "Crop Type",
            "Nitrogen",
            "Potassium",
            "Phosphorous"
        ]
    )

    prediction = model.predict(sample)

    fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

    st.success(f"Recommended Fertilizer: {fertilizer}")

    st.subheader("Application Schedule")
    st.write("""
    - Day 1: Apply 50% of the fertilizer.
    - Day 15: Apply the remaining 50%.
    - Irrigate immediately after application.
    """)

    record = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Moisture": moisture,
        "Soil Type": soil,
        "Crop Type": crop,
        "Nitrogen": nitrogen,
        "Potassium": potassium,
        "Phosphorous": phosphorous,
        "Recommended Fertilizer": fertilizer
    }])

    if os.path.exists("farm_records.csv"):
        record.to_csv("farm_records.csv", mode="a", header=False, index=False)
    else:
        record.to_csv("farm_records.csv", index=False)

st.subheader("Farm Records")

if st.button("View Records"):

    if os.path.exists("farm_records.csv"):
        st.dataframe(pd.read_csv("farm_records.csv"))
    else:
        st.warning("No records found.")