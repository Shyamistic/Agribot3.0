import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import random
import tensorflow as tf
import joblib
from deep_translator import GoogleTranslator

# --- Neon Nature Responsive Theme CSS ---
st.set_page_config(page_title="Agribot – Smart Farming", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
        html, body, .stApp { background: linear-gradient(115deg,#12251f 0%,#176440 100%);
            font-family:'Montserrat','Segoe UI',Arial,sans-serif;}
        section.main > div { max-width:1200px; margin:auto;}
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            color: #f7fff3 !important; font-weight:900 !important;
            text-shadow:0 4px 16px #1c2b1c88; margin-bottom:0.5em; letter-spacing:-1px;}
        label, .stTextInput label, .stNumberInput label, .stSelectbox label { color:#fbfffa !important; font-size:1.09em; font-weight:700;}
        .stTextInput input, .stNumberInput input {
            border-radius:6px !important; border:2px solid #32ff79 !important;
            background:#183728 !important; color:#fff !important; font-weight:700 !important;}
        .stSelectbox > div > div > div {background:#17381f !important;color:#edffee !important;}
        .stSlider>div {background:#112619 !important;}
        .stSlider>div>div>span {color:#ffffff !important;font-weight:bold !important;background:transparent !important;}
        .stSlider label {color:#eafffa !important;}
        .stButton>button, .stDownloadButton>button {
            border-radius:7px !important;
            background:linear-gradient(90deg,#32ff79 0%,#0bb36c 100%) !important;
            color:#142f21 !important; font-weight:900; font-size:1.1em !important; box-shadow:0 2px 9px #66d1aa77;}
        .output-card {
            background:linear-gradient(90deg,#17de7e 0%,#20552c 100%);
            color:#fff !important; padding:16px 28px;border-radius:17px !important;font-size:1.18em;font-weight:800 !important;
            box-shadow:0 6px 16px #13301577;margin:8px 0;}
        .stMarkdown {color:#f7fff3 !important; font-weight:700;}
    </style>
""", unsafe_allow_html=True)

# --- TITLE & SUBTITLE ---
st.markdown("<h1 style='font-size:2.4em;margin-bottom:0.1em;'>Agribot: AI Smart Farming — Deep Learning & Analytics</h1>", unsafe_allow_html=True)
st.markdown("<div style='font-size:1.2em;color:#71ffa8;font-weight:600;margin-bottom:1.5em;'>🌱 Empowering farmers with data-driven intelligence for sustainable growth</div>", unsafe_allow_html=True)

# --- Load crop/state data ---
try:
    df = pd.read_csv('data/crops_clean.csv')
    crops = sorted(df['Crop'].dropna().unique())
    states = sorted(df['State'].dropna().unique())
except Exception as e:
    st.warning("Could not load crops_clean.csv. Using demo options.")
    crops = ["Wheat", "Rice", "Maize", "Pulses", "Cotton", "Arecanut"]
    states = ["Bihar", "Punjab", "Maharashtra", "Assam", "UP"]
    df = pd.DataFrame([])

# --- Inputs Section ---
with st.container():
    st.subheader("🌾 Farm & Crop Inputs")
    area = st.slider('Area (hectares)', 0.1, 100.0, 2.5, 0.1)
    production = st.slider('Production (tons)', 0.1, 100.0, 3.8, 0.1)
    rainfall = st.slider('Annual Rainfall (mm)', 0.0, 2000.0, 750.0, 1.0)
    fertilizer = st.slider('Fertilizer used (kg)', 0.0, 1000.0, 60.0, 1.0)
    pesticide = st.slider('Pesticide used (kg)', 0.0, 500.0, 10.0, 1.0)
    crop = st.selectbox("🌾 Crop", crops)
    state = st.selectbox("📍 State", states)

# --- Weather API Demo ---
st.markdown("---")
st.subheader("☁️ Current Weather Check")
city = st.text_input("City (for weather info, e.g., 'Delhi')", "Delhi")
if st.button("Get Current Weather"):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=d0724cdbb8d00865d45c8f387977d78c&units=metric"
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            st.markdown(
                f"<div class='output-card'>Current in {city}: {data['main']['temp']}°C, {data['weather'][0]['description']}, Rainfall 1h: {data.get('rain', {}).get('1h', 0)} mm</div>",
                unsafe_allow_html=True)
        else:
            st.warning("City not found or API limit reached.")
    except Exception:
        st.warning("Weather API failed to respond.")

# --- IoT Sensor Demo ---
st.markdown("---")
st.subheader("🔌 IoT Sensor Simulation")
if st.button("Get Live Sensor Values"):
    soil = round(random.uniform(10, 45), 2)
    t = round(random.uniform(22, 36), 2)
    hum = round(random.uniform(40, 90), 2)
    st.markdown(
        f"<div class='output-card'>🌱 Soil Moisture: {soil}% | 🌡️ Temp: {t}°C | 💧 Humidity: {hum}%</div>",
        unsafe_allow_html=True
    )

# --- Deep Learning Prediction (demo/fallback only, add your actual model code) ---
def generate_advice(predicted_yield, rainfall, fertilizer, crop=None, state=None):
    advice = []
    if predicted_yield < 1:
        advice.append("Low yield predicted. Consider additional fertilizers or switch crop variety.")
    if rainfall < 500:
        advice.append("Low rainfall expected. Use water-saving irrigation.")
    if fertilizer < 50:
        advice.append("Fertilizer input low. Review soil and crop needs.")
    if crop == "Onion" and state == "Assam":
        advice.append("Check for local onion varieties suitable for Assam soil and climate.")
    if not advice:
        advice.append("Inputs indicate good yield. Maintain regular monitoring and good practices.")
    return advice

st.markdown("---")
st.subheader("🤖 Smart Yield AI & Advice")
if st.button("Predict Yield & Get Advice"):
    # ----- Replace this logic with your ML/ensemble model prediction (currently a demo formula) -----
    predicted_yield = round(area * production * (rainfall/1000) * (fertilizer + 10)/(pesticide + 10)/100, 2)
    st.markdown(f"<div class='output-card'>Estimated Yield: {predicted_yield} t/ha</div>", unsafe_allow_html=True)
    advices = generate_advice(predicted_yield, rainfall, fertilizer, crop, state)
    st.markdown(f"<div class='output-card'>Recommendation: {' '.join(advices)}</div>", unsafe_allow_html=True)
    # --- Yield trend graph ---
    if not df.empty:
        graph = df[(df['Crop'] == crop) & (df['State'] == state) & df['Yield'].notnull() & df['Crop_Year'].notnull()]
        if not graph.empty:
            st.line_chart(graph.set_index('Crop_Year')['Yield'])
        else:
            st.info("No historical yield data for this crop and state combination.")
    else:
        st.info("No dataset available for yield trends.")

if st.button("Show Hindi Advice"):
    st.markdown("<div class='output-card'>सलाह: सिंचाई बेहतर बनाएं, उर्वरक योजना सुधारें, और मंडी मूल्य जांचें।</div>", unsafe_allow_html=True)

st.download_button("⬇️ Download Advice (.txt)", "Sample advice here...", file_name="agribot_advice.txt")

# --- Latest Govt MSP Card ---
st.markdown("---")
st.subheader("💸 Get Latest Govt Crop Price (Demo)")
price_demo = {"Wheat": 2125, "Rice": 2183, "Maize": 1962, "Cotton": 6160, "Pulses": 5250, "Arecanut": 35100}
st.markdown(
    f"<div class='output-card' style='background:linear-gradient(90deg,#1edc88 0%,#21815b 100%);margin-top:10px;'>Latest MSP for {crop}: ₹{price_demo.get(crop,'N/A')} per quintal</div>",
    unsafe_allow_html=True
)

# --- Yield correlation graph ---
st.markdown("---")
st.subheader("📈 Rainfall vs Yield Trend (Insights)")
if not df.empty:
    corr_df = df[(df['Crop'] == crop) & (df['State'] == state) & df['Yield'].notnull() & df['Annual_Rainfall'].notnull()]
    if not corr_df.empty:
        st.scatter_chart(corr_df[['Annual_Rainfall', 'Yield']])
    else:
        st.info("No rainfall-yield data for this crop and state.")
else:
    st.info("No dataset for rainfall correlation.")

# --- Regional Yield Comparison ---
st.subheader("📊 Regional Comparison: Yield by State")
if not df.empty:
    cmp_df = df[df['Crop'] == crop]
    if 'State' in cmp_df.columns and 'Yield' in cmp_df.columns:
        st.bar_chart(cmp_df.groupby('State')['Yield'].mean())
    else:
        st.info("No regional yield data.")
else:
    st.info("No dataset available for regional comparison.")

# --- Footer ---
st.markdown("""
<hr style='border:1px solid #22ed7c; margin: 30px 0;'>
<center>
<b style='font-size:1.2em;color:#65fa99;'>Built for the AI for ALL India Summit 2025 & powered by open tech 🚀</b>
</center>
""", unsafe_allow_html=True)
