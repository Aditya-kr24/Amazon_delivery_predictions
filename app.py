# app.py
import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from math import radians, sin, cos, asin, sqrt

# -----------------------
# Utility Functions
# -----------------------
def haversine(lon1, lat1, lon2, lat2):
    try:
        lon1, lat1, lon2, lat2 = map(float, (lon1, lat1, lon2, lat2))
    except:
        return None
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

@st.cache_resource
def load_model():
    files = [f for f in os.listdir("models") if f.endswith(".joblib")]
    if not files:
        st.error("⚠ No model found in models/. Run training first.")
        return None
    return joblib.load(os.path.join("models", files[0]))

@st.cache_data
def load_data():
    if os.path.exists("data/processed_amazon_delivery.csv"):
        return pd.read_csv("data/processed_amazon_delivery.csv")
    else:
        st.warning("⚠ Processed dataset not found (data/processed_amazon_delivery.csv).")
        return None

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Amazon Delivery Time Predictor")

model = load_model()
data = load_data()

# -----------------------
# Sidebar Navigation
# -----------------------
menu = st.sidebar.radio("📍 Navigate", ["Prediction", "Data Insights"])

# -----------------------
# Prediction Page
# -----------------------
if menu == "Prediction":
    st.header("📝 Enter Order Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Locations")
        store_lat = st.number_input("Store Latitude", value=12.9716, format="%.6f")
        store_lon = st.number_input("Store Longitude", value=77.5946, format="%.6f")
        drop_lat = st.number_input("Drop Latitude", value=12.9279, format="%.6f")
        drop_lon = st.number_input("Drop Longitude", value=77.6271, format="%.6f")

    with col2:
        st.subheader("⏰ Time & Pickup")
        order_hour = st.slider("Order Hour (0-23)", 0, 23, 12)
        order_dayofweek = st.selectbox("Order Day of Week", options=list(range(7)), 
                                       format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        pickup_hour = st.slider("Pickup Hour (0-23)", 0, 23, 13)
        order_to_pickup_hours = st.number_input("Order→Pickup (hrs)", value=0.5, step=0.1)

    with col3:
        st.subheader("👤 Agent & Context")
        agent_age = st.number_input("Agent Age", min_value=18, max_value=80, value=30)
        agent_rating = st.number_input("Agent Rating (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

        weather = st.selectbox("Weather", ["clear","rainy","cloudy","fog","storm","missing"])
        traffic = st.selectbox("Traffic", ["low","medium","high","jam","missing"])
        vehicle = st.selectbox("Vehicle", ["bike","motorcycle","car","van","truck","missing"])
        area = st.selectbox("Area", ["urban","metropolitan","suburban","rural","missing"])
        category = st.text_input("Category", "grocery")

    if st.button("🔮 Predict Delivery Time", use_container_width=True):
        distance_km = haversine(store_lon, store_lat, drop_lon, drop_lat)

        input_df = pd.DataFrame([{
            'distance_km': distance_km,
            'order_hour': order_hour,
            'order_dayofweek': order_dayofweek,
            'pickup_hour': pickup_hour,
            'order_to_pickup_hours': order_to_pickup_hours,
            'Agent_Age': agent_age,
            'Agent_Rating': agent_rating,
            'Weather': str(weather).lower(),
            'Traffic': str(traffic).lower(),
            'Vehicle': str(vehicle).lower(),
            'Area': str(area).lower(),
            'Category': str(category).lower()
        }])

        pred = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div style="padding:20px; border-radius:15px; background-color:#f9f9f9; border:1px solid #ddd; text-align:center; margin-top:20px;">
                <h2>⏱ Estimated Delivery Time</h2>
                <h1 style="color:#2E86C1;">{pred:.2f} hours</h1>
                <p><b>Distance:</b> {distance_km:.2f} km</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------
# Insights Page
# -----------------------
elif menu == "Data Insights":
    st.header("📊 Delivery Data Insights")

    if data is not None:
        col1, col2 = st.columns(2)

        # 1. Delivery Time Distribution
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(data['Delivery_Time'], bins=40, kde=True, ax=ax)
            ax.set_title("Delivery Time Distribution (hours)")
            st.pyplot(fig)

        # 2. Distance vs Delivery Time
        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(x="distance_km", y="Delivery_Time", data=data, alpha=0.5, ax=ax)
            ax.set_title("Distance vs Delivery Time")
            st.pyplot(fig)

        # 3. Traffic Impact
        if "Traffic" in data.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x="Traffic", y="Delivery_Time", data=data, ax=ax)
            ax.set_title("Traffic vs Delivery Time")
            st.pyplot(fig)

        # 4. Correlation Heatmap
        numeric_cols = data.select_dtypes(include=['float64','int64']).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
