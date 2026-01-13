import streamlit as st
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "rf_columns.pkl"))

st.set_page_config(page_title="Shivam House Price Predictor")
st.title("🏠 Shivam – Indian House Price Prediction")

# ---------------- INPUTS ----------------
size = st.slider("Area (sqft)", 300, 6000, 1200)
bhk = st.slider("BHK", 1, 6, 2)
age = st.slider("Age of Property", 0, 50, 5)
floor = st.slider("Floor No", 0, 50, 1)
total_floors = st.slider("Total Floors", 1, 60, 5)

parking = st.selectbox("Parking", ["yes", "no"])
security = st.selectbox("Security", ["yes", "no"])

cities = [c.replace("city_", "") for c in columns if c.startswith("city_")]
localities = [c.replace("locality_", "") for c in columns if c.startswith("locality_")]
types = [c.replace("property_type_", "") for c in columns if c.startswith("property_type_")]

city = st.selectbox("City", cities)
locality = st.selectbox("Locality", localities)
ptype = st.selectbox("Property Type", types)

# ---------------- VECTOR ----------------
x = np.zeros(len(columns))

def set_col(name, val=1):
    if name in columns:
        x[columns.index(name)] = val

set_col("size_in_sqft", size)
set_col("bhk", bhk)
set_col("age_of_property", age)
set_col("floor_no", floor)
set_col("total_floors", total_floors)

set_col(f"parking_{parking}")
set_col(f"security_{security}")
set_col(f"city_{city}")
set_col(f"locality_{locality}")
set_col(f"property_type_{ptype}")

# ---------------- PREDICT ----------------
if st.button("Predict Price"):
    price = model.predict([x])[0]
    st.success(f"💰 Estimated Price: ₹ {price:.2f} Lakhs")