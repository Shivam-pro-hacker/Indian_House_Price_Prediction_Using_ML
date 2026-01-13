import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD CSV ----------------
csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("❌ No CSV file found in data folder")

DATA_PATH = os.path.join(DATA_DIR, csv_files[0])
print("✅ Using dataset:", csv_files[0])

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip()

print("📌 Columns found:", df.columns.tolist())

# ---------------- COLUMN MAPPING (CRITICAL FIX) ----------------
COLUMN_MAP = {
    "size_in_sqft": "size_in_sqft",
    "bhk": "bhk",
    "city": "city",
    "locality": "locality",
    "property_type": "property_type",
    "age_of_property": "age_of_property",
    "floor_no": "floor_no",
    "total_floors": "total_floors",
    "parking": "parking_space",      # 👈 FIX
    "security": "security",
    "price_in_lakhs": "price_in_lakhs"
}

for key, col in COLUMN_MAP.items():
    if col not in df.columns:
        raise ValueError(f"❌ Required column not found: {col}")

df = df[list(COLUMN_MAP.values())]
df.columns = list(COLUMN_MAP.keys())

# ---------------- CLEAN DATA ----------------
numeric_cols = [
    "size_in_sqft", "bhk", "age_of_property",
    "floor_no", "total_floors", "price_in_lakhs"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# ---------------- BINARY NORMALIZATION ----------------
def normalize_binary(x):
    return "yes" if str(x).lower() in ["yes", "y", "1", "true"] else "no"

df["parking"] = df["parking"].apply(normalize_binary)
df["security"] = df["security"].apply(normalize_binary)

# ---------------- REDUCE RARE CATEGORIES ----------------
def reduce_categories(series, min_count=50):
    counts = series.value_counts()
    return series.apply(lambda x: x if counts[x] >= min_count else "other")

df["city"] = reduce_categories(df["city"])
df["locality"] = reduce_categories(df["locality"])
df["property_type"] = reduce_categories(df["property_type"])

# ---------------- ONE HOT ENCODING ----------------
df = pd.get_dummies(
    df,
    columns=["city", "locality", "property_type", "parking", "security"],
    drop_first=True
)

# ---------------- TRAIN ----------------
X = df.drop("price_in_lakhs", axis=1)
y = df["price_in_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\n📊 Linear Regression Results")
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE      :", mean_absolute_error(y_test, y_pred))
print("RMSE     :", mean_squared_error(y_test, y_pred) ** 0.5)

# ---------------- SAVE ----------------
joblib.dump(model, os.path.join(MODEL_DIR, "linear_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "linear_columns.pkl"))

print("\n✅ Linear Regression model trained successfully")