# 🏠 Indian House Price Prediction Using Machine Learning

This project predicts **house prices in India** using Machine Learning and a CSV dataset.
It uses real housing data to estimate **market-based house prices** and is deployed
as a **Streamlit web application**.

---

## 📌 Project Summary

- CSV-based Machine Learning project
- Predicts house prices based on property features
- Uses regression models
- Interactive UI built with Streamlit
- Designed for college-level ML understanding

---

## 🧠 Machine Learning Models

- **Linear Regression** – baseline model
- **Random Forest Regressor** – improved accuracy model

Random Forest is used as the main model because it captures
non-linear relationships better than linear models.

---

## 🗂️ Features Used

- Area (square feet)
- Number of bedrooms (BHK)
- City / locality
- Property type
- Age of property
- Floor details
- Parking availability
- Security availability

**Target Variable:**  
- House price (in lakhs)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure
Indian-House-Price-Prediction-Using-ML/ │ ├── data/ │   └── india_house_price.csv ├── model/ │   ├── rf_model.pkl │   └── rf_columns.pkl ├── train_model_lr.py ├── train_model_rf.py ├── app.py ├── requirements.txt └── README.md
Copy code

---

## ▶ How to Run the Project

```bash
pip install -r requirements.txt
python train_model_lr.py
python train_model_rf.py
streamlit run app.py
⚠️ Disclaimer
The predicted prices are approximate market estimates based on historical data. Actual prices may vary due to market conditions, negotiation, and other factors.
👤 Author
Shivam