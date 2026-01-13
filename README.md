# Indian_House_Price_Prediction_Using_ML
Machine Learning Project to Predict Indian House Price Uisng csv data and streamlit
This project is a Machine Learning–based Indian House Price Prediction System designed to estimate realistic house prices using historical housing data stored in a CSV file. The primary objective of this project is to understand how different property features influence house prices and to build a predictive system that can provide market-based price estimates, not exact deal prices.
The project uses supervised learning regression techniques to analyze housing data and predict prices based on multiple property-related factors such as area, number of bedrooms, location, property type, and amenities. The system is implemented in Python and deployed as an interactive web application using Streamlit, allowing users to easily input property details and receive a predicted price.

🧠 How the Project Works
Dataset Loading
The system loads an Indian house price dataset from a CSV file. This dataset contains historical housing information such as:
Property size (square feet)
Number of bedrooms (BHK)
City and locality
Property type
Age of property
Floor information
Parking and security availability
House price (in lakhs)
Data Preprocessing
Before training the model, the dataset is cleaned and prepared:
Column names are standardized
Missing and invalid values are removed
Numerical features are converted to proper numeric formats
Categorical features (such as city, locality, and property type) are encoded
Rare categories are grouped to reduce noise and overfitting
Model Training
Two Machine Learning models are used:
Linear Regression is trained as a baseline model to understand basic price relationships.
Random Forest Regressor is trained as the main model to capture complex and non-linear relationships in the data.
Random Forest is chosen because it performs well on real-world tabular data and provides more stable predictions compared to simple linear models.
Model Evaluation
The trained models are evaluated using standard regression metrics:
R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
These metrics help measure how closely the predicted prices match actual market prices.
Model Deployment
After training, the final model is saved and used inside a Streamlit web application. This allows users to interact with the model without needing to understand the underlying Machine Learning code.
🖥️ What the Application Does
Accepts property details from the user through a simple web interface
Processes the inputs using the trained Machine Learning model
Predicts the estimated house price in Indian Rupees (Lakhs)
Displays the result instantly in a user-friendly format
The application predicts approximate market values, similar to real-world real estate platforms. It does not claim to provide exact selling prices.
▶ How to Use the Project
Step 1: Install Dependencies
Install all required Python libraries using:
Copy code
Bash
pip install -r requirements.txt
Step 2: Train the Models
Train the baseline and final models using:
Copy code
Bash
python train_model_lr.py
python train_model_rf.py
This step processes the dataset and saves trained models for prediction.
Step 3: Run the Web Application
Launch the Streamlit app using:
Copy code
Bash
streamlit run app.py
A browser window will open where you can:
Enter house details
Select location and property features
View the predicted house price

🎯 Project Purpose and Learning Outcomes
This project is intended for:
Learning how Machine Learning regression models work
Understanding data preprocessing and feature engineering
Applying Random Forest for real-world prediction tasks
Deploying ML models using Streamlit
Building a complete end-to-end ML project
