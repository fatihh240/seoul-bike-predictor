🚲 Seoul Bike Demand Forecasting

Live App:
🔗 https://fatih-seoul-bike-predictor.streamlit.app/

📌 Project Overview

This project predicts hourly bike rental demand in Seoul using a machine learning regression model.

The model is trained on the Seoul Bike Sharing Dataset and deployed as a fully functional interactive web application using Streamlit Cloud.

The objective is to forecast bike rental demand based on:

Weather conditions

Time-related features

Seasonal information

🧠 Modeling Approach
🔹 Target Variable

Rented Bike Count

Due to right-skewed distribution (count data), the target variable was transformed using:

log(y + 1)


Predictions are converted back using:

expm1(prediction)

⚙️ Feature Engineering

Key engineered features:

Cyclical encoding for Hour

Cyclical encoding for Month

Day of Week extraction

Weather interaction features

Precipitation flag

Feature engineering was carefully mirrored in deployment to maintain pipeline consistency.

🤖 Model

Algorithm: LightGBM Regressor

Hyperparameter tuning via Random Search

Early stopping to prevent overfitting

📊 Performance
Metric	Train	Test
R²	0.9288	0.9033
RMSE (original scale)	147.3	185.6

The model shows strong generalization with minimal overfitting.

🌐 Deployment

The model is deployed using:

Streamlit

Streamlit Community Cloud

Python 3.12 runtime (ensuring LightGBM compatibility)

The dashboard:

Restricts user inputs to training data ranges

Applies identical feature transformations

Automatically performs inverse log transformation

Displays top feature importances for transparency

📁 Repository Structure
├── app.py
├── lgbm_model.pkl
├── model_columns.pkl
├── input_stats.pkl
├── feature_importance.pkl
├── requirements.txt
├── runtime.txt

🛡️ Architectural Principles

No training in production

No dynamic feature creation at inference

Strict feature column alignment

Controlled input ranges to prevent out-of-distribution errors

📈 Future Improvements

Add SHAP explainability

Time-series cross validation

Forecasting future dates

Model comparison (XGBoost / CatBoost)

👨‍💻 Author

Machine Learning & Data Science Project
Built and deployed end-to-end.
