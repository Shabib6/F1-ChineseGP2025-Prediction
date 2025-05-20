# 🏎️ F1 Race Prediction using Gradient Boosting and FastF1 API
This project uses Machine Learning to predict final race outcomes for the 2025 Formula 1 Chinese Grand Prix. Leveraging the FastF1 API and Gradient Boosting, the model forecasts driver standings based on sector times and qualifying results.

## 📊 Features
  - Predicts race performance using **qualifying times** and **sector averages**.
  - Built using **Gradient Boosting Regressor** for non-linear pattern learning.
  - Uses **FastF1 API** to extract real-world data from the 2024 Chinese GP.
  - Handles missing data via mean imputation for improved reliability.
  - **Outputs final predicted race order and identifies the projected winner.**

## 🛠️ Technologies Used
  - Python
  - FastF1
  - scikit-learn
  - pandas, numpy
  - GradientBoostingRegressor

## 🧪 Model Evaluation
  - *Achieved Mean Absolute Error (MAE): 1.96*
  - *Correctly predicted McLaren as winner, top 6 included Verstappen, Norris, etc.*
  - *Highlighted real-world unpredictability (e.g., Ferrari disqualification)*
