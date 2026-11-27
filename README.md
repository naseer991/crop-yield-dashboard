# 🌾 Crop Yield Dashboard

An end-to-end ML web app for exploring and predicting crop yield (hg/ha) 
across countries, crops, and years.

🔗 **Live Demo:** https://your-app.streamlit.app
📂 **Dataset:** [Kaggle Crop Yield Prediction](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

## Features

- 📋 **Overview** — KPIs, data sample, summary stats, CSV export
- 📊 **EDA** — distributions, correlations, outliers, relationships
- 🌍 **Geo & Trends** — top/bottom countries, choropleth, crop×year heatmap
- 🤖 **Model Performance** — RF metrics, residuals, feature importance, 7-model comparison
- 🎯 **Predict** — interactive inference with sensitivity analysis

## Tech Stack

Python · Streamlit · Scikit-learn · Plotly · Pandas · NumPy

## Model

Random Forest (200 trees) with a Scikit-Learn Pipeline (StandardScaler + OneHotEncoder). 
Test R² ≈ 0.98, MAE ≈ 1,500 hg/ha.

> Note: The model and preprocessor are pre-trained and serialized (`.pkl`) for fast loading, or trained on-the-fly depending on the deployment environment.

## Project Structure

```text
crop-yield-dashboard/
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── CropYield-Prediction.ipynb # EDA and model training notebook
├── crop_yield_model.pkl    # Trained Random Forest model
├── preprocessor.pkl        # Fitted ColumnTransformer pipeline
├── data/
│   └── yield_df.csv        # Dataset
└── screenshots/            # App screenshots for README