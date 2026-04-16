import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Streamlit App Configuration
st.set_page_config(page_title="GAC Motors GCC Forecast", layout="wide")
st.title("🚙 GAC Motors Exports to GCC Region: Historical Dataset & Forecast")

# 1. Generate Synthetic Data
@st.cache_data
def load_data():
    dates = pd.date_range(start='2019-01-01', end='2024-12-01', freq='MS')
    volumes = []
    base_volume = 800
    for d in dates:
        year = d.year
        month = d.month
        seasonality = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12 - np.pi/2)
        
        if year == 2019: growth = 1.0
        elif year == 2020: growth = 0.9
        elif year == 2021: growth = 1.2
        elif year == 2022: growth = 1.6
        elif year == 2023: growth = 2.5
        elif year == 2024: growth = 4.0
        
        noise = np.random.normal(1.0, 0.05)
        vol = int(base_volume * growth * seasonality * noise)
        volumes.append(vol)

    df = pd.DataFrame({'ds': dates, 'y': volumes})
    return df

df = load_data()

st.subheader("📊 Historical Data (2019-2024)")
st.write("Using a parameterized expansion curve matching reported CAAM exponential leapfrog growth over the last 5 years.")
st.line_chart(df.set_index('ds')['y'])

# 2. Fit and Predict with Prophet
st.subheader("🔮 1-Year Forecast using Prophet")
with st.spinner("Fitting Prophet model parameters..."):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)

# 3. Visualize the Forecast
fig, ax = plt.subplots(figsize=(12, 5))

# Plot historical data
ax.plot(df['ds'], df['y'], label='Historical GAC GCC Exports', color='blue', marker='o')

# Plot predicted data
forecast_future = forecast[forecast['ds'] > df['ds'].max()]
ax.plot(forecast_future['ds'], forecast_future['yhat'], label='Prophet Prediction (2025)', color='red', linestyle='--', marker='x')
ax.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'], color='red', alpha=0.2, label='Confidence Interval')

ax.set_title('GAC Motors Exports to GCC Region')
ax.set_xlabel('Date')
ax.set_ylabel('Export Volumes (Units)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)

st.subheader("📈 Upcoming 6 Months Forecast Table")
st.dataframe(forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(6).rename(
    columns={'ds': 'Date', 'yhat': 'Predicted Volume', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
))
