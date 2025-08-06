import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# --------------------------------------------------------
# Step 1: Load and clean the data
# --------------------------------------------------------

# Path to your CSV file
csv_path = "data/future-gc00-daily-prices.csv"

# Read CSV, parse date column, and use it as index
data = pd.read_csv(
    csv_path,
    parse_dates=["Date"],
    dayfirst=False,       # Dates are in MM/DD/YYYY format
    index_col="Date"
)

# Sort data by date
data.sort_index(inplace=True)

# Set frequency to business days ('B') to prevent ARIMA warnings
data = data.asfreq('B')

# Infer frequency from index if possible
if data.index.freq is None:
    data.index.freq = pd.infer_freq(data.index)

# Fallback: explicitly set to business day frequency
if data.index.freq is None:
    data.index.freq = 'B'

# Clean up 'Close' column
data["Close"] = data["Close"].replace(',', '', regex=True)          # Remove commas
data["Close"] = pd.to_numeric(data["Close"], errors='coerce')       # Convert to numeric
data["Close"].replace([np.inf, -np.inf], np.nan, inplace=True)      # Replace inf with NaN
data.dropna(subset=["Close"], inplace=True)                         # Drop rows with missing Close

# --------------------------------------------------------
# Step 2: Visualize the original close price
# --------------------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(data.index, data["Close"], label='Close Price')
plt.title('Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Step 3: Test for stationarity using ADF
# --------------------------------------------------------

# ADF Test on Original Series
result_original = adfuller(data["Close"])
print(f"ADF Statistic (Original): {result_original[0]:.4f}")
print(f"p-value (Original): {result_original[1]:.4f}")
if result_original[1] < 0.05:
    print("Interpretation: The original series is Stationary.\n")
else:
    print("Interpretation: The original series is Non-Stationary.\n")

# First-order Differencing
data['Close_Diff'] = data['Close'].diff()

# ADF Test on Differenced Series
result_diff = adfuller(data["Close_Diff"].dropna())
print(f"ADF Statistic (Differenced): {result_diff[0]:.4f}")
print(f"p-value (Differenced): {result_diff[1]:.4f}")
if result_diff[1] < 0.05:
    print("Interpretation: The differenced series is Stationary.")
else:
    print("Interpretation: The differenced series is Non-Stationary.")

# --------------------------------------------------------
# Step 4: Visualize the differenced data
# --------------------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(data.index, data['Close_Diff'], label='Differenced Close Price', color='orange')
plt.title('Differenced Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Differenced Close')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Step 5: Plot ACF and PACF to determine ARIMA terms
# --------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ACF (q): moving average lag
plot_acf(data['Close_Diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')

# PACF (p): autoregressive lag
plot_pacf(data['Close_Diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Step 6: Split into training and testing sets
# --------------------------------------------------------

train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# --------------------------------------------------------
# Step 7: Fit ARIMA model on training data
# --------------------------------------------------------

model = ARIMA(train["Close"], order=(1, 1, 1))  # (p=1, d=1, q=1)
model_fit = model.fit()

# --------------------------------------------------------
# Step 8: Forecast future values
# --------------------------------------------------------

forecast = model_fit.forecast(steps=len(test))  # Forecast same length as test

# --------------------------------------------------------
# Step 9: Plot actual vs forecasted values
# --------------------------------------------------------

plt.figure(figsize=(12, 7))
plt.plot(train.index, train["Close"], label='Train', color='#203147')
plt.plot(test.index, test["Close"], label='Test', color='#01ef63')
plt.plot(test.index, forecast, label='Forecast', color='orange')
plt.title('Close Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# Step 10: Evaluate model performance
# --------------------------------------------------------

print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")

# Compute RMSE
forecast = forecast[:len(test)]
test_close = test["Close"][:len(forecast)]
rmse = np.sqrt(mean_squared_error(test_close, forecast))
print(f"RMSE: {rmse:.4f}")
