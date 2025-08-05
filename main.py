import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error


# Path to future-gc00-daily-prices.csv
csv_path = "data/future-gc00-daily-prices.csv"

# Reading the csv, parsing the "Date" column as datetime and using it as the index
data = pd.read_csv(
    csv_path,
    parse_dates=["Date"],   # parse_dates assuming column name is "Date"
    dayfirst=False,         # Setting to False since dates are set 1/12/2020
    index_col="Date"        # Setting the Date Column as the DateFrame's index
)

# Sort data set by date
data.sort_index(inplace=True)

# Step 1: Remove commas (e.g., "1,200" → "1200")
data["Close"] = data["Close"].replace(',', '', regex=True)

# Step 2: Convert to numeric (coerce invalid values to NaN)
data["Close"] = pd.to_numeric(data["Close"], errors='coerce')

# Step 3: Replace infinite values with NaN
data["Close"].replace([np.inf, -np.inf], np.nan, inplace=True)

# Step 4: Drop any rows where "Close" is still NaN
data.dropna(subset=["Close"], inplace=True)

# plotting the original Close
plt.figure(figsize=(10,4))
plt.grid()

# Plot the 'Close' column over time
plt.plot(data.index, data["Close"], label='Close Price')

# Add a title and axis labels
plt.title('Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')

# Show the legend
plt.legend()

# Display the plot
plt.show()



# ------------------------------------------------------------------------

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



# Plotting the differenced Close price
plt.figure(figsize=(10, 4))
plt.plot(data.index, data['Close_Diff'], label='Differenced Close Price', color='orange')
plt.title('Differenced Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Differenced Close')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Inspection Time
#print(data.head())