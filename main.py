import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------------
# Step 1: Load and clean the data
# -------------------------------
csv_path = "data/future-gc00-daily-prices.csv"
data = pd.read_csv(
    csv_path,
    parse_dates=["Date"],
    dayfirst=False,
    index_col="Date"
)
data.sort_index(inplace=True)
data = data.asfreq('B')

if data.index.freq is None:
    data.index.freq = pd.infer_freq(data.index)
if data.index.freq is None:
    data.index.freq = 'B'

data["Close"] = data["Close"].replace(',', '', regex=True)
data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
data["Close"].replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(subset=["Close"], inplace=True)

# -------------------------------
# Step 2: Prepare for GUI
# -------------------------------
root = tk.Tk()
root.title("Gold Price ARIMA Analysis")
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

def add_plot_tab(fig, title):
    """Embed a Matplotlib figure into a new Tkinter tab."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=title)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def add_text_tab(notebook, title, content):
    """Add a scrollable read-only Text widget with content to a new tab."""
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=title)

    text_box = tk.Text(frame, wrap='word', height=15)
    scrollbar = ttk.Scrollbar(frame, orient='vertical', command=text_box.yview)
    text_box.configure(yscrollcommand=scrollbar.set)

    text_box.insert('1.0', content)
    text_box.config(state='disabled')  # read-only

    text_box.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

# -------------------------------
# Step 3: Original Close Price Plot
# -------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(data.index, data["Close"], label='Close Price')
ax1.set_title('Gold Close Price Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price')
ax1.grid()
ax1.legend()
add_plot_tab(fig1, "Original Series")

# -------------------------------
# Step 4: Stationarity Test
# -------------------------------
result_original = adfuller(data["Close"])
result_original_stat = f"ADF Statistic (Original): {result_original[0]:.4f}\n" \
                       f"p-value (Original): {result_original[1]:.4f}\n" \
                       + ("Stationary" if result_original[1] < 0.05 else "Non-Stationary") + "\n\n"

data['Close_Diff'] = data['Close'].diff()
result_diff = adfuller(data["Close_Diff"].dropna())
result_diff_stat = f"ADF Statistic (Differenced): {result_diff[0]:.4f}\n" \
                   f"p-value (Differenced): {result_diff[1]:.4f}\n" \
                   + ("Stationary" if result_diff[1] < 0.05 else "Non-Stationary")

# -------------------------------
# Step 5: Differenced Plot
# -------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(data.index, data['Close_Diff'], label='Differenced Close Price', color='orange')
ax2.set_title('Differenced Close Price Over Time')
ax2.set_xlabel('Date')
ax2.set_ylabel('Differenced Close')
ax2.grid()
ax2.legend()
add_plot_tab(fig2, "Differenced Series")

# -------------------------------
# Step 6: ACF and PACF
# -------------------------------
fig3, axes = plt.subplots(1, 2, figsize=(10, 6))
plot_acf(data['Close_Diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('ACF')
plot_pacf(data['Close_Diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('PACF')
add_plot_tab(fig3, "ACF & PACF")

# -------------------------------
# Step 7: Train-Test Split
# -------------------------------
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# -------------------------------
# Step 8: Fit ARIMA
# -------------------------------
model = ARIMA(train["Close"], order=(1, 1, 1))
model_fit = model.fit()

# -------------------------------
# Step 9: Forecast
# -------------------------------
forecast = model_fit.forecast(steps=len(test))

# -------------------------------
# Step 10: Forecast Plot
# -------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(train.index, train["Close"], label='Train', color='#203147')
ax4.plot(test.index, test["Close"], label='Test', color='#01ef63')
ax4.plot(test.index, forecast, label='Forecast Mean', color='orange')
ax4.set_title('Close Price Forecast')
ax4.set_xlabel('Date')
ax4.set_ylabel('Close Price')
ax4.legend()
add_plot_tab(fig4, "Forecast Mean")

# -------------------------------
# Step 11: Metrics
# -------------------------------
rmse = np.sqrt(mean_squared_error(test["Close"][:len(forecast)], forecast))
metrics_text = f"AIC: {model_fit.aic:.4f}\n" \
               f"BIC: {model_fit.bic:.4f}\n" \
               f"RMSE: {rmse:.4f}\n" \
               f"Forecast Mean: {forecast.mean():.4f}\n" \
               f"Training Mean: {train['Close'].mean():.4f}"

# -------------------------------
# Step 12: Add Text Tabs
# -------------------------------
add_text_tab(notebook, "Stationarity Test Results", result_original_stat + result_diff_stat)
add_text_tab(notebook, "Model Metrics", metrics_text)

# -------------------------------
# Step 13: Close Handler
# -------------------------------
def on_close():
    root.destroy()
    sys.exit()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
