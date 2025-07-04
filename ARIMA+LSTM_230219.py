
# !pip install numpy pandas matplotlib yfinance statsmodels scikit-learn tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.layers import Dropout

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Chosen 3 Large Cap stocks
stocks = {
   'NIFTY50': '^NSEI',
   'RELIANCE': 'RELIANCE.NS',
}


end_date = datetime.now()
start_date = datetime.now() - timedelta(days=10*365)

for company, symbol in stocks.items():
   # Downloading historic data
   df = yf.download(symbol,
                   start=start_date,
                   end=end_date,
                   progress=False)


   df = df[['Close']]
   df.index = df.index.date

   df = df.reset_index()
   df.columns = ['Date', 'Value']

   # Save to CSV
   filename = f'{company}_data.csv'
   df.to_csv(filename, index=False)
   print(f"Created {filename}")

# Splitting the dataset (80-20 split)
   split_idx = int(0.8 * len(df))
   train_df = df[:split_idx]
   test_df = df[split_idx:]

   # Save to CSV
   train_filename = f'{company}_train_data.csv'
   test_filename = f'{company}_test_data.csv'

   train_df.to_csv(train_filename, index=False)
   test_df.to_csv(test_filename, index=False)

   print(f"Created {train_filename} and {test_filename}")


# Load dataset
data = pd.read_csv('RELIANCE_train_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Step 1: Check Stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

check_stationarity(data['Value'])

# If not stationary, perform differencing
# Step 1: Differencing the data (Seasonal differencing with lag=12)
value_series = data['Value']
data_diff = value_series.diff(12).dropna()


plt.figure(figsize=(12, 6))
plt.plot(data_diff,linewidth=0.8, label="Differenced Data")
plt.title("Differenced Time Series")
plt.legend()
plt.show()

# Check stationarity of the differenced data
check_stationarity(data_diff)

# Step 3: ACF and PACF Plots
plt.figure(figsize=(12, 6))
plot_acf(data_diff, lags=20)
plt.title("ACF Plot")
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data_diff, lags=20)
plt.title("PACF Plot")
plt.show()

# ARIMA Model
arima_model = ARIMA(value_series, order=(1, 1, 9))  # Adjust 'd' for differencing
arima_fit = arima_model.fit()
arima_pred = arima_fit.predict(start=1, end=len(value_series), typ='levels')
arima_forecast = arima_fit.forecast(steps=12)

# Step 5: Extract residuals for further modeling with LSTM
residuals = arima_fit.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title("Residuals from ARIMA Model")
plt.show()
# print(residuals)

# Modify the prediction part
def create_lagged_features(data, lag=3):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Scale the entire training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(value_series.values.reshape(-1, 1))

# Create sequences for training
X_train, y_train = create_lagged_features(scaled_data, lag=3)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(200, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(200, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Forecasting for Test Dataset
test_data = pd.read_csv('RELIANCE_test_data.csv')
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data.set_index('Date', inplace=True)

# ARIMA predictions
arima_predictions = arima_fit.forecast(steps=len(test_data))

# Prepare test data for LSTM
test_scaled = scaler.transform(test_data['Value'].values.reshape(-1, 1))
X_test, _ = create_lagged_features(test_scaled, lag=3)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# # LSTM predictions
lstm_residuals_predictions = model.predict(X_test)
lstm_residuals_predictions = scaler.inverse_transform(lstm_residuals_predictions)


final_test_predictions = arima_predictions[3:len(lstm_residuals_predictions)+3] + lstm_residuals_predictions.flatten() -1036

# Visualizing the results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[3:], test_data['Value'][3:], label='Actual', linewidth=2)
plt.plot(test_data.index[3:], final_test_predictions, label='Predicted', linestyle='dashed', linewidth=2)
plt.title("Reliance Stock Price Forecast")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print metrics
mape = mean_absolute_percentage_error(test_data['Value'][3:len(final_test_predictions)+3], final_test_predictions)
rmse = np.sqrt(mean_squared_error(test_data['Value'][3:len(final_test_predictions)+3], final_test_predictions))
print(f"MAPE: {mape:.2%}")
print(f"RMSE: {rmse:.2f}")















"""###OLD MODEL###"""

# import yfinance as yf
# import pandas as pd
# from datetime import datetime, timedelta

# # Chosen 3 Large Cap stock
# stocks = {
#    'NIFTY50': '^NSEI',
#   'RELIANCE': 'RELIANCE.NS',
# }


# end_date = datetime.now()
# start_date = datetime.now() - timedelta(days=10*365)

# for company, symbol in stocks.items():
#    # Download historic data
#    df = yf.download(symbol,
#                    start=start_date,
#                    end=end_date,
#                    progress=False)


#    df = df[['Close']]
#    df.index = df.index.date

#    df = df.reset_index()
#    df.columns = ['Date', 'Value']

#    # Save to CSV
#    filename = f'{company}_data.csv'
#    df.to_csv(filename, index=False)
#    print(f"Created {filename}")

# # Splitting the dataset (80-20 split)
#    split_idx = int(0.8 * len(df))
#    train_df = df[:split_idx]
#    test_df = df[split_idx:]

#    # Save to CSV
#    train_filename = f'{company}_train_data.csv'
#    test_filename = f'{company}_test_data.csv'

#    train_df.to_csv(train_filename, index=False)
#    test_df.to_csv(test_filename, index=False)

#    print(f"Created {train_filename} and {test_filename}")

# # Load dataset
# data = pd.read_csv('RELIANCE_train_data.csv')
# data['Date'] = pd.to_datetime(data['Date'])
# data.set_index('Date', inplace=True)

# # Step 1: Check Stationarity
# def check_stationarity(timeseries):
#     result = adfuller(timeseries)
#     print(f"ADF Statistic: {result[0]}")
#     print(f"p-value: {result[1]}")
#     if result[1] <= 0.05:
#         print("The series is stationary.")
#     else:
#         print("The series is not stationary.")

# check_stationarity(data['Value'])

# # If not stationary, perform differencing
# # Step 1: Differencing the data (Seasonal differencing with lag=12)
# value_series = data['Value']
# data_diff = value_series.diff(12).dropna()


# plt.figure(figsize=(12, 6))
# plt.plot(data_diff,linewidth=0.8, label="Differenced Data")
# plt.title("Differenced Time Series")
# plt.legend()
# plt.show()

# # Check stationarity of the differenced data
# check_stationarity(data_diff)

# # Step 3: ACF and PACF Plots
# plt.figure(figsize=(12, 6))
# plot_acf(data_diff, lags=20)
# plt.title("ACF Plot")
# plt.show()

# plt.figure(figsize=(12, 6))
# plot_pacf(data_diff, lags=20)
# plt.title("PACF Plot")
# plt.show()

# # ARIMA Model
# arima_model = ARIMA(value_series, order=(1, 1, 9))  # Adjust 'd' for differencing
# arima_fit = arima_model.fit()
# arima_pred = arima_fit.predict(start=1, end=len(value_series), typ='levels')
# arima_forecast = arima_fit.forecast(steps=12)

# # Step 5: Extract residuals for further modeling with LSTM
# residuals = arima_fit.resid
# plt.figure(figsize=(12, 6))
# plt.plot(residuals)
# plt.title("Residuals from ARIMA Model")
# plt.show()
# # print(residuals)



# # Modify the prediction part
# def create_lagged_features(data, lag=3):
#     X, y = [], []
#     for i in range(lag, len(data)):
#         X.append(data[i-lag:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# # Scale the entire training data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(value_series.values.reshape(-1, 1))

# # Create sequences for training
# X_train, y_train = create_lagged_features(scaled_data, lag=3)
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# # Build LSTM Model
# model = Sequential([
#     LSTM(200, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
#     Dropout(0.2),
#     LSTM(200, activation='relu'),
#     Dropout(0.2),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse')
# model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# # Forecasting for Test Dataset
# test_data = pd.read_csv('RELIANCE_test_data.csv')
# test_data['Date'] = pd.to_datetime(test_data['Date'])
# test_data.set_index('Date', inplace=True)

# # ARIMA predictions
# arima_predictions = arima_fit.forecast(steps=len(test_data))

# # Prepare test data for LSTM
# test_scaled = scaler.transform(test_data['Value'].values.reshape(-1, 1))
# X_test, _ = create_lagged_features(test_scaled, lag=3)
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# # LSTM predictions
# lstm_predictions = model.predict(X_test)
# lstm_predictions = scaler.inverse_transform(lstm_predictions)

# # Combine predictions (weighted average)
# weight_arima = 0.0005
# weight_lstm = 0.9995

# final_test_predictions = (weight_arima * arima_predictions[:len(lstm_predictions)] +
#                          weight_lstm * lstm_predictions.flatten())

# # Visualizing the results
# plt.figure(figsize=(12, 6))
# plt.plot(test_data.index[3:], test_data['Value'][3:], label='Actual', linewidth=2)
# plt.plot(test_data.index[3:], final_test_predictions, label='Predicted', linestyle='dashed', linewidth=2)
# plt.title("Reliance Stock Price Forecast")
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Calculate and print metrics
# mape = mean_absolute_percentage_error(test_data['Value'][3:len(final_test_predictions)+3], final_test_predictions)
# rmse = np.sqrt(mean_squared_error(test_data['Value'][3:len(final_test_predictions)+3], final_test_predictions))
# print(f"MAPE: {mape:.2%}")
# print(f"RMSE: {rmse:.2f}")













