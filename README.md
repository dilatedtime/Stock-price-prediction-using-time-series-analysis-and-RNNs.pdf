# Stock Price Forecasting with Time Series Models and RNNs

This repository collects notebooks and a Python script from a stock-forecasting study. The material starts with regression, neural-network, and time-series foundations, then moves into ARIMA, SARIMA, recurrent neural networks, LSTM, GRU, and a combined ARIMA-LSTM experiment.

## Contents

| File | Focus |
| --- | --- |
| `introduction_to_neural_networks.ipynb` | Neurons, feed-forward networks, gradient descent, and backpropagation in PyTorch and TensorFlow |
| `intro_to_recurrent_neural_networks_lstm_gru.ipynb` | RNNs, vanishing gradients, LSTM, GRU, and sequence generation |
| `GDG_TSA.ipynb` | Time-series exploration, decomposition, and stationarity |
| `Time_Series_Models.ipynb` | ADF tests, ACF/PACF, ARIMA, and SARIMA |
| `TimeSeriesAnalysisandForecastingwithStockPrices.ipynb` | Stock data from `yfinance` and classical forecasting models |
| `AryanRaj_LinearRegression.ipynb` | Linear-regression practice |
| `CLRM_implementation_.ipynb` | Classical linear-regression assumptions and calculations |
| `ARIMA+LSTM_230219.py` | Downloads NIFTY 50 and Reliance data, trains ARIMA and LSTM models, and plots a Reliance forecast |

## Run the Python experiment

```bash
git clone https://github.com/dilatedtime/Stock-price-prediction-using-time-series-analysis-and-RNNs.pdf.git
cd Stock-price-prediction-using-time-series-analysis-and-RNNs.pdf
python -m venv .venv
python -m pip install numpy pandas matplotlib yfinance statsmodels scikit-learn tensorflow jupyter
python "ARIMA+LSTM_230219.py"
```

The script downloads roughly 10 years of daily data, writes train and test CSV files, trains an ARIMA model and a two-layer LSTM, then reports MAPE and RMSE for Reliance.

## Read before using the results

This is an experimental learning project. The combined forecast currently includes a manually chosen `-1036` adjustment, and the model settings were tuned around one saved workflow. Remove that adjustment, define a repeatable validation plan, and retune the models before comparing performance. Forecasts in this repository are not investment advice.

