
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm
from ta.volume import MFIIndicator
from ta.utils import dropna
from statsmodels.tsa.seasonal import seasonal_decompose
# Device configuration

def get_stock_data(ticker, start='2024-01-01', end='2025-03-24'):
    stock = yf.download(ticker, start=start, end=end)
    return stock

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Default to neutral if not enough data
    return rsi

def data_preparation(data):
    data = dropna(data)  # Drop NA values from the dataframe
    data[['Log Price']] = np.log(data['Close'])  # Log transformation for stationarity
    data[['Log Volume']] = np.log(data['Volume'].replace(0,np.nan))  # Log transformation for volume, add 1 to avoid log(0)
    data[['Log Volume']].fillna(data[['Log Volume']].rolling(window=5, min_periods=1).mean(), inplace=True)

    data[['Log Price Diff']] = data[['Log Price']].diff()

    data['Percent Change'] = data['Close'].pct_change()  # Calculate percentage change for the close price

    data['RSI'] = calculate_rsi(data)  # Calculate RSI
    
    mfi = MFIIndicator(
        high=data['High'].squeeze(),
        low=data['Low'].squeeze(),
        close=data['Close'].squeeze(),
        volume=data['Volume'].squeeze(),
        window=14,
        fillna=True)
    data['MFI'] = mfi.money_flow_index()
    
    data['Log Volume Diff'] = data['Log Volume'].diff()  # Log volume difference for stationarity
    
    # Beta calculation (S&P500)
    market = yf.download('SPY', start=data.index[0], end=data.index[-1])['Close']
    data['Market Return'] = market.pct_change()
    data['Stock Return'] = data['Close'].pct_change()
    data['Beta'] = data['Stock Return'].rolling(30).cov(data['Market Return']) / data['Market Return'].rolling(30).var()
    
    data['PE Ratio'] = data['Close'] / data['Earnings'] if 'Earnings' in data.columns else np.nan
     # Example PE ratio calculation, ensure 'Earnings' column exists in your data

    data['MA50'] = data['Close'].rolling(50).mean()
    print(data['MA50'].shape())
    print(data['MA50'].head())
    print(data['Close'].shape())
    print(data['Close'].head())
    data['Log Diff MA50'] = np.log(data['Close']) - np.log(data['MA50'])

    data['MA200'] = data['Close'].rolling(200).mean()
    data['Log Diff MA200'] = np.log(data['Close']) - np.log(data['MA200'])

    decomposition = seasonal_decompose(data['Close'], model='additive', period=252)  # Assuming yearly seasonality
    data['Seasonality'] = decomposition.seasonal

    feature_columns = ['Log Price Diff', 'Percent Change', 'RSI', 'MFI', 'Log Volume Diff',
                       'Beta', 'PE Ratio', 'Log Diff MA50', 'Log Diff MA200', 'Seasonality']
    
    return data[feature_columns]

apple = get_stock_data('AAPL')
#print(apple['Volume'])
prepped_data = data_preparation(apple)