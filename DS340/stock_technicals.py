import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table


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

def plot_rsi(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"No data available for {stock_symbol} in the given date range.")
        return
    
    stock_data['RSI'] = calculate_rsi(stock_data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['RSI'], label=f'RSI ({stock_symbol})', color='blue')
    plt.axhline(70, linestyle='--', color='red', alpha=0.7, label='Overbought (70)')
    plt.axhline(30, linestyle='--', color='green', alpha=0.7, label='Oversold (30)')
    plt.ylim(0, 100)
    plt.title(f'Relative Strength Index (RSI) - {stock_symbol}')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def plot_macd(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"No data available for {stock_symbol} in the given date range.")
        return
    
    stock_data['MACD'], stock_data['Signal'] = calculate_macd(stock_data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['MACD'], label='MACD', color='blue')
    plt.plot(stock_data.index, stock_data['Signal'], label='Signal Line', color='red', linestyle='--')
    plt.axhline(0, linestyle='-', color='black', alpha=0.7)
    plt.title(f'Moving Average Convergence Divergence (MACD) - {stock_symbol}')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()
    plt.show()



def plot_stock_price_and_volume(stock_symbol, start_date, end_date):
    # Download historical data for the stock
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    if stock_data.empty:
        print(f"No data available for {stock_symbol} in the given date range.")
        return
    
    # Create a figure with two subplots: one for price and one for volume
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the stock price on the first axis (ax1)
    ax1.plot(stock_data.index, stock_data['Close'], color='blue', label='Closing Price')
    ax1.set_title(f'{stock_symbol} Price and Volume')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    

    # Show the plot
    fig.tight_layout()
    plt.show()

# Example usage:
plot_stock_price_and_volume('AAPL', '2024-01-01', '2025-01-01')

# Example usage:
plot_rsi('AAPL', '2024-01-01', '2025-01-01')
plot_macd('AAPL', '2024-01-01', '2025-01-01')

