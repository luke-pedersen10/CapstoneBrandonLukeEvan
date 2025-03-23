import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


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

# Example usage:
plot_rsi('AAPL', '2024-01-01', '2025-01-01')