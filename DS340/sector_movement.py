import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def get_sector_stocks(sector):
    # Get a list of stock tickers in the sector (for demonstration, we use a predefined list)
    # Ideally, you'd fetch this from a reliable source such as an API or database
    sector_stocks = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL'],
        'Healthcare': ['PFE', 'JNJ', 'MRK'],
        'Financials': ['JPM', 'GS', 'WFC'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'NKE'],
        'Energy': ['XOM', 'CVX', 'COP']
    }
    
    return sector_stocks.get(sector, [])




###### ISSUE
def plot_sector_price_movement(sector):
    tickers = get_sector_stocks(sector)
    
    if not tickers:
        print(f"No stock tickers found for the {sector} sector.")
        return
    
    # Download historical stock data for the tickers in the sector
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, start='2020-01-01', end='2025-01-01')
        stock_data[ticker] = data['Close']
    
    # Combine the data into one DataFrame
    ###### ISSUE 
    sector_df = pd.DataFrame(stock_data)
    
    # Calculate the average price movement (mean of all stocks in the sector)
    sector_df['Average'] = sector_df.mean(axis=1)
    
    # Plot the price movement of the sector
    plt.figure(figsize=(12, 6))
    plt.plot(sector_df.index, sector_df['Average'], label=f'{sector} Sector', color='blue')
    plt.title(f'Price Movement of the {sector} Sector')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
plot_sector_price_movement('Technology')
