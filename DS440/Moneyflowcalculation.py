import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  
from io import BytesIO
import base64

def calculate_money_flow(tickers, period='5d', interval='1d'):
    """
    Calculate the money flow for a list of stocks using yfinance.
    
    Parameters:
        tickers (list): List of stock symbols.
        period (str): Time period to fetch data for (default: '5d').
        interval (str): Data interval (default: '1d').
    
    Returns:
        DataFrame: A DataFrame containing the money flow for each stock.
    """
    # Download data for all tickers at once
    data = yf.download(tickers, period=period, interval=interval)
    results = []
    
    # Loop over each ticker to process its data individually
    for ticker in tickers:
        # Check if the data has a MultiIndex (as is the case when downloading multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # Extract data for this ticker using the .xs method.
            # The MultiIndex is organized with attributes as level 0 and ticker symbols as level 1.
            ticker_data = data.xs(ticker, axis=1, level=1)
        else:
            ticker_data = data.copy()
        
        # Verify required columns exist
        required_columns = ['High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in ticker_data.columns:
                raise KeyError(f"Missing expected column: {col} for ticker: {ticker}")
        
        # Calculate Typical Price
        ticker_data['Typical Price'] = (ticker_data['High'] + ticker_data['Low'] + ticker_data['Close']) / 3
        
        # Calculate Money Flow
        ticker_data['Money Flow'] = ticker_data['Typical Price'] * ticker_data['Volume']
        
        # Add the ticker as a column for identification
        ticker_data['Ticker'] = ticker
        
        # Append the results for this ticker (reset index to make Date a column)
        results.append(ticker_data[['Ticker', 'Money Flow']].reset_index())
    
    # Combine all individual ticker results into one DataFrame
    result_df = pd.concat(results, ignore_index=True)
    return result_df

def generate_heatmap():
    tickers = ['AAPL', 'MSFT', 'TSLA','GOOGL',"AMZN"]
    money_flow_data = calculate_money_flow(tickers)
    heatmap_data = money_flow_data.pivot(index='Date', columns='Ticker', values='Money Flow')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap='coolwarm', norm=mcolors.LogNorm())
    plt.title("Stock Money Flow Heat Map")
    plt.xticks(rotation=45)
    
    # Save plot to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def handler(request):
    try:
        image_data = generate_heatmap()
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'image/png',
                'Access-Control-Allow-Origin': '*'
            },
            'body': image_data,
            'isBase64Encoded': True
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }