import yfinance as yf
import pandas as pd
import torch
import math
from scipy.stats import norm

# ⚡ Define Ticker Symbol
TICKER = "AAPL"
EXPIRATION_COUNT = 3  # Number of expiration dates to fetch

# Black-Scholes Greeks Calculation with T=0 Fix
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute Black-Scholes Greeks while handling T=0 cases."""
    epsilon = 1/365  # Small constant to prevent division by zero
    T = max(T, epsilon)  # Ensure T is never exactly 0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) -
             r * K * math.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    rho = K * T * math.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)

    return delta, gamma, theta, vega, rho


# ⚡ Fetch Live Stock Price
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d", interval="1m")
    if hist.empty:
        raise ValueError(f"No stock data found for {ticker}")
    
    latest_price = hist["Close"].iloc[-1]
    return torch.tensor(latest_price, dtype=torch.float32)

# ⚡ Fetch Option Data for Multiple Expirations
def fetch_option_data(ticker, num_expirations=EXPIRATION_COUNT):
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options[:num_expirations]  # Select first `num_expirations` dates
    if not expiration_dates:
        raise ValueError(f"No options data found for {ticker}")
    
    all_options = []
    stock_price = fetch_stock_data(ticker).item()
    risk_free_rate = 0.05  # Assume 5% risk-free rate

    for expiry in expiration_dates:
        opt_chain = stock.option_chain(expiry)
        T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365  # Time to expiration in years
        
        for option_df, option_type in [(opt_chain.calls, "call"), (opt_chain.puts, "put")]:
            option_df["option_type"] = option_type
            option_df = option_df[["strike", "impliedVolatility", "option_type"]].dropna()
            option_df["impliedVolatility"] /= 100  # Convert to decimal

            # Compute Greeks
            greeks = option_df.apply(
                lambda row: calculate_greeks(
                    S=stock_price,
                    K=row["strike"],
                    T=T,
                    r=risk_free_rate,
                    sigma=row["impliedVolatility"],
                    option_type=row["option_type"]
                ), axis=1
            )

            option_df[["delta", "gamma", "theta", "vega", "rho"]] = pd.DataFrame(greeks.tolist(), index=option_df.index)
            option_df["expiration_days"] = T * 365  # Store expiration in days

            all_options.append(option_df)

    options = pd.concat(all_options)

    # Convert to PyTorch tensor
    option_tensor = torch.tensor(
        options[["strike", "impliedVolatility", "delta", "gamma", "theta", "vega", "rho", "expiration_days"]].values,
        dtype=torch.float32
    )

    return option_tensor

# ⚡ Combine Stock & Multi-Expiration Option Data for GAN Training
def create_market_data(ticker):
    stock_price = fetch_stock_data(ticker)
    options = fetch_option_data(ticker)
    
    # Add stock price column
    stock_prices = stock_price.expand(options.shape[0], 1)

    # Concatenate stock price with options
    market_data = torch.cat([stock_prices, options], dim=1)

    # Save as a PyTorch file
    torch.save(market_data, "market_data.pt")
    print(f"✅ Market data saved as market_data.pt | Shape: {market_data.shape}")

    return market_data

# ⚡ Fetch & Save Data
market_data = create_market_data(TICKER)
print(market_data.shape)  # Example: (300, 9) where 300 options and 9 features (stock + option data)
