import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import io

app = Flask(__name__)

def plot_stock_price_and_volume(stock_symbol, start_date, end_date):
    # Download historical data for the stock
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    if stock_data.empty:
        return None
    
    # Create a figure with two subplots: one for price and one for volume
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the stock price on the first axis (ax1)
    ax1.plot(stock_data.index, stock_data['Close'], color='blue', label='Closing Price')
    ax1.set_title(f'{stock_symbol} Price and Volume')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Closing Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')


    # Save the plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)  # Rewind the image pointer to the start
    return img_bytes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    # Get the form data
    stock_symbol = request.form['stock_symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
    # Generate the plot
    img_bytes = plot_stock_price_and_volume(stock_symbol, start_date, end_date)
    
    if img_bytes is None:
        return "No data available for the given stock symbol and date range."
    
    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
