from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def plot_graph(values, full_data, extra_data=None, extra_dataset=None):
    plt.figure(figsize=(15, 6))
    plt.plot(values, 'orange', linewidth=2, label='Moving Average')
    plt.plot(full_data['Close'], 'b', linewidth=1, alpha=0.7, label='Close Price')
    
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset, 'g', linewidth=2, label='Extra Data')
        
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save plot to bytes buffer
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', 'GOOG')
        
        # Download stock data
        end = datetime.now()
        start = datetime(end.year - 20, end.month, end.day)
        stock_data = yf.download(stock, start, end)
        
        # Check if stock_data is empty
        if stock_data is None or stock_data.empty:
            return render_template('index.html', error=f"No data found for symbol '{stock}'. Please try another.")
        
        # Handle multi-index columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)
        
        # Create moving averages
        stock_data['MA_250'] = stock_data['Close'].rolling(250).mean()
        stock_data['MA_200'] = stock_data['Close'].rolling(200).mean()
        stock_data['MA_100'] = stock_data['Close'].rolling(100).mean()
        
        # Generate plots
        plot_250 = plot_graph(stock_data['MA_250'], stock_data)
        plot_200 = plot_graph(stock_data['MA_200'], stock_data)
        plot_100 = plot_graph(stock_data['MA_100'], stock_data)
        plot_combined = plot_graph(stock_data['MA_100'], stock_data, True, stock_data['MA_250'])
        
        # Prepare data for prediction
        splitting_len = int(len(stock_data) * 0.7)
        x_test = pd.DataFrame(stock_data['Close'][splitting_len:])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']])
        
        x_data = []
        y_data = []
        
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])
            
        x_data, y_data = np.array(x_data), np.array(y_data)
        
        # Load model and make predictions
        model = load_model("models/Latest_stock_price_model.keras", compile=False)
        predictions = model.predict(x_data)
        
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)
        
        # Create prediction dataframe
        ploting_data = pd.DataFrame({
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        }, index=stock_data.index[splitting_len+100:])
        
        # Generate prediction plot
        plt.figure(figsize=(15, 6))
        plt.plot(stock_data['Close'][:splitting_len+100], 'b', label='Data not used')
        plt.plot(ploting_data['original_test_data'], 'g', label='Original Test data')
        plt.plot(ploting_data['predictions'], 'r', label='Predicted Test data')
        plt.legend()
        plt.grid(True)
        
        pred_img = BytesIO()
        plt.savefig(pred_img, format='png')
        plt.close()
        pred_img.seek(0)
        pred_plot = base64.b64encode(pred_img.getvalue()).decode('utf-8')
        
        # Convert data to HTML tables
        stock_table = stock_data.tail().to_html(classes='data-table')
        prediction_table = ploting_data.tail().to_html(classes='data-table')
        
        return render_template('results.html', 
                               stock=stock,
                               stock_table=stock_table,
                               prediction_table=prediction_table,
                               plot_250=plot_250,
                               plot_200=plot_200,
                               plot_100=plot_100,
                               plot_combined=plot_combined,
                               pred_plot=pred_plot)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


    