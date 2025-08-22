from flask import Flask, request, jsonify, render_template_string
import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_collection import StockDataCollector
    from feature_engineering import FeatureEngineer
    from model import LSTMStockPredictor
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

app = Flask(__name__)
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>🚀 AI Stock Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container { 
            background: white; 
            padding: 40px; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            max-width: 600px;
            width: 90%;
        }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .input-group { 
            display: flex; 
            gap: 10px; 
            margin-bottom: 20px; 
            flex-wrap: wrap;
        }
        input { 
            flex: 1;
            min-width: 120px;
            padding: 15px; 
            border: 2px solid #e1e5e9; 
            border-radius: 10px; 
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus { 
            outline: none; 
            border-color: #667eea; 
        }
        button { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border: none; 
            padding: 15px 30px; 
            border-radius: 10px; 
            font-size: 16px; 
            cursor: pointer;
            transition: transform 0.2s;
            white-space: nowrap;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { 
            background: #ccc; 
            cursor: not-allowed; 
            transform: none;
        }
        .loading { 
            text-align: center; 
            color: #667eea; 
            margin: 20px 0;
            display: none;
        }
        .results { 
            margin-top: 20px; 
            padding: 20px; 
            background: #f8f9fa; 
            border-radius: 10px;
            display: none;
        }
        .error { 
            color: #e74c3c; 
            text-align: center; 
            margin: 20px 0;
            display: none;
        }
        .prediction-table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 15px;
            border-radius: 8px;
            overflow: hidden;
        }
        .prediction-table th { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 12px; 
            text-align: center;
        }
        .prediction-table td { 
            padding: 12px; 
            text-align: center; 
            border-bottom: 1px solid #e1e5e9;
        }
        .positive { color: #27ae60; font-weight: bold; }
        .negative { color: #e74c3c; font-weight: bold; }
        .current-price { 
            font-size: 24px; 
            font-weight: bold; 
            color: #2c3e50; 
            text-align: center; 
            margin: 15px 0;
        }
        .metrics { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
            gap: 10px; 
            margin: 15px 0;
        }
        .metric { 
            text-align: center; 
            padding: 10px; 
            background: white; 
            border-radius: 8px;
        }
        .disclaimer { 
            font-size: 12px; 
            color: #7f8c8d; 
            text-align: center; 
            margin-top: 20px; 
            font-style: italic;
        }
        @media (max-width: 600px) {
            .container { padding: 20px; }
            .input-group { flex-direction: column; }
            input, button { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 AI Stock Price Predictor</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            Get AI-powered predictions for any stock symbol
        </p>
        
        <div class="input-group">
            <input type="text" id="symbol" placeholder="Stock Symbol (e.g., AAPL)" value="AAPL">
            <input type="number" id="days" placeholder="Days" value="5" min="1" max="10">
            <button onclick="predictStock()" id="predictBtn">🔮 Predict</button>
        </div>
        
        <div id="loading" class="loading">
            <p>🤖 Training AI model...</p>
            <p>Please wait 30-60 seconds</p>
        </div>
        
        <div id="results" class="results">
            <h3 style="text-align: center; margin-bottom: 20px;">
                📈 Predictions for <span id="resultSymbol"></span>
            </h3>
            
            <div class="current-price">
                Current: $<span id="currentPrice"></span>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div style="font-weight: bold;">RMSE</div>
                    <div id="rmse">-</div>
                </div>
                <div class="metric">
                    <div style="font-weight: bold;">MAPE</div>
                    <div id="mape">-</div>%
                </div>
                <div class="metric">
                    <div style="font-weight: bold;">R²</div>
                    <div id="r2">-</div>
                </div>
            </div>
            
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Day</th>
                        <th>Predicted Price</th>
                        <th>Change</th>
                        <th>Change %</th>
                    </tr>
                </thead>
                <tbody id="predictions"></tbody>
            </table>
            
            <div class="disclaimer">
                ⚠️ This is for educational purposes only. Not financial advice.
            </div>
        </div>
        
        <div id="error" class="error"></div>
    </div>

    <script>
        async function predictStock() {
            const symbol = document.getElementById('symbol').value.trim().toUpperCase();
            const days = parseInt(document.getElementById('days').value);
            const predictBtn = document.getElementById('predictBtn');
            
            if (!symbol) {
                alert('Please enter a stock symbol');
                return;
            }
            
            // Show loading state
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            predictBtn.disabled = true;
            predictBtn.textContent = '🔄 Predicting...';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol: symbol, days: days})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = '❌ ' + data.error;
                    document.getElementById('error').style.display = 'block';
                } else {
                    displayResults(data);
                }
            } catch (error) {
                document.getElementById('error').textContent = '❌ Error: ' + error.message;
                document.getElementById('error').style.display = 'block';
            }
            
            // Reset loading state
            document.getElementById('loading').style.display = 'none';
            predictBtn.disabled = false;
            predictBtn.textContent = '🔮 Predict';
        }
        
        function displayResults(data) {
            document.getElementById('resultSymbol').textContent = data.symbol;
            document.getElementById('currentPrice').textContent = data.current_price;
            document.getElementById('rmse').textContent = data.metrics.rmse;
            document.getElementById('mape').textContent = data.metrics.mape;
            document.getElementById('r2').textContent = data.metrics.r2;
            
            let predictionsHtml = '';
            data.predictions.forEach(pred => {
                const changeClass = pred.change >= 0 ? 'positive' : 'negative';
                const changeIcon = pred.change >= 0 ? '📈' : '📉';
                predictionsHtml += `<tr>
                    <td>Day ${pred.day}</td>
                    <td>$${pred.price}</td>
                    <td class="${changeClass}">${changeIcon} $${pred.change >= 0 ? '+' : ''}${pred.change}</td>
                    <td class="${changeClass}">${pred.change_percent >= 0 ? '+' : ''}${pred.change_percent}%</td>
                </tr>`;
            });
            
            document.getElementById('predictions').innerHTML = predictionsHtml;
            document.getElementById('results').style.display = 'block';
        }
        
        // Allow Enter key to trigger prediction
        document.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                predictStock();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        days = min(int(data.get('days', 5)), 10)  # Limit to 10 days
        
        logger.info(f"Predicting {symbol} for {days} days")
        
        # Collect data
        collector = StockDataCollector()
        stock_data = collector.fetch_stock_data(symbol, period="1y", interval="1d")
        
        if not collector.validate_data(stock_data):
            return jsonify({'error': f'Invalid stock symbol "{symbol}" or insufficient data'})
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test, scaler = engineer.prepare_data_for_lstm(
            stock_data, sequence_length=30, test_size=0.2
        )
        
        # Build and train model (lightweight for deployment)
        predictor = LSTMStockPredictor()
        predictor.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=[32, 16],  # Smaller model for faster training
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Train with fewer epochs for speed
        predictor.train(X_train, y_train, epochs=15, verbose=0)
        
        # Evaluate
        metrics = predictor.evaluate(X_test, y_test)
        
        # Future predictions
        future_preds = predictor.predict_future(X_test[-1], steps=days)
        
        # Format results
        current_price = float(y_test[-1])
        predictions = []
        
        for i, pred in enumerate(future_preds, 1):
            change = float(pred - current_price)
            change_pct = (change / current_price) * 100
            predictions.append({
                'day': i,
                'price': round(float(pred), 2),
                'change': round(change, 2),
                'change_percent': round(change_pct, 2)
            })
            current_price = pred
        
        return jsonify({
            'symbol': symbol,
            'current_price': round(float(y_test[-1]), 2),
            'predictions': predictions,
            'metrics': {
                'rmse': round(metrics['rmse'], 4),
                'mape': round(metrics['mape'], 2),
                'r2': round(metrics['r2'], 4)
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Stock Predictor API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
