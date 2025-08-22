#!/usr/bin/env python3
"""
Stock Price Prediction using LSTM
Main execution script
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import StockDataCollector
from feature_engineering import FeatureEngineer
from model import LSTMStockPredictor

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_prediction.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main execution function"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Stock Price Prediction Pipeline")
    
    # Configuration
    SYMBOL = 'AAPL'  # Change this to any stock symbol
    SEQUENCE_LENGTH = 60
    EPOCHS = 50
    BATCH_SIZE = 32
    TEST_SIZE = 0.2
    
    try:
        # Step 1: Data Collection
        logger.info("Step 1: Collecting stock data")
        collector = StockDataCollector()
        
        # Fetch stock data
        stock_data = collector.fetch_stock_data(
            symbol=SYMBOL,
            period="5y",  # 5 years of data
            interval="1d"
        )
        
        # Validate data
        if not collector.validate_data(stock_data):
            raise ValueError("Data validation failed")
        
        logger.info(f"Collected {len(stock_data)} data points for {SYMBOL}")
        
        # Save raw data
        collector.save_data(stock_data, f'{SYMBOL}_raw_data.csv')
        
        # Step 2: Feature Engineering
        logger.info("Step 2: Feature engineering")
        feature_engineer = FeatureEngineer()
        
        # Prepare data for LSTM
        X_train, X_test, y_train, y_test, scaler = feature_engineer.prepare_data_for_lstm(
            stock_data,
            sequence_length=SEQUENCE_LENGTH,
            test_size=TEST_SIZE,
            target_col='close'
        )
        
        logger.info(f"Feature engineering completed:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Features per sample: {X_train.shape[2]}")
        
        # Step 3: Model Building and Training
        logger.info("Step 3: Building and training LSTM model")
        predictor = LSTMStockPredictor()
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        predictor.build_model(
            input_shape=input_shape,
            lstm_units=[100, 50, 25],  # 3-layer LSTM
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Split training data for validation
        val_split = int(0.8 * len(X_train))
        X_train_split = X_train[:val_split]
        X_val = X_train[val_split:]
        y_train_split = y_train[:val_split]
        y_val = y_train[val_split:]
        
        # Train model
        history = predictor.train(
            X_train_split, y_train_split,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # Step 4: Model Evaluation
        logger.info("Step 4: Evaluating model")
        
        # Evaluate on test data
        metrics = predictor.evaluate(X_test, y_test)
        
        # Make predictions
        train_predictions = predictor.predict(X_train)
        test_predictions = predictor.predict(X_test)
        
        # Print evaluation metrics
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Step 5: Visualization
        logger.info("Step 5: Creating visualizations")
        
        # Plot training history
        predictor.plot_training_history()
        
        # Plot predictions vs actual
        predictor.plot_predictions(y_test, test_predictions, 
                                 f"{SYMBOL} - Test Predictions vs Actual")
        
        # Create a comprehensive prediction plot
        plt.figure(figsize=(15, 8))
        
        # Combine all data for plotting
        total_samples = len(y_train) + len(y_test)
        train_plot = np.empty(total_samples)
        train_plot[:] = np.nan
        train_plot[:len(train_predictions)] = train_predictions
        
        test_plot = np.empty(total_samples)
        test_plot[:] = np.nan
        test_plot[len(y_train):] = test_predictions
        
        actual_plot = np.concatenate([y_train, y_test])
        
        # Plot
        plt.plot(actual_plot, label='Actual Price', color='blue', alpha=0.7)
        plt.plot(train_plot, label='Training Predictions', color='orange', alpha=0.7)
        plt.plot(test_plot, label='Test Predictions', color='red', alpha=0.8)
        
        # Add vertical line to separate train/test
        plt.axvline(x=len(y_train), color='green', linestyle='--', 
                   label='Train/Test Split', alpha=0.7)
        
        plt.title(f'{SYMBOL} Stock Price Prediction - LSTM Model')
        plt.xlabel('Time (Days)')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Step 6: Future Predictions
        logger.info("Step 6: Making future predictions")
        
        # Get the last sequence for future predictions
        last_sequence = X_test[-1]  # Use the last test sequence
        
        # Predict next 5 days
        future_predictions = predictor.predict_future(last_sequence, steps=5)
        
        print("\n" + "="*50)
        print("FUTURE PRICE PREDICTIONS (Next 5 Days)")
        print("="*50)
        last_price = y_test[-1]
        print(f"Current Price: ${last_price:.2f}")
        print("-" * 30)
        
        for i, pred in enumerate(future_predictions, 1):
            change = pred - last_price
            change_pct = (change / last_price) * 100
            print(f"Day {i}: ${pred:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
            last_price = pred
        
        # Step 7: Save Model and Results
        logger.info("Step 7: Saving model and results")
        
        # Save the trained model
        model_path = f'models/{SYMBOL}_lstm_model.h5'
        os.makedirs('models', exist_ok=True)
        predictor.save_model(model_path)
        
        # Save predictions
        results_df = pd.DataFrame({
            'actual_train': np.concatenate([y_train, [np.nan] * len(y_test)]),
            'predicted_train': np.concatenate([train_predictions, [np.nan] * len(y_test)]),
            'actual_test': np.concatenate([[np.nan] * len(y_train), y_test]),
            'predicted_test': np.concatenate([[np.nan] * len(y_train), test_predictions])
        })
        
        results_df.to_csv(f'{SYMBOL}_predictions.csv', index=False)
        
        # Save model configuration and metrics
        config = {
            'symbol': SYMBOL,
            'sequence_length': SEQUENCE_LENGTH,
            'test_size': TEST_SIZE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'model_architecture': [100, 50, 25],
            'metrics': metrics,
            'future_predictions': future_predictions.tolist()
        }
        
        import json
        with open(f'{SYMBOL}_model_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("Pipeline completed successfully!")
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved to: {model_path}")
        print(f"Predictions saved to: {SYMBOL}_predictions.csv")
        print(f"Configuration saved to: {SYMBOL}_model_config.json")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

def predict_single_stock(symbol, days_ahead=5):
    """
    Simplified function to predict a single stock
    
    Args:
        symbol (str): Stock symbol
        days_ahead (int): Number of days to predict
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Quick prediction for {symbol}")
    
    try:
        # Collect data
        collector = StockDataCollector()
        data = collector.fetch_stock_data(symbol, period="2y")
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test, scaler = engineer.prepare_data_for_lstm(
            data, sequence_length=60, test_size=0.2
        )
        
        # Build and train model
        predictor = LSTMStockPredictor()
        predictor.build_model((X_train.shape[1], X_train.shape[2]))
        
        predictor.train(X_train, y_train, epochs=30, verbose=0)
        
        # Evaluate
        metrics = predictor.evaluate(X_test, y_test)
        
        # Future predictions
        future_preds = predictor.predict_future(X_test[-1], steps=days_ahead)
        
        print(f"\n{symbol} - Quick Prediction Results:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"\nNext {days_ahead} days predictions:")
        for i, pred in enumerate(future_preds, 1):
            print(f"Day {i}: ${pred:.2f}")
            
        return future_preds, metrics
        
    except Exception as e:
        logger.error(f"Error predicting {symbol}: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Run main pipeline
    main()
    
    # Uncomment below for quick predictions of multiple stocks
    # stocks = ['GOOGL', 'MSFT', 'TSLA']
    # for stock in stocks:
    #     predict_single_stock(stock, days_ahead=3)
