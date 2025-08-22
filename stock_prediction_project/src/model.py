import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import logging
import os

class LSTMStockPredictor:
    """
    LSTM model for stock price prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.sequence_length = None
        self.logger = logging.getLogger(__name__)
        self.history = None
    
    def build_model(self, input_shape, lstm_units=[50, 50], dropout_rate=0.2, learning_rate=0.001):
        """
        Build LSTM model architecture
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            lstm_units (list): List of LSTM units for each layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        
        Returns:
            tensorflow.keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=lstm_units[0], 
                      return_sequences=len(lstm_units) > 1,
                      input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i in range(1, len(lstm_units)):
            return_sequences = i < len(lstm_units) - 1
            model.add(LSTM(units=lstm_units[i], return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        self.logger.info(f"Model built with architecture: {lstm_units}")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the LSTM model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity level
        
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                         patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                            factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('best_model.h5', save_best_only=True,
                          monitor='val_loss' if X_val is not None else 'loss')
        ]
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        self.logger.info(f"Training completed. Best loss: {min(self.history.history['loss']):.6f}")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X (np.array): Input features
        
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
        
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        self.logger.info(f"Model evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            self.logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, title="Predictions vs Actual"):
        """
        Plot predictions against actual values
        
        Args:
            y_true (np.array): Actual values
            y_pred (np.array): Predicted values
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted
        plt.subplot(1, 2, 1)
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Plot scatter
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Scatter')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'input_shape': self.model.input_shape
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        
        Args:
            filepath (str): Path to the saved model
        """
        # Load model
        self.model = load_model(filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.sequence_length = metadata.get('sequence_length')
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def predict_future(self, last_sequence, steps=5):
        """
        Predict future prices using the last sequence
        
        Args:
            last_sequence (np.array): Last sequence from the data
            steps (int): Number of steps to predict into the future
        
        Returns:
            np.array: Future predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape))
            predictions.append(pred[0, 0])
            
            # Update sequence (this is simplified - in practice, you'd need to update all features)
            # For now, we'll just append the prediction and remove the first value
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0, 0]  # Assuming close price is the first feature
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
