import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

class BaseRNNForecaster(nn.Module):
    """
    Base class for RNN-based time series forecasters.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2, bidirectional: bool = False):
        """
        Initialize base RNN forecaster.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of RNN layers
            output_size: Number of output features
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
        """
        super(BaseRNNForecaster, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.scaler = StandardScaler()
        self.is_fitted = False

    def create_sequences(self, data: np.ndarray, seq_length: int,
                        forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for training.

        Args:
            data: Time series data
            seq_length: Length of input sequences
            forecast_horizon: Number of steps to forecast

        Returns:
            Input sequences and target values
        """
        X, y = [], []
        for i in range(len(data) - seq_length - forecast_horizon + 1):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length:i + seq_length + forecast_horizon])

        return np.array(X), np.array(y)

    def prepare_data(self, data: Union[pd.Series, np.ndarray], seq_length: int,
                    forecast_horizon: int = 1, fit_scaler: bool = True
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training/inference.

        Args:
            data: Time series data
            seq_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            fit_scaler: Whether to fit the scaler

        Returns:
            Input and target tensors
        """
        if isinstance(data, pd.Series):
            data = data.values

        # Handle multivariate case
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Scale data
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)

        # Create sequences
        X, y = self.create_sequences(scaled_data, seq_length, forecast_horizon)

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return self.scaler.inverse_transform(predictions)

class RNNForecaster(BaseRNNForecaster):
    """
    Vanilla RNN for time series forecasting.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2, bidirectional: bool = False):
        super(RNNForecaster, self).__init__(input_size, hidden_size, num_layers,
                                          output_size, dropout, bidirectional)

        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                         dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional, batch_first=True)

        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Output predictions [batch_size, output_size]
        """
        rnn_out, _ = self.rnn(x)
        # Take the last output
        last_output = rnn_out[:, -1, :]
        last_output = self.dropout_layer(last_output)
        predictions = self.fc(last_output)
        return predictions

class LSTMForecaster(BaseRNNForecaster):
    """
    LSTM for time series forecasting.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2, bidirectional: bool = False):
        super(LSTMForecaster, self).__init__(input_size, hidden_size, num_layers,
                                           output_size, dropout, bidirectional)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional, batch_first=True)

        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Output predictions [batch_size, output_size]
        """
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout_layer(last_output)
        predictions = self.fc(last_output)
        return predictions

class GRUForecaster(BaseRNNForecaster):
    """
    GRU for time series forecasting.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2, bidirectional: bool = False):
        super(GRUForecaster, self).__init__(input_size, hidden_size, num_layers,
                                          output_size, dropout, bidirectional)

        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional, batch_first=True)

        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Output predictions [batch_size, output_size]
        """
        gru_out, _ = self.gru(x)
        # Take the last output
        last_output = gru_out[:, -1, :]
        last_output = self.dropout_layer(last_output)
        predictions = self.fc(last_output)
        return predictions

class RNNTrainer:
    """
    Training utilities for RNN models.
    """

    def __init__(self, model: BaseRNNForecaster, learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize trainer.

        Args:
            model: RNN model to train
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)

            # Handle different output shapes
            if batch_y.dim() == 3 and predictions.dim() == 2:
                batch_y = batch_y.squeeze(-1)

            loss = self.criterion(predictions, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)

                # Handle different output shapes
                if batch_y.dim() == 3 and predictions.dim() == 2:
                    batch_y = batch_y.squeeze(-1)

                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_data: Union[pd.Series, np.ndarray], seq_length: int,
            forecast_horizon: int = 1, epochs: int = 100, batch_size: int = 32,
            validation_split: float = 0.2, early_stopping_patience: int = 10,
            verbose: bool = True) -> Dict[str, Any]:
        """
        Fit the model to data.

        Args:
            train_data: Training time series data
            seq_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Early stopping patience
            verbose: Whether to print training progress

        Returns:
            Training history
        """
        # Prepare data
        X, y = self.model.prepare_data(train_data, seq_length, forecast_horizon)

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1}')
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.model.is_fitted = True

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }

    def predict(self, data: Union[pd.Series, np.ndarray], seq_length: int,
                steps: int = 1) -> np.ndarray:
        """
        Generate predictions.

        Args:
            data: Input data
            seq_length: Length of input sequences
            steps: Number of steps to predict

        Returns:
            Predictions
        """
        self.model.eval()
        predictions = []

        # Prepare initial sequence
        if isinstance(data, pd.Series):
            data = data.values

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Use the last seq_length points as initial sequence
        current_seq = self.model.scaler.transform(data[-seq_length:])
        current_seq = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_seq)
                predictions.append(pred.cpu().numpy())

                # Update sequence for next prediction (sliding window)
                if steps > 1:
                    # Add prediction to sequence and remove first element
                    new_point = pred.unsqueeze(1)
                    current_seq = torch.cat([current_seq[:, 1:, :], new_point], dim=1)

        predictions = np.array(predictions).squeeze()
        return self.model.inverse_transform(predictions)