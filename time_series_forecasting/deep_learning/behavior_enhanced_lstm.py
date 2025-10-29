"""
Behavior-Enhanced LSTM Model

LSTM model that incorporates behavior analysis features
in addition to pure price time series.

Architecture:
1. LSTM layers process price history
2. Dense layers process behavior features
3. Combined representation for prediction
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler


class BehaviorEnhancedLSTM(nn.Module):
    """
    LSTM model enhanced with behavior features.

    Architecture:
    - Price LSTM: Processes price time series
    - Behavior Network: Processes behavior features
    - Fusion Layer: Combines both representations
    - Output Layer: Produces price prediction
    """

    def __init__(self,
                 price_input_size: int = 1,
                 behavior_input_size: int = 17,  # Number of behavior features
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 2,
                 behavior_hidden_size: int = 32,
                 fusion_hidden_size: int = 32,
                 output_size: int = 1,
                 dropout: float = 0.2):
        """
        Initialize behavior-enhanced LSTM.

        Parameters:
        -----------
        price_input_size : int
            Size of price features
        behavior_input_size : int
            Size of behavior features
        lstm_hidden_size : int
            LSTM hidden state size
        lstm_num_layers : int
            Number of LSTM layers
        behavior_hidden_size : int
            Hidden size for behavior network
        fusion_hidden_size : int
            Hidden size for fusion layer
        output_size : int
            Output size (typically 1 for price prediction)
        dropout : float
            Dropout probability
        """
        super(BehaviorEnhancedLSTM, self).__init__()

        self.price_input_size = price_input_size
        self.behavior_input_size = behavior_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # Price LSTM pathway
        self.price_lstm = nn.LSTM(
            input_size=price_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True
        )

        # Behavior feature pathway
        self.behavior_network = nn.Sequential(
            nn.Linear(behavior_input_size, behavior_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(behavior_hidden_size, behavior_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        combined_size = lstm_hidden_size + behavior_hidden_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, output_size)
        )

        # Scalers
        self.price_scaler = StandardScaler()
        self.behavior_scaler = StandardScaler()
        self.is_fitted = False

    def forward(self, price_seq: torch.Tensor, behavior_features: torch.Tensor
               ) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        price_seq : torch.Tensor
            Price sequence [batch_size, seq_len, price_input_size]
        behavior_features : torch.Tensor
            Behavior features [batch_size, behavior_input_size]

        Returns:
        --------
        torch.Tensor: Predictions [batch_size, output_size]
        """
        # Process price sequence through LSTM
        lstm_out, _ = self.price_lstm(price_seq)
        # Take last output
        lstm_last = lstm_out[:, -1, :]

        # Process behavior features
        behavior_out = self.behavior_network(behavior_features)

        # Combine
        combined = torch.cat([lstm_last, behavior_out], dim=1)

        # Final prediction
        output = self.fusion_layer(combined)

        return output


class BehaviorEnhancedLSTMTrainer:
    """
    Trainer for behavior-enhanced LSTM.
    """

    def __init__(self,
                 model: BehaviorEnhancedLSTM,
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize trainer.

        Parameters:
        -----------
        model : BehaviorEnhancedLSTM
            Model to train
        learning_rate : float
            Learning rate
        device : Optional[str]
            Device to train on
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def prepare_data(self,
                    price_data: np.ndarray,
                    behavior_data: np.ndarray,
                    seq_length: int,
                    forecast_horizon: int = 1,
                    fit_scalers: bool = True
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training.

        Parameters:
        -----------
        price_data : np.ndarray
            Price time series [T, price_features]
        behavior_data : np.ndarray
            Behavior features [T, behavior_features]
        seq_length : int
            Length of input sequences
        forecast_horizon : int
            Steps to forecast ahead
        fit_scalers : bool
            Whether to fit scalers

        Returns:
        --------
        Tuple: (price_sequences, behavior_features, targets)
        """
        # Ensure 2D
        if price_data.ndim == 1:
            price_data = price_data.reshape(-1, 1)

        # Scale data
        if fit_scalers:
            scaled_prices = self.model.price_scaler.fit_transform(price_data)
            scaled_behavior = self.model.behavior_scaler.fit_transform(behavior_data)
        else:
            scaled_prices = self.model.price_scaler.transform(price_data)
            scaled_behavior = self.model.behavior_scaler.transform(behavior_data)

        # Create sequences
        price_seqs, behavior_feats, targets = [], [], []

        for i in range(len(scaled_prices) - seq_length - forecast_horizon + 1):
            # Price sequence
            price_seqs.append(scaled_prices[i:i+seq_length])

            # Behavior features at end of sequence
            behavior_feats.append(scaled_behavior[i+seq_length-1])

            # Target (future price)
            targets.append(scaled_prices[i+seq_length+forecast_horizon-1])

        return (
            torch.FloatTensor(np.array(price_seqs)),
            torch.FloatTensor(np.array(behavior_feats)),
            torch.FloatTensor(np.array(targets))
        )

    def train_epoch(self,
                   price_seqs: torch.Tensor,
                   behavior_feats: torch.Tensor,
                   targets: torch.Tensor,
                   batch_size: int = 32) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Create batches
        dataset = torch.utils.data.TensorDataset(price_seqs, behavior_feats, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for batch_prices, batch_behavior, batch_targets in dataloader:
            batch_prices = batch_prices.to(self.device)
            batch_behavior = batch_behavior.to(self.device)
            batch_targets = batch_targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(batch_prices, batch_behavior)

            # Handle target shape
            if batch_targets.dim() == 3:
                batch_targets = batch_targets.squeeze(-1)

            # Loss
            loss = self.criterion(predictions, batch_targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self,
                      price_seqs: torch.Tensor,
                      behavior_feats: torch.Tensor,
                      targets: torch.Tensor,
                      batch_size: int = 32) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        dataset = torch.utils.data.TensorDataset(price_seqs, behavior_feats, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch_prices, batch_behavior, batch_targets in dataloader:
                batch_prices = batch_prices.to(self.device)
                batch_behavior = batch_behavior.to(self.device)
                batch_targets = batch_targets.to(self.device)

                predictions = self.model(batch_prices, batch_behavior)

                if batch_targets.dim() == 3:
                    batch_targets = batch_targets.squeeze(-1)

                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def fit(self,
            price_data: np.ndarray,
            behavior_data: np.ndarray,
            seq_length: int,
            forecast_horizon: int = 1,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.2,
            early_stopping_patience: int = 10,
            verbose: bool = True) -> Dict:
        """
        Fit model to data.

        Parameters:
        -----------
        price_data : np.ndarray
            Price time series
        behavior_data : np.ndarray
            Behavior features
        seq_length : int
            Sequence length
        forecast_horizon : int
            Forecast horizon
        epochs : int
            Training epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        early_stopping_patience : int
            Early stopping patience
        verbose : bool
            Print training progress

        Returns:
        --------
        Dict: Training history
        """
        # Prepare data
        price_seqs, behavior_feats, targets = self.prepare_data(
            price_data, behavior_data, seq_length, forecast_horizon, fit_scalers=True
        )

        # Split data
        split_idx = int(len(price_seqs) * (1 - validation_split))

        train_price = price_seqs[:split_idx]
        train_behavior = behavior_feats[:split_idx]
        train_targets = targets[:split_idx]

        val_price = price_seqs[split_idx:]
        val_behavior = behavior_feats[split_idx:]
        val_targets = targets[split_idx:]

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(
                train_price, train_behavior, train_targets, batch_size
            )
            val_loss = self.validate_epoch(
                val_price, val_behavior, val_targets, batch_size
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_behavior_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_behavior_model.pth'))
        self.model.is_fitted = True

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }

    def predict(self,
                price_data: np.ndarray,
                behavior_data: np.ndarray,
                seq_length: int,
                steps: int = 1) -> np.ndarray:
        """
        Generate predictions.

        Parameters:
        -----------
        price_data : np.ndarray
            Price time series
        behavior_data : np.ndarray
            Behavior features
        seq_length : int
            Sequence length
        steps : int
            Number of steps to predict

        Returns:
        --------
        np.ndarray: Predictions
        """
        self.model.eval()
        predictions = []

        # Prepare initial sequence
        if price_data.ndim == 1:
            price_data = price_data.reshape(-1, 1)

        current_seq = self.model.price_scaler.transform(price_data[-seq_length:])
        current_seq = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for step in range(steps):
                # Get behavior features for current step
                behavior_idx = min(-1 - (steps - step - 1), -1)
                # Use proper slicing for negative indices
                if behavior_idx == -1:
                    behavior_feat_raw = behavior_data[-1:].reshape(1, -1)
                else:
                    behavior_feat_raw = behavior_data[behavior_idx:behavior_idx+1]
                behavior_feat = self.model.behavior_scaler.transform(behavior_feat_raw)
                behavior_tensor = torch.FloatTensor(behavior_feat).to(self.device)

                # Predict
                pred = self.model(current_seq, behavior_tensor)
                predictions.append(pred.cpu().numpy())

                # Update sequence for next prediction
                if steps > 1 and step < steps - 1:
                    new_point = pred.unsqueeze(1)
                    current_seq = torch.cat([current_seq[:, 1:, :], new_point], dim=1)

        predictions = np.array(predictions).squeeze()

        # Inverse transform
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return self.model.price_scaler.inverse_transform(predictions).flatten()
