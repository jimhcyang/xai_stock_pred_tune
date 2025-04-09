# my_models.py

"""
This module provides PyTorch model definitions (MLP, CNN, RNN, LSTM, GRU)
and a build_dl_model function that instantiates them with user-specified
hyperparameters. Import and use in your training scripts or notebooks.
"""

import os
import json
import pandas as pd
import numpy as np

# For data scaling & metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_activation_fn(name: str) -> nn.Module:
    """
    Returns a PyTorch activation module given a string name.
    Supported names: 'relu', 'tanh', 'leakyrelu'.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    """
    A multi-layer perceptron for simple feed-forward tasks.
    Expects input (batch_size, input_dim).
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        n_layers: int = 1, 
        dropout: float = 0.0, 
        activation: str = "relu", 
        use_batchnorm: bool = False
    ):
        """
        :param input_dim: Number of input features
        :param hidden_dim: Width of each hidden layer
        :param n_layers: How many hidden layers
        :param dropout: Drop probability
        :param activation: Activation name (relu, tanh, leakyrelu)
        :param use_batchnorm: Whether to apply BatchNorm1d after each linear
        """
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        act_layer = get_activation_fn(activation)

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            lin = nn.Linear(in_dim, hidden_dim)
            layers.append(lin)
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(act_layer)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, input_dim)
        :return: shape (batch_size, 1)
        """
        return self.net(x)

class CNN(nn.Module):
    """
    Very simple 1D CNN. Assumes input shape is (batch_size, 1, input_dim).
    Applies multiple conv layers, optional pooling, then a linear output.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
        kernel_size: int = 3,
        pooling_type: str = "max",
        activation: str = "relu"
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.pooling_type = pooling_type.lower()
        self.act_layer = get_activation_fn(activation)

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_ch = 1 if i == 0 else hidden_dim
            conv = nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=kernel_size // 2)
            self.convs.append(conv)

        if self.pooling_type == "max":
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        elif self.pooling_type == "avg":
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            self.pool = None

        self.fc = nn.Linear(hidden_dim * (input_dim // 2), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, 1, input_dim)
        :return: shape (batch_size, 1)
        """
        for conv in self.convs:
            x = conv(x)
            x = self.act_layer(x)
            x = self.dropout(x)

        if self.pool is not None:
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class RNN(nn.Module):
    """
    A simple RNN-based model for time-series data.
    Expects input (batch_size, seq_len, input_dim).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_features = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, seq_len, input_dim)
        :return: shape (batch_size, 1)
        """
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last time-step
        out = self.fc(out)
        return out

class LSTM(nn.Module):
    """
    LSTM-based model for time-series data.
    Expects input (batch_size, seq_len, input_dim).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        recurrent_dropout: float = 0.0
    ):
        super().__init__()
        if recurrent_dropout != 0.0:
            print("Warning: PyTorch LSTM does not implement true 'recurrent dropout' natively.")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_features = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, seq_len, input_dim)
        :return: shape (batch_size, 1)
        """
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class GRU(nn.Module):
    """
    GRU-based model for time-series data.
    Expects input (batch_size, seq_len, input_dim).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        recurrent_dropout: float = 0.0
    ):
        super().__init__()
        if recurrent_dropout != 0.0:
            print("Warning: PyTorch GRU does not implement 'true recurrent dropout' natively.")
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_features = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (batch_size, seq_len, input_dim)
        :return: shape (batch_size, 1)
        """
        out, hn = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def build_dl_model(model_name: str, input_dim: int, params: dict) -> nn.Module:
    """
    Creates and returns an instance of the requested model
    with architecture hyperparams from 'params'.

    :param model_name: One of: "MLP", "CNN", "RNN", "LSTM", "GRU"
    :param input_dim:  Number of input features per sample
    :param params:     Dict of hyperparameters (hidden_dim, n_layers, dropout, etc.)

    :return: An nn.Module on the current 'device'
    """
    hidden_dim = params.get("hidden_dim", 64)
    n_layers   = params.get("n_layers", 1)
    dropout    = params.get("dropout", 0.0)
    activation = params.get("activation", "relu")
    use_bn     = params.get("use_batchnorm", False)
    kernel_size= params.get("kernel_size", 3)
    pooling_ty = params.get("pooling_type", "max")
    bidirect   = params.get("bidirectional", False)
    rec_dropout= params.get("recurrent_dropout", 0.0)

    if model_name == "MLP":
        model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            activation=activation,
            use_batchnorm=use_bn
        )
    elif model_name == "CNN":
        model = CNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            kernel_size=kernel_size,
            pooling_type=pooling_ty,
            activation=activation
        )
    elif model_name == "RNN":
        model = RNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirect
        )
    elif model_name == "LSTM":
        model = LSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirect,
            recurrent_dropout=rec_dropout
        )
    elif model_name == "GRU":
        model = GRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirect,
            recurrent_dropout=rec_dropout
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.to(device)
    return model

def sliding_window_forecast_with_torch(
    df,
    target_col,
    model_name,
    model_params,
    window_size=25,
    test_ratio=0.2,
    drop_cols=None,
    log_dir="model_dl_logs",
    log_filename="prediction_log.csv",
    device=None
):
    """
    Rolls a window of size `window_size` through the data. For each window:
      - Splits train/test by `test_ratio`
      - Standard-scales X and y
      - Builds and trains a PyTorch model with the given `model_name` and `model_params`
      - Predicts on test portion, inverse-scales predictions
      - Logs performance metrics (MSE, MAPE, R²) to CSV
    
    Parameters
    ----------
    df : pd.DataFrame
        Source data, assumed time-indexed. Must contain `target_col`.
    target_col : str
        Name of the column we want to forecast.
    model_name : str
        One of ["MLP", "CNN", "RNN", "LSTM", "GRU"].
    model_params : dict
        Hyperparams for both architecture and training (hidden_dim, dropout, learning_rate, etc.).
    window_size : int
        How many rows (days) to include in each rolling window.
    test_ratio : float
        Fraction of rows in each window that will be used as test set.
    drop_cols : list
        Any columns to drop before building features.
    log_dir : str
        Directory to which logs are written.
    log_filename : str
        Filename for the CSV log.
    device : torch.device or None
        If None, picks cuda if available else cpu.

    Returns
    -------
    pd.DataFrame
        The log of each sliding window’s metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if drop_cols is None:
        drop_cols = []
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_filename)
    log_entries = []

    # Sort and drop NA
    df = df.dropna().sort_index()
    n = len(df)
    
    for start in range(0, n - window_size + 1):
        # Extract the window
        window_df = df.iloc[start : start + window_size].copy()
        X_window = window_df.drop(columns=[target_col] + drop_cols, errors='ignore')
        y_window = window_df[target_col].values

        # Split into train/test by ratio
        train_size = int(len(X_window) * (1 - test_ratio))
        if train_size < 1 or train_size >= len(X_window):
            # Not enough samples to do a split
            continue
        
        X_train = X_window.iloc[:train_size].values
        X_test  = X_window.iloc[train_size:].values
        y_train = y_window[:train_size]
        y_test  = y_window[train_size:]

        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)

        # Scale target
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Format for model
        input_dim = X_train_scaled.shape[1]

        # If model is CNN or RNN-based, we expect a 3D input:
        #   CNN: (batch_size, 1, input_dim)
        #   RNN: (batch_size, seq_len=1, input_dim) => effectively the same shape if you expand axis=1
        if model_name in ["CNN", "RNN", "LSTM", "GRU"]:
            X_train_scaled = np.expand_dims(X_train_scaled, axis=1)  # => shape (batch_size, 1, input_dim)
            X_test_scaled  = np.expand_dims(X_test_scaled, axis=1)

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
        X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_t  = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1).to(device)

        # Build model
        model = build_dl_model(model_name, input_dim, model_params)
        model.to(device)

        # Extract training hyperparams
        lr         = model_params.get("learning_rate", 1e-3)
        opt_name   = model_params.get("optimizer", "adam").lower()
        epochs     = model_params.get("epochs", 20)
        batch_size = model_params.get("batch_size", 32)
        wd         = model_params.get("weight_decay", 0.0)
        momentum   = model_params.get("momentum", 0.0)

        # Create optimizer
        if opt_name == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # Optionally: handle gradient clipping or schedulers if in model_params
        grad_clip_val = model_params.get("gradient_clip_val", 0.0)
        scheduler_type = model_params.get("scheduler", "none").lower()

        if scheduler_type == "step":
            # Hard-coded step size or read from params
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.1)
        else:
            scheduler = None

        criterion = nn.MSELoss()
        
        # Training loop
        dataset_size = X_train_t.shape[0]
        num_batches = (dataset_size + batch_size - 1) // batch_size
        model.train()

        for ep in range(epochs):
            # Shuffle
            perm = torch.randperm(dataset_size)
            X_train_t = X_train_t[perm]
            y_train_t = y_train_t[perm]

            epoch_loss = 0.0

            for b_idx in range(num_batches):
                start_b = b_idx * batch_size
                end_b = min(start_b + batch_size, dataset_size)

                xb = X_train_t[start_b:end_b]
                yb = y_train_t[start_b:end_b]

                opt.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()

                # gradient clipping if needed
                if grad_clip_val > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

                opt.step()
                epoch_loss += loss.item()

            if scheduler:
                # If StepLR, call scheduler.step() each epoch
                # If ReduceLROnPlateau, call scheduler.step(epoch_loss) or so
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

        # Inference
        model.eval()
        with torch.no_grad():
            preds_test = model(X_test_t)
        preds_test = preds_test.cpu().numpy().ravel()
        y_true_test = y_test_t.cpu().numpy().ravel()

        # Inverse transform
        preds_test = scaler_y.inverse_transform(preds_test.reshape(-1, 1)).ravel()
        y_true_test = scaler_y.inverse_transform(y_true_test.reshape(-1, 1)).ravel()

        # Guard against invalid output
        if np.any(np.isnan(preds_test)) or np.any(np.isinf(preds_test)):
            print("Skipping window due to NaN/inf in predictions.")
            continue

        # Metrics
        mse = mean_squared_error(y_true_test, preds_test)
        # If all ground truth are zero, MAPE can blow up. 
        # So you might check if np.allclose(y_true_test, 0.0) etc. 
        # We'll just do naive approach:
        def mean_absolute_percentage_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0

        mape = mean_absolute_percentage_error(y_true_test, preds_test)
        r2 = r2_score(y_true_test, preds_test)

        log_entry = {
            "model_name": model_name,
            "model_hyperparameters_dict": json.dumps(model_params),
            "window_size": window_size,
            "test_ratio": test_ratio,
            "start_date": str(df.index[start]),
            "end_date": str(df.index[start + window_size - 1]),
            "test_data_values_list": y_true_test.tolist(),
            "test_data_model_predictions_list": preds_test.tolist(),
            "MSE_score": mse,
            "MAPE_score": mape,
            "R^2_score": r2
        }
        log_entries.append(log_entry)

    log_df = pd.DataFrame(log_entries)
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    return log_df