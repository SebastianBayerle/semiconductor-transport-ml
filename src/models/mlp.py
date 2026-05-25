import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Flexible MLP for regression.

    Example:
        model = MLPRegressor(
            in_dim=100,
            out_dim=6,
            hidden_dims=[128, 256, 128],
            dropout=0.0,
        )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims=(128, 256, 128),
        activation="relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = list(hidden_dims)

        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "gelu":
            activation_layer = nn.GELU
        elif activation == "tanh":
            activation_layer = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = in_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_layer())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPMeanVariance(nn.Module):
    """
    Flexible MLP for Mean-Variance Estimation.

    Returns:
        mean, log_var
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims=(128, 256, 128),
        activation="relu",
        dropout: float = 0.0,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = list(hidden_dims)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "gelu":
            activation_layer = nn.GELU
        elif activation == "tanh":
            activation_layer = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = in_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_layer())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev_dim, out_dim)
        self.log_var_head = nn.Linear(prev_dim, out_dim)

    def forward(self, x):
        h = self.shared(x)

        mean = self.mean_head(h)
        log_var = self.log_var_head(h)
        log_var = torch.clamp(log_var, self.log_var_min, self.log_var_max)

        return mean, log_var