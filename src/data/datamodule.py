import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 
from src.data.dataset_raw import TransportDataset
from src.data.preprocessing import build_xy_from_specs

class TransportDataModule(L.LightningDataModule):
    """
    LightningDataModule for transport data.

    Responsibilities:
    - load raw TransportDataset
    - build X/Y using input/target specs
    - optionally filter data
    - split train/val/test
    - normalize X/Y using train statistics only
    - create DataLoaders
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.scaler_X = None
        self.scaler_Y = None

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.input_dim = None
        self.output_dim = None

        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

    def _validate_config(self):
        cfg = self.config

        if "input" not in cfg:
            raise ValueError("Config must contain an 'input' spec.")

        if "target" not in cfg:
            raise ValueError("Config must contain a 'target' spec.")

        target_type = cfg["target"]["type"]

        if target_type == "classification" and cfg.get("normalize_y", False):
            raise ValueError(
                "normalize_y=True is not allowed for classification. "
                "Classification labels must remain discrete."
            )

        split_sum = (
            cfg["train_fraction"]
            + cfg["val_fraction"]
            + cfg["test_fraction"]
        )

        if not np.isclose(split_sum, 1.0):
            raise ValueError(
                f"train/val/test fractions must sum to 1. Got {split_sum}"
            )

    def setup(self, stage=None):
        self._validate_config()
        cfg = self.config

        # --------------------------------------------------
        # 1) Load raw cached dataset
        # --------------------------------------------------
        raw_dataset = TransportDataset(
            dataset_path=cfg["dataset_path"],
            cache_dir=cfg["cache_dir"],
            use_cache=cfg.get("use_cache", True),
        )

        # --------------------------------------------------
        # 2) Build task-specific X/Y
        # --------------------------------------------------
        X, Y = build_xy_from_specs(
            dataset=raw_dataset,
            input_spec=cfg["input"],
            target_spec=cfg["target"],
            filters=cfg.get("filters", None),
        )

        # Keep useful metadata
        self.input_spec = cfg["input"]
        self.target_spec = cfg["target"]
        self.filters = cfg.get("filters", None)

        # --------------------------------------------------
        # 3) Convert to numpy
        # --------------------------------------------------
        X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        Y = Y.detach().cpu().numpy() if hasattr(Y, "detach") else np.asarray(Y)

        # Classification target can be shape (N,), regression often (N, D)
        if cfg["target"]["type"] == "classification":
            Y = Y.reshape(-1)

        # --------------------------------------------------
        # 4) Deterministic split
        # --------------------------------------------------
        rng = np.random.default_rng(cfg.get("seed", 42))

        n = len(X)
        idx = rng.permutation(n)

        n_train = round(n * cfg["train_fraction"])
        n_val = round(n * cfg["val_fraction"])
        n_test = n - n_train - n_val

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        # --------------------------------------------------
        # 5) Normalize X using train only
        # --------------------------------------------------
        if cfg.get("normalize_x", False):
            self.scaler_X = StandardScaler()
            X_train = self.scaler_X.fit_transform(X_train)
            X_val = self.scaler_X.transform(X_val)
            X_test = self.scaler_X.transform(X_test)

        # --------------------------------------------------
        # 6) Normalize Y using train only, regression only
        # --------------------------------------------------
        if cfg.get("normalize_y", False):
            if cfg["target"]["type"] == "classification":
                raise ValueError("Y normalization is only allowed for regression.")

            self.scaler_Y = StandardScaler()
            Y_train = self.scaler_Y.fit_transform(Y_train)
            Y_val = self.scaler_Y.transform(Y_val)
            Y_test = self.scaler_Y.transform(Y_test)

        # --------------------------------------------------
        # 7) Convert to tensors
        # --------------------------------------------------
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        if cfg["target"]["type"] == "classification":
            Y_train_t = torch.tensor(Y_train, dtype=torch.long)
            Y_val_t = torch.tensor(Y_val, dtype=torch.long)
            Y_test_t = torch.tensor(Y_test, dtype=torch.long)
        else:
            Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
            Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
            Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

        # --------------------------------------------------
        # 8) Store TensorDatasets
        # --------------------------------------------------
        self.train_set = TensorDataset(X_train_t, Y_train_t)
        self.val_set = TensorDataset(X_val_t, Y_val_t)
        self.test_set = TensorDataset(X_test_t, Y_test_t)

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        self.input_dim = X_train_t.shape[1]

        if cfg["target"]["type"] == "classification":
            self.output_dim = 1
        else:
            self.output_dim = Y_train_t.shape[1]

        self.n_train = len(self.train_set)
        self.n_val = len(self.val_set)
        self.n_test = len(self.test_set)
        self.tempaxis = raw_dataset.get_tempaxis()

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=self.config.get("pin_memory", False),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=self.config.get("pin_memory", False),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=self.config.get("pin_memory", False),
        )
    def get_tempaxis(self):
        return self.tempaxis