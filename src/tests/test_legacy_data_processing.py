import sys
from pathlib import Path
import json
import os
import torch
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset



PROJECT_ROOT = Path("/gpfs/data/fs72205/sbayerle/ML/LuttingerWard_from_ML")
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SRC_ROOT))

PROJECT_ROOT = Path.cwd() 
sys.path.append(str(PROJECT_ROOT))

from src.data.datamodule import TransportDataModule

config1 = {
    "dataset_path": "/gpfs/data/fs72205/sbayerle/ML/gather_mega_imp.hdf5",
    "cache_dir": "/gpfs/data/fs72205/sbayerle/ML/cache",
    "use_cache": True,

    "input": {
        "type": "curves",
        "curves": ["L11"],
        "log": True,
        "flatten": True,
    },

    "target": {
        "type": "parameters",
        "params": ["t", "alpha", "delta", "gamma", "eposs", "eweight"],
        "log_params": ["gamma", "eweight"],
    },

    "filters": {
        "accdon": 1.0,
    },

    "normalize_x": False,
    "normalize_y": True,

    "train_fraction": 0.7,
    "val_fraction": 0.1,
    "test_fraction": 0.2,

    "batch_size": 512,
    "num_workers": 0,
    "pin_memory": True,
    "seed": 42,
}

dataset = TransportDataModule(config1)

print(dataset.train_dataloader)
dataset.setup()
X, Y = dataset.train_set.tensors
train_loader = dataset.train_dataloader()


import load_data
from load_data import DatasetKuboMegaRaw
def load_config(savepath):
    """Load config.json from a training run directory."""
    savepath = Path(savepath)
    with open(savepath / "config.json", "r") as f:
        config = json.load(f)
    return config

def load_dataset(config):
    """Instantiate dataset/data loader object from config."""
    loader_name = config["DATA_LOADER"]
    dataset = getattr(load_data, loader_name)(config)
    return dataset
SAVEPATH = "/gpfs/data/fs72205/sbayerle/ML/saves/gather_noimp_digamma-mu-5-11-24/save_Encoder_BS256_2026-01-09/version_1/"

config = load_config(SAVEPATH)
config["PATH_TRAIN"] = "/gpfs/data/fs72205/sbayerle/ML/gather_mega_imp.hdf5"
config["CACHE_DIR"] = "/gpfs/data/fs72205/sbayerle/ML/cache"
dataset = load_dataset(config)


X_all = dataset.get_in_data()

Y_all = dataset.get_target_data()  # first 6 numerical targets

X_all = np.asarray(X_all, dtype=np.float32)
Y_all = np.asarray(Y_all, dtype=np.float32)
mask_neg = Y_all[:, 6] == -1
mask_pos = Y_all[:, 6] == 1

# --- split data set into acdon = 1 / acdon = -1 ---
X_all_neg = X_all[mask_neg]
Y_all_neg = Y_all[mask_neg]
Y_all_neg = Y_all_neg[:,:6]

X_all_pos = X_all[mask_pos]
Y_all_pos = Y_all[mask_pos]
Y_all_pos = Y_all_pos[:,:6]

from sklearn.model_selection import train_test_split
config = config1
rng = np.random.default_rng(config.get("seed", 42))

n = len(X_all_pos)
idx = rng.permutation(n)
train_idx, temp_idx = train_test_split(
            idx,
            test_size=config["val_fraction"] + config["test_fraction"],
            random_state=config.get("seed", 42),
            shuffle=True,
        )

relative_test_size = config["test_fraction"] / (
            config["val_fraction"] + config["test_fraction"]
        )

val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_size,
            random_state=config.get("seed", 42),
            shuffle=True,
        )

n_train = round(n * config["train_fraction"])
train_idx = idx[:n_train]
X_pos_train = X_all_pos[train_idx]

print(f"Does new DataModule result in the same training set as legacy code: {torch.equal(X,torch.tensor(X_pos_train))}")


