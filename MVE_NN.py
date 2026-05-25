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


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
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

from src.data.dataset_raw import TransportDataset
from src.data.preprocessing import prepare_transport_data

dataset=TransportDataset(dataset_path="/gpfs/data/fs72205/sbayerle/ML/gather_mega_imp.hdf5",cache_dir="/gpfs/data/fs72205/sbayerle/ML/cache")

X, Y = prepare_transport_data(
    dataset,
    input_curves=("L11B",),
    target_params=("t", "alpha", "delta", "gamma", "eposs", "eweight", "accdon"),
    log_input=True,
    log_targets=("gamma", "eweight"),
    flatten=True,
)

config = load_config(SAVEPATH)
config["PATH_TRAIN"] = "/gpfs/data/fs72205/sbayerle/ML/gather_mega_imp.hdf5"
config["CACHE_DIR"] = "/gpfs/data/fs72205/sbayerle/ML/cache"
dataset = load_dataset(config)


X_all = dataset.get_in_data()
print(X_all.shape)
print(X.shape)
print("is same::")
print(torch.equal(X_all,X))

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

# split once
X_train_pos, X_test_pos, Y_train_pos, Y_test_pos = train_test_split(
    X_all_pos, Y_all_pos, test_size=0.2, random_state=42
)



# --- normalize X ---
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_pos)
X_test_scaled  = scaler_X.transform(X_test_pos)

# --- normalize Y (for regression!) ---
scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train_pos)
Y_test_scaled  = scaler_Y.transform(Y_test_pos)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
# --------------------------------------------------
# Normalize inputs and targets
# --------------------------------------------------
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_pos)
X_test_scaled  = scaler_X.transform(X_test_pos)

scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train_pos)
Y_test_scaled  = scaler_Y.transform(Y_test_pos)

# convert to torch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train_scaled, dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test_scaled, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=2048,
    shuffle=True,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)