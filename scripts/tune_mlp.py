#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
import copy
from datetime import datetime
import math

import yaml
import optuna
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import TransportDataModule
from src.models.mlp import MLPRegressor
from src.lightning_modules.regression_module import RegressionModule
from configs.config import load_config


torch.set_float32_matmul_precision("medium")




class BestMetricTracker(pl.Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        self.monitor = monitor
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf

    def on_validation_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get(self.monitor)

        if value is None:
            return

        value = float(value.detach().cpu().item())

        if self.mode == "min":
            self.best = min(self.best, value)
        else:
            self.best = max(self.best, value)

def suggest_config(trial, base_config):
    config = copy.deepcopy(base_config)

    # -------------------------
    # Model hyperparameters
    # -------------------------
    hidden_name = trial.suggest_categorical("hidden_dims",["64,64","128,128","256,256","128,128,128", "128,256,128","256,256,256","256,512,256","128,128,128,128","128,128,128,128,128"],)

    config["model"]["hidden_dims"] = [int(x) for x in hidden_name.split(",")]

    config["model"]["dropout"] = trial.suggest_float(
        "dropout",
        0.0,
        0.2,
    )

    config["model"]["activation"] = trial.suggest_categorical(
        "activation",
        ["relu", "gelu"],
    )

    # -------------------------
    # Training hyperparameters
    # -------------------------
    config["training"]["lr"] = trial.suggest_float(
        "lr",
        1e-4,
        3e-3,
        log=True,
    )

    config["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        1e-8,
        1e-3,
        log=True,
    )

    config["data"]["batch_size"] = trial.suggest_categorical(
        "batch_size",
        [256, 512, 1024],
    )

    # Shorter tuning runs first
    config["training"]["max_epochs"] = base_config["training"].get("tune_epochs", 100)

    # Important for tuning
    config["training"]["monitor"] = "val_loss"
    config["training"]["monitor_mode"] = "min"
    config["training"]["save_top_k"] = 0

    return config


def build_model(config, datamodule):
    return MLPRegressor(
        in_dim=datamodule.input_dim,
        out_dim=datamodule.output_dim,
        hidden_dims=config["model"]["hidden_dims"],
        activation=config["model"].get("activation", "relu"),
        dropout=config["model"].get("dropout", 0.0),
    )


def objective(trial, base_config, output_root):
    config = suggest_config(trial, base_config)

    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    trial_dir = output_root / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    with open(trial_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # -------------------------
    # Data
    # -------------------------
    datamodule = TransportDataModule(config["data"])
    datamodule.setup()

    # -------------------------
    # Model
    # -------------------------
    model = build_model(config, datamodule)

    lit_model = RegressionModule(
        model=model,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # -------------------------
    # Logging/checkpointing
    # -------------------------
    logger = TensorBoardLogger(

        save_dir=str(trial_dir),

        name="tensorboard",

    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config["training"].get("tune_patience", 20),
    )
    best_tracker = BestMetricTracker(monitor="val_loss", mode="min")

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", "auto"),
        deterministic=config["training"].get("deterministic", True),
        callbacks=[best_tracker, early_stopping],
        logger = logger,
        log_every_n_steps=config["training"].get("log_every_n_steps", 20),
        plugins=[LightningEnvironment()],
        enable_progress_bar=False,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    best_val_loss = best_tracker.best
    trial.set_user_attr("best_checkpoint", None)
    trial.set_user_attr("trial_dir", str(trial_dir))
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="mlp_tuning")
    args = parser.parse_args()

    base_config = load_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("tuning_runs") / f"{timestamp}_{args.study_name}"
    output_root.mkdir(parents=True, exist_ok=True)

    storage_path = output_root / "study.db"

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, base_config, output_root),
        n_trials=args.n_trials,
    )

    print("\nBest trial:")
    print("value:", study.best_trial.value)
    print("params:", study.best_trial.params)
    print("attrs:", study.best_trial.user_attrs)

    with open(output_root / "best_trial.yaml", "w") as f:
        yaml.safe_dump(
            {
                "value": study.best_trial.value,
                "params": study.best_trial.params,
                "user_attrs": study.best_trial.user_attrs,
            },
            f,
            sort_keys=False,
        )

    print("\nSaved tuning results to:")
    print(output_root)


if __name__ == "__main__":
    main()