#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT))
from datetime import datetime

import yaml
import joblib
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment

from src.data.datamodule import TransportDataModule
from src.models.mlp import MLPRegressor, MLPMeanVariance
from src.lightning_modules.regression_module import RegressionModule
from src.lightning_modules.mve_module import MVEModule
from configs.config import load_config

import torch
torch.set_float32_matmul_precision("medium")



def make_model(config, datamodule):
    model_cfg = config["model"]
    model_type = model_cfg["type"]

    if model_type == "mlp_regressor":
        return MLPRegressor(
            in_dim=datamodule.input_dim,
            out_dim=datamodule.output_dim,
            hidden_dims=model_cfg.get("hidden_dims", [128, 256, 128]),
            activation=model_cfg.get("activation", "relu"),
            dropout=model_cfg.get("dropout", 0.0),
        )

    if model_type == "mlp_mve":
        return MLPMeanVariance(
            in_dim=datamodule.input_dim,
            out_dim=datamodule.output_dim,
            hidden_dims=model_cfg.get("hidden_dims", [128, 256, 128]),
            activation=model_cfg.get("activation", "relu"),
            dropout=model_cfg.get("dropout", 0.0),
            log_var_min=model_cfg.get("log_var_min", -10.0),
            log_var_max=model_cfg.get("log_var_max", 10.0),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def make_lightning_module(config, model):
    model_type = config["model"]["type"]
    train_cfg = config["training"]

    if model_type == "mlp_regressor":
        return RegressionModule(
            model=model,
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

    if model_type == "mlp_mve":
        return MVEModule(
            model=model,
            lr=train_cfg.get("lr", 1e-3),
            mean_weight_decay=train_cfg.get("mean_weight_decay", 0.0),
            var_weight_decay = train_cfg.get("var_weight_decay",0.0),
            mean_epochs=train_cfg.get("mean_epochs", 250),
            variance_epochs=train_cfg.get("variance_epochs", 250),
            joint_training=train_cfg.get("joint_training", False),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def make_run_dir(config_path, config):
    run_root = Path(config.get("run_root", "runs"))
    experiment_name = config.get("experiment_name", Path(config_path).stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = run_root / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def save_artifacts(run_dir, config_path, config, datamodule):
    # Save config copy
    shutil.copy(config_path, run_dir / "config.yaml")

    # Save split indices
    joblib.dump(
        {
            "train_idx": datamodule.train_idx,
            "val_idx": datamodule.val_idx,
            "test_idx": datamodule.test_idx,
        },
        run_dir / "split_indices.pkl",
    )

    # Save scalers
    joblib.dump(
        {
            "scaler_X": datamodule.scaler_X,
            "scaler_Y": datamodule.scaler_Y,
        },
        run_dir / "scalers.pkl",
    )

    # Save basic metadata
    metadata = {
        "input_dim": datamodule.input_dim,
        "output_dim": datamodule.output_dim,
        "n_train": datamodule.n_train,
        "n_val": datamodule.n_val,
        "n_test": datamodule.n_test,
        "input_spec": getattr(datamodule, "input_spec", None),
        "target_spec": getattr(datamodule, "target_spec", None),
        "filters": getattr(datamodule, "filters", None),
    }

    with open(run_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    run_dir = make_run_dir(config_path, config)
    datamodule = TransportDataModule(config["data"])
    datamodule.setup()

    print("Data ready")
    print("input_dim:", datamodule.input_dim)
    print("output_dim:", datamodule.output_dim)
    print("train:", datamodule.n_train)
    print("val:", datamodule.n_val)
    print("test:", datamodule.n_test)
    model = make_model(config, datamodule)
    lit_model = make_lightning_module(config, model)
    logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="tensorboard",
    )

    monitor = config["training"].get("monitor", "val_loss")
    monitor_mode = config["training"].get("monitor_mode", "min")
    checkpoint_filename = f"epoch={{epoch:03d}}-{monitor}={{{monitor}:.6f}}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename=checkpoint_filename,
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=config["training"].get("save_top_k", 1),
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]

    if config["training"].get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor=config["training"].get("monitor", "val_loss"),
                mode=config["training"].get("monitor_mode", "min"),
                patience=config["training"].get("patience", 50),
            )
        )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=config["training"].get("max_epochs", 500),
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", "auto"),
        deterministic=config["training"].get("deterministic", True),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config["training"].get("log_every_n_steps", 20),
        plugins=[LightningEnvironment()],
    )

    # --------------------------------------------------
    # Save preprocessing/scaler/split info before training
    # --------------------------------------------------
    save_artifacts(run_dir, config_path, config, datamodule)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    trainer.fit(lit_model, datamodule=datamodule)

    # --------------------------------------------------
    # Validate/test best checkpoint
    # --------------------------------------------------
    val_metrics = trainer.validate(
        lit_model,
        datamodule=datamodule,
        ckpt_path="best",
    )

    test_metrics = trainer.test(
        lit_model,
        datamodule=datamodule,
        ckpt_path="best",
    )

    with open(run_dir / "final_metrics.yaml", "w") as f:
        yaml.safe_dump(
            {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "best_checkpoint": checkpoint_callback.best_model_path,
            },
            f,
            sort_keys=False,
        )

    print("\nTraining finished.")
    print("Run directory:", run_dir)
    print("Best checkpoint:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
