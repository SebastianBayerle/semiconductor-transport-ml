#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
import yaml

import torch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import TransportDataModule
from src.models.mlp import MLPMeanVariance
from src.lightning_modules.mve_module import MVEModule
from src.evaluation.uncertainty_mve_module import (
    PARAM_NAMES,
    predict_mve_normalized,
    uncertainty_by_parameter,
    gaussian_interval_coverage,
    gaussian_interval_coverage_by_parameter,
    uncertainty_error_correlation,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_mve_from_config(config, datamodule):
    model_cfg = config["model"]

    model = MLPMeanVariance(
        in_dim=datamodule.input_dim,
        out_dim=datamodule.output_dim,
        hidden_dims=model_cfg.get("hidden_dims", [128, 256, 128]),
        activation=model_cfg.get("activation", "relu"),
        dropout=model_cfg.get("dropout", 0.0),
        log_var_min=model_cfg.get("log_var_min", -10.0),
        log_var_max=model_cfg.get("log_var_max", 10.0),
    )

    lit_model = MVEModule.load_from_checkpoint(
        checkpoint_path=config["checkpoint_path"],
        model=model,
        lr=config["training"].get("lr", 1e-3),
        weight_decay=config["training"].get("weight_decay", 0.0),
        warmup_epochs=config["training"].get("warmup_epochs", 100),
    )

    return lit_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the original config.yaml used for training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained MVE checkpoint.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for diagnostic CSV files.",
    )
    parser.add_argument(
        "--group-param",
        type=str,
        default="eweight",
        help="Parameter used for grouped diagnostics.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["checkpoint_path"] = args.checkpoint

    if config["model"]["type"] != "mlp_mve":
        raise ValueError(
            f"This diagnostic script expects model.type='mlp_mve', "
            f"got {config['model']['type']}"
        )

    outdir = Path(args.outdir) if args.outdir is not None else Path(args.checkpoint).parents[1] / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("medium")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    datamodule = TransportDataModule(config["data"])
    datamodule.setup()

    target_names = config["data"]["target"].get("params", PARAM_NAMES)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    lit_model = build_mve_from_config(config, datamodule)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)

    # --------------------------------------------------
    # Predict in normalized/model space
    # --------------------------------------------------
    test_loader = datamodule.test_dataloader()

    Y_true, Y_mean, Y_std = predict_mve_normalized(
        lit_model=lit_model,
        dataloader=test_loader,
        device=device,
    )

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    coverage = gaussian_interval_coverage(
        Y_true=Y_true,
        Y_mean=Y_mean,
        Y_std=Y_std,
        target_names=target_names,
        sigmas=(1.0, 2.0, 3.0),
    )

    corr = uncertainty_error_correlation(
        Y_true=Y_true,
        Y_mean=Y_mean,
        Y_std=Y_std,
        target_names=target_names,
    )

    unc_by_param = uncertainty_by_parameter(
        Y_true=Y_true,
        Y_pred_std=Y_std,
        group_param=args.group_param,
        target_names=target_names,
        round_group_values=8,
    )

    coverage_by_param = gaussian_interval_coverage_by_parameter(
        Y_true=Y_true,
        Y_mean=Y_mean,
        Y_std=Y_std,
        group_param=args.group_param,
        target_names=target_names,
        sigmas=(1.0, 2.0),
        round_group_values=8,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    coverage.to_csv(outdir / "coverage.csv", index=False)
    corr.to_csv(outdir / "uncertainty_error_correlation.csv", index=False)
    unc_by_param.to_csv(outdir / f"uncertainty_by_{args.group_param}.csv", index=False)
    coverage_by_param.to_csv(outdir / f"coverage_by_{args.group_param}.csv", index=False)

    # Also save raw predictions for later plotting
    pd.DataFrame(Y_true, columns=[f"true_{n}" for n in target_names]).to_csv(
        outdir / "Y_true_normalized.csv",
        index=False,
    )

    pd.DataFrame(Y_mean, columns=[f"mean_{n}" for n in target_names]).to_csv(
        outdir / "Y_mean_normalized.csv",
        index=False,
    )

    pd.DataFrame(Y_std, columns=[f"std_{n}" for n in target_names]).to_csv(
        outdir / "Y_std_normalized.csv",
        index=False,
    )

    print("\nSaved diagnostics to:")
    print(outdir)

    print("\nCoverage:")
    print(coverage)

    print("\nUncertainty-error correlation:")
    print(corr)


if __name__ == "__main__":
    main()
