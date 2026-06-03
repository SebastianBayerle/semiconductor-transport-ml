import numpy as np
import pandas as pd
import torch


PARAM_NAMES = ["t", "alpha", "delta", "gamma", "eposs", "eweight"]


def predict_mve_normalized(lit_model, dataloader, device=None):
    """
    Collect MVE predictions in model/native target space.

    Usually this is normalized target space if normalize_y=True.
    """
    if device is None:
        device = next(lit_model.parameters()).device

    lit_model.eval()

    y_true_all = []
    y_mean_all = []
    y_std_all = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            mean, log_var = lit_model(x)
            std = torch.exp(0.5 * log_var)

            y_true_all.append(y.detach().cpu())
            y_mean_all.append(mean.detach().cpu())
            y_std_all.append(std.detach().cpu())

    Y_true = torch.cat(y_true_all, dim=0).numpy()
    Y_mean = torch.cat(y_mean_all, dim=0).numpy()
    Y_std = torch.cat(y_std_all, dim=0).numpy()

    return Y_true, Y_mean, Y_std


def uncertainty_by_parameter(
    Y_true,
    Y_pred_std,
    group_param,
    target_names=PARAM_NAMES,
    agg="mean",
    round_group_values=None,
):
    """
    Summarize predictive uncertainty grouped by one true parameter.
    """
    Y_true = np.asarray(Y_true)
    Y_pred_std = np.asarray(Y_pred_std)

    if Y_true.shape != Y_pred_std.shape:
        raise ValueError("Y_true and Y_pred_std must have same shape.")

    if group_param not in target_names:
        raise ValueError(f"group_param must be one of {target_names}")

    df = pd.DataFrame(Y_true, columns=target_names)

    if round_group_values is not None:
        df[group_param] = df[group_param].round(round_group_values)

    for i, name in enumerate(target_names):
        df[f"unc_{name}"] = Y_pred_std[:, i]

    df["unc_mean_all_targets"] = Y_pred_std.mean(axis=1)
    df["unc_max_all_targets"] = Y_pred_std.max(axis=1)

    agg_cols = [f"unc_{name}" for name in target_names]
    agg_cols += ["unc_mean_all_targets", "unc_max_all_targets"]

    out = (
        df.groupby(group_param)[agg_cols]
        .agg(agg)
        .reset_index()
        .sort_values(group_param)
    )

    counts = df.groupby(group_param).size().reset_index(name="count")
    out = out.merge(counts, on=group_param, how="left")

    return out


def gaussian_interval_coverage(
    Y_true,
    Y_mean,
    Y_std,
    target_names=PARAM_NAMES,
    sigmas=(1.0, 2.0, 3.0),
    eps=1e-8,
):
    """
    Check how often true values lie inside predicted Gaussian intervals.

    For a calibrated Gaussian model, approximately:
        1 sigma -> 68.3%
        2 sigma -> 95.4%
        3 sigma -> 99.7%

    Works best in normalized target space.
    """
    Y_true = np.asarray(Y_true)
    Y_mean = np.asarray(Y_mean)
    Y_std = np.asarray(Y_std)

    if not (Y_true.shape == Y_mean.shape == Y_std.shape):
        raise ValueError(
            f"Shapes must match. Got true={Y_true.shape}, "
            f"mean={Y_mean.shape}, std={Y_std.shape}"
        )

    Y_std = np.maximum(Y_std, eps)
    abs_err = np.abs(Y_true - Y_mean)

    rows = []

    for i, name in enumerate(target_names):
        for sigma in sigmas:
            inside = abs_err[:, i] <= sigma * Y_std[:, i]

            rows.append(
                {
                    "target": name,
                    "sigma": sigma,
                    "empirical_coverage": inside.mean(),
                    "mean_pred_std": Y_std[:, i].mean(),
                    "mae": abs_err[:, i].mean(),
                    "rmse": np.sqrt(np.mean((Y_true[:, i] - Y_mean[:, i]) ** 2)),
                    "mean_abs_z": np.mean(abs_err[:, i] / Y_std[:, i]),
                    "count": len(Y_true),
                }
            )

    # all targets pooled
    for sigma in sigmas:
        inside = abs_err <= sigma * Y_std

        rows.append(
            {
                "target": "all",
                "sigma": sigma,
                "empirical_coverage": inside.mean(),
                "mean_pred_std": Y_std.mean(),
                "mae": abs_err.mean(),
                "rmse": np.sqrt(np.mean((Y_true - Y_mean) ** 2)),
                "mean_abs_z": np.mean(abs_err / Y_std),
                "count": Y_true.size,
            }
        )

    return pd.DataFrame(rows)


def gaussian_interval_coverage_by_parameter(
    Y_true,
    Y_mean,
    Y_std,
    group_param,
    target_names=PARAM_NAMES,
    sigmas=(1.0, 2.0),
    round_group_values=None,
    eps=1e-8,
):
    """
    Coverage diagnostic grouped by one true parameter value.

    Example:
        group_param="eweight"

    This answers:
        Is uncertainty calibrated equally well across different eweight values?
    """
    Y_true = np.asarray(Y_true)
    Y_mean = np.asarray(Y_mean)
    Y_std = np.asarray(Y_std)

    if not (Y_true.shape == Y_mean.shape == Y_std.shape):
        raise ValueError("Y_true, Y_mean, and Y_std must have same shape.")

    if group_param not in target_names:
        raise ValueError(f"group_param must be one of {target_names}")

    Y_std = np.maximum(Y_std, eps)
    abs_err = np.abs(Y_true - Y_mean)

    group_idx = target_names.index(group_param)
    group_values = Y_true[:, group_idx].copy()

    if round_group_values is not None:
        group_values = np.round(group_values, round_group_values)

    rows = []

    for value in np.sort(np.unique(group_values)):
        mask = group_values == value

        for i, target in enumerate(target_names):
            for sigma in sigmas:
                inside = abs_err[mask, i] <= sigma * Y_std[mask, i]

                rows.append(
                    {
                        group_param: value,
                        "target": target,
                        "sigma": sigma,
                        "empirical_coverage": inside.mean(),
                        "mean_pred_std": Y_std[mask, i].mean(),
                        "mae": abs_err[mask, i].mean(),
                        "count": mask.sum(),
                    }
                )

        for sigma in sigmas:
            inside_all = abs_err[mask] <= sigma * Y_std[mask]

            rows.append(
                {
                    group_param: value,
                    "target": "all",
                    "sigma": sigma,
                    "empirical_coverage": inside_all.mean(),
                    "mean_pred_std": Y_std[mask].mean(),
                    "mae": abs_err[mask].mean(),
                    "count": mask.sum() * Y_true.shape[1],
                }
            )

    return pd.DataFrame(rows)


def uncertainty_error_correlation(
    Y_true,
    Y_mean,
    Y_std,
    target_names=PARAM_NAMES,
    eps=1e-8,
):
    """
    Check whether larger predicted uncertainty corresponds to larger error.

    Returns Pearson and Spearman correlations between:
        predicted std
        absolute error

    A useful uncertainty model should usually have positive correlation.
    """
    Y_true = np.asarray(Y_true)
    Y_mean = np.asarray(Y_mean)
    Y_std = np.asarray(Y_std)

    abs_err = np.abs(Y_true - Y_mean)
    Y_std = np.maximum(Y_std, eps)

    rows = []

    for i, name in enumerate(target_names):
        s = pd.Series(Y_std[:, i])
        e = pd.Series(abs_err[:, i])

        rows.append(
            {
                "target": name,
                "pearson_std_abs_error": s.corr(e, method="pearson"),
                "spearman_std_abs_error": s.corr(e, method="spearman"),
                "mean_pred_std": Y_std[:, i].mean(),
                "mae": abs_err[:, i].mean(),
            }
        )

    s_all = pd.Series(Y_std.reshape(-1))
    e_all = pd.Series(abs_err.reshape(-1))

    rows.append(
        {
            "target": "all",
            "pearson_std_abs_error": s_all.corr(e_all, method="pearson"),
            "spearman_std_abs_error": s_all.corr(e_all, method="spearman"),
            "mean_pred_std": Y_std.mean(),
            "mae": abs_err.mean(),
        }
    )

    return pd.DataFrame(rows)