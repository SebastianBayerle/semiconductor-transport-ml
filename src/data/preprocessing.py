import torch


def signed_log10(x, eps=1):
    return torch.sign(x) * torch.log10(1.0 + torch.abs(x) / eps)



def build_curve_tensor(
    dataset,
    curve_names,
    log=False,
    flatten=True,
    add_first_derivative=False,
    add_second_derivative=False,
):
    """
    Build X or Y tensor from selected transport curves.

    Derivatives are computed per curve, never across curve boundaries.

    Returns:
        flatten=True:  shape (N, features)
        flatten=False: shape (N, channels, length)
    """
    curves = dataset.get_curves().clone()  # (N, 4, 100)

    curve_idx = [dataset.curve_names.index(name) for name in curve_names]
    X = curves[:, curve_idx, :]            # (N, C, 100)

    if log:
        X = signed_log10(X)

    parts = [X]

    if add_first_derivative or add_second_derivative:
        d1 = torch.diff(X, dim=2)          # (N, C, 99)

        if add_first_derivative:
            parts.append(d1)

        if add_second_derivative:
            d2 = torch.diff(d1, dim=2)     # (N, C, 98)
            parts.append(d2)

    if flatten:
        parts = [p.reshape(p.shape[0], -1) for p in parts]
        X = torch.cat(parts, dim=1)
    else:
        # Only possible if all parts have the same length, which derivatives do not.
        # So keep this strict to avoid silent shape bugs.
        if len(parts) > 1:
            raise ValueError(
                "flatten=False with derivative features is ambiguous because "
                "raw curves, first derivatives, and second derivatives have different lengths."
            )
        X = parts[0]

    return X
    


def build_parameter_tensor(dataset, param_names, log_params=()):
    """
    Build tensor from selected physical parameters.
    """
    params = dataset.get_parameters().clone()  # (N, 7)

    param_idx = [dataset.param_names.index(name) for name in param_names]
    out = params[:, param_idx]

    for name in log_params:
        if name in param_names:
            local_idx = param_names.index(name)
            out[:, local_idx] = torch.log10(out[:, local_idx])

    return out


def build_classification_tensor(dataset, param="accdon", map_pm1_to_01=True):
    """
    Build classification target from a parameter, usually accdon.

    For accdon:
        -1 -> 0
        +1 -> 1
    """
    params = dataset.get_parameters().clone()

    idx = dataset.param_names.index(param)
    y = params[:, idx]

    if map_pm1_to_01:
        y = ((y + 1) / 2).long()
    else:
        y = y.long()

    return y


def build_tensor_from_spec(dataset, spec):
    """
    Build a tensor according to a config/spec dictionary.

    Examples
    --------
    Curve spec:
        {
            "type": "curves",
            "curves": ["L12B"],
            "log": True,
            "flatten": True
        }

    Parameter spec:
        {
            "type": "parameters",
            "params": ["t", "alpha", "delta"],
            "log_params": ["gamma", "eweight"]
        }

    Classification spec:
        {
            "type": "classification",
            "param": "accdon",
            "map_pm1_to_01": True
        }
    """
    spec_type = spec["type"]

    if spec_type == "curves":
        return build_curve_tensor(
            dataset,
            curve_names=spec["curves"],
            log=spec.get("log", False),
            flatten=spec.get("flatten", True),
            add_first_derivative=spec.get("add_first_derivative", False),
            add_second_derivative=spec.get("add_second_derivative", False),
        )

    if spec_type == "parameters":
        return build_parameter_tensor(
            dataset,
            param_names=spec["params"],
            log_params=spec.get("log_params", ()),
        )

    if spec_type == "classification":
        return build_classification_tensor(
            dataset,
            param=spec.get("param", "accdon"),
            map_pm1_to_01=spec.get("map_pm1_to_01", True),
        )

    raise ValueError(f"Unknown spec type: {spec_type}")


def apply_filters(dataset, X, Y, filters=None):
    """
    Filter X and Y based on raw dataset parameters.

    Example:
        filters = {"accdon": 1.0}
        filters = {"accdon": -1.0, "eweight": -2.0}
    """
    if not filters:
        return X, Y

    params = dataset.get_parameters()
    mask = torch.ones(len(params), dtype=torch.bool)

    for name, value in filters.items():
        idx = dataset.param_names.index(name)
        mask &= params[:, idx] == value

    return X[mask], Y[mask]


def add_derivative_features(X, first=False, second=False):
    """
    Add first/second finite-difference features.

    Expects flattened input of shape (N, features).
    """
    parts = [X]

    if first or second:
        d1 = torch.diff(X, dim=1)

        if first:
            parts.append(d1)

        if second:
            d2 = torch.diff(d1, dim=1)
            parts.append(d2)

    return torch.cat(parts, dim=1)


def build_xy_from_specs(dataset, input_spec, target_spec, filters=None):
    """
    Main high-level helper.

    Builds arbitrary X/Y pairs from raw TransportDataset.

    Examples:
        L12B -> parameters
        L12B -> accdon
        L11  -> L12
        all curves -> parameters
    """
    X = build_tensor_from_spec(dataset, input_spec)
    Y = build_tensor_from_spec(dataset, target_spec)

    X, Y = apply_filters(dataset, X, Y, filters=filters)
    return X, Y