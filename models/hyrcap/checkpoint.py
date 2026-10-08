"""Validation for the released unified checkpoint."""

import math
from pathlib import Path

import torch

FORMAT = "hyrcap-full-v1"
MODULES = {"cpcc": True, "lqe": True, "prs": True}


def validate_full_checkpoint(checkpoint):
    """Reject incomplete or incompatible files instead of disabling modules."""
    from .calibration import FEATURE_NAMES, FEATURE_VERSION

    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(
            "Full evaluation requires a unified HyrCap checkpoint containing detector, CPCC, LQE, and PRS weights."
        )
    for key in ("detector", "calibrator"):
        values = checkpoint.get(key)
        if (
            not isinstance(values, dict)
            or not values
            or not all(torch.is_tensor(v) for v in values.values())
        ):
            raise ValueError(
                f"Invalid or missing {key} state in the unified checkpoint."
            )
    if (
        checkpoint.get("feature_version") != FEATURE_VERSION
        or checkpoint.get("feature_names") != FEATURE_NAMES
    ):
        raise ValueError(
            "Unified checkpoint has an incompatible calibration feature contract."
        )
    if checkpoint.get("modules_enabled") != MODULES:
        raise ValueError("Unified checkpoint must enable CPCC, LQE, and PRS.")
    for name in MODULES:
        if not any(k.startswith(name + ".") for k in checkpoint["calibrator"]):
            raise ValueError(f"Missing {name} weights in the unified checkpoint.")
    config = checkpoint.get("calibration_config")
    if not isinstance(config, dict) or not {"hidden", "dropout"} <= config.keys():
        raise ValueError("Missing calibration architecture configuration.")
    scoring = checkpoint.get("calibration_scoring", {})
    for key in ("cpcc_beta", "lqe_gamma", "prs_gamma"):
        value = scoring.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Unified checkpoint requires a positive finite {key}.")
    return checkpoint


def save_full_checkpoint(
    path, detector_state, calibrator, training_config, scoring, metadata=None
):
    """Save detector and calibration weights without overwriting an existing file."""
    from .calibration import FEATURE_NAMES, FEATURE_VERSION

    checkpoint = {
        "format": FORMAT,
        "detector": {
            key: value.detach().cpu() for key, value in detector_state.items()
        },
        "calibrator": {
            key: value.detach().cpu() for key, value in calibrator.state_dict().items()
        },
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "calibration_config": dict(training_config),
        "calibration_scoring": dict(scoring),
        "modules_enabled": dict(MODULES),
        "metadata": dict(metadata or {}),
    }
    validate_full_checkpoint(checkpoint)
    path = Path(path)
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
    return path
