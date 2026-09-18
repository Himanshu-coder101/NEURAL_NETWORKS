#!/usr/bin/env python3

"""
BITS F445 - Programming Assignment 1

Check the format of a submitted JSON model and report its MSE on
validation.csv.

The script assumes that the following files are available:
    - train.csv
    - validation.csv
    - the submitted JSON model

The feature means and population standard deviations are computed from
train.csv only, in accordance with the assignment instructions.

Usage:
    python check_model_json.py GRP000_2027A7PS0000.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


NUM_FEATURES = 100
FEATURE_COLUMNS = [f"x{i}" for i in range(1, NUM_FEATURES + 1)]
TARGET_COLUMN = "electricity_consumption_kwh"


def is_number(value):
    """Return True for a finite int/float, but not for bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def load_and_validate_model(json_path):
    """Load the JSON file and validate the required format."""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            model = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"JSON file not found: {json_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON syntax at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        )

    if not isinstance(model, dict):
        raise ValueError("Top-level JSON value must be an object.")

    required_keys = {"weights", "bias"}
    actual_keys = set(model.keys())

    missing = required_keys - actual_keys
    extra = actual_keys - required_keys

    if missing:
        raise ValueError(
            "Missing required JSON key(s): " + ", ".join(sorted(missing))
        )

    if extra:
        raise ValueError(
            "Unexpected JSON key(s): " + ", ".join(sorted(extra))
        )

    weights = model["weights"]
    bias = model["bias"]

    if not isinstance(weights, list):
        raise ValueError('"weights" must be a one-dimensional JSON list.')

    if len(weights) != NUM_FEATURES:
        raise ValueError(
            f'"weights" must contain exactly {NUM_FEATURES} values; '
            f"found {len(weights)}."
        )

    # Reject nested lists and non-numerical/non-finite entries.
    for i, value in enumerate(weights, start=1):
        if not is_number(value):
            raise ValueError(
                f'weights[{i - 1}] must be a finite numerical value.'
            )

    if not is_number(bias):
        raise ValueError('"bias" must be a single finite numerical value.')

    weights = np.asarray(weights, dtype=np.float64)
    bias = float(bias)

    return weights, bias


def load_dataset(csv_path, dataset_name):
    """Load and validate one CSV dataset."""
    if not csv_path.exists():
        raise ValueError(f"{dataset_name} file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [c for c in required_columns if c not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required column(s): "
            + ", ".join(missing_columns)
        )

    try:
        x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            f"{dataset_name} contains non-numerical values in required columns."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            f"{dataset_name} contains missing or non-finite feature values."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{dataset_name} contains missing or non-finite target values."
        )

    return x, y


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check a BITS F445 PA1 JSON model and report validation MSE."
        )
    )
    parser.add_argument(
        "model_json",
        type=Path,
        help="Path to the submitted JSON model file.",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("train.csv"),
        help="Training CSV used to compute feature means/stds "
             "(default: train.csv).",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("validation.csv"),
        help="Validation CSV used for evaluation "
             "(default: validation.csv).",
    )

    args = parser.parse_args()

    try:
        weights, bias = load_and_validate_model(args.model_json)

        x_train, _ = load_dataset(args.train, "Training")
        x_validation, y_validation = load_dataset(
            args.validation, "Validation"
        )

        # Fixed preprocessing rule from the assignment:
        # compute these statistics ONCE from the complete training set.
        feature_mean = x_train.mean(axis=0)
        feature_std = x_train.std(axis=0, ddof=0)

        zero_std_features = [
            FEATURE_COLUMNS[i]
            for i, std in enumerate(feature_std)
            if std == 0.0
        ]
        if zero_std_features:
            raise ValueError(
                "The following training features have zero standard "
                "deviation and cannot be standardized: "
                + ", ".join(zero_std_features)
            )

        x_validation_std = (
            x_validation - feature_mean
        ) / feature_std

        predictions = x_validation_std @ weights + bias

        mse = float(
            np.mean((predictions - y_validation) ** 2)
        )

    except ValueError as exc:
        print("Model check: FAILED")
        print(f"Reason: {exc}")
        raise SystemExit(1)

    print("Model check: PASSED")
    print(f"JSON file       : {args.model_json}")
    print(f"Number of weights: {len(weights)}")
    print(f"Bias            : {bias:.6f}")
    print(f"Validation rows : {len(y_validation)}")
    print(f"Validation MSE  : {mse:.6f}")


if __name__ == "__main__":
    main()
