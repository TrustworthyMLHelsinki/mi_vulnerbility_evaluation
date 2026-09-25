#!/usr/bin/env python3
"""
Prepare cleaned German Credit / credit-g dataset arrays for TabPFN-style use.

Dataset:
  Statlog German Credit Data, available from UCI and OpenML as credit-g.

German Credit dataset:
  1,000 total rows
    700 good credit risks
    300 bad credit risks

Default split:
  800 train rows
  200 test rows

Outputs:
  X.npy
  y.npy
  X_train.npy
  X_test.npy
  y_train.npy
  y_test.npy
  feature_names.npy
  target_names.npy
  categorical_feature_indices.npy
  metadata.json
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


OPENML_DATASET_NAME = "credit-g"
OPENML_DATA_ID = 31

UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

FEATURE_NAMES = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
]

TARGET_COLUMN = "class"

COLUMNS = FEATURE_NAMES + [TARGET_COLUMN]

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    col for col in FEATURE_NAMES if col not in NUMERIC_FEATURES
]

TARGET_NAMES = np.array(["good", "bad"])

CATEGORICAL_FEATURE_INDICES = np.array(
    [FEATURE_NAMES.index(col) for col in CATEGORICAL_FEATURES],
    dtype=np.int64,
)


def download_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 german-credit-tabpfn-prep"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def read_uci_german_credit(raw: bytes | str | Path) -> pd.DataFrame:
    """
    Read the original UCI german.data format.

    The UCI file has 20 whitespace-separated feature columns plus a target
    column encoded as:
      1 = good credit risk
      2 = bad credit risk
    """
    if isinstance(raw, bytes):
        source: Any = io.BytesIO(raw)
    else:
        source = raw

    df = pd.read_csv(
        source,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        na_values=["?", "nan", "NaN", ""],
        engine="python",
    )

    return df


def load_openml_data(
    data_id: int = OPENML_DATA_ID,
    dataset_name: str = OPENML_DATASET_NAME,
) -> tuple[pd.DataFrame, dict]:
    """
    Load credit-g from OpenML.

    data_id=31 is used by default for reproducibility. dataset_name is only
    recorded in metadata and helps document the expected OpenML dataset.
    """
    dataset = fetch_openml(data_id=data_id, as_frame=True, parser="auto")

    if dataset.frame is None:
        raise ValueError("OpenML did not return a pandas frame.")

    frame = dataset.frame.copy()

    if dataset.target is None:
        raise ValueError("OpenML dataset did not include a target column.")

    # fetch_openml usually returns X and y separately. Use those when present
    # to avoid relying on the exact target-column name in the returned frame.
    if dataset.data is not None:
        X_df = dataset.data.copy()
    else:
        X_df = frame.drop(columns=[dataset.target.name])

    y = pd.Series(dataset.target).copy()

    if len(X_df.columns) != len(FEATURE_NAMES):
        raise ValueError(
            "Unexpected number of OpenML feature columns: "
            f"expected {len(FEATURE_NAMES)}, got {len(X_df.columns)}. "
            "Use --source uci or --raw-path for the classic UCI format."
        )

    # The OpenML names for credit-g commonly match FEATURE_NAMES. Rename by
    # position to keep output schema stable across small metadata variations.
    X_df.columns = FEATURE_NAMES

    df = X_df.copy()
    df[TARGET_COLUMN] = y.to_numpy()

    metadata = {
        "source": "openml",
        "openml_dataset_name_requested": dataset_name,
        "openml_data_id": int(data_id),
        "openml_name_returned": str(getattr(dataset, "details", {}).get("name", "")),
        "openml_version": str(getattr(dataset, "details", {}).get("version", "")),
    }

    return df, metadata


def load_raw_data(
    source: str,
    raw_path: str | None = None,
    openml_data_id: int = OPENML_DATA_ID,
) -> tuple[pd.DataFrame, dict]:
    """
    Load German Credit from OpenML, UCI URL, or a local UCI-format file.
    """
    if raw_path is not None:
        df = read_uci_german_credit(Path(raw_path))
        metadata = {
            "source": "local_uci_format",
            "raw_path": str(raw_path),
        }
        return df, metadata

    if source == "openml":
        return load_openml_data(data_id=openml_data_id)

    if source == "uci":
        raw = download_bytes(UCI_DATA_URL)
        df = read_uci_german_credit(raw)
        metadata = {
            "source": "uci",
            "uci_data_url": UCI_DATA_URL,
        }
        return df, metadata

    raise ValueError(f"Unknown source: {source!r}")


def clean_no_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop rows with missing values and validate labels/features.
    """
    n_before = len(df)

    missing_markers = ["?", "nan", "NaN", "None", ""]
    df = df.replace(missing_markers, np.nan)

    missing_before_drop = {
        col: int(df[col].isna().sum())
        for col in COLUMNS
    }

    df = df.dropna(axis=0).reset_index(drop=True)
    n_after = len(df)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    df[TARGET_COLUMN] = normalize_target_series(df[TARGET_COLUMN])

    metadata = {
        "n_rows_before_missing_drop": int(n_before),
        "n_rows_after_missing_drop": int(n_after),
        "n_rows_dropped_missing": int(n_before - n_after),
        "missing_values_before_drop": missing_before_drop,
        "missing_values_after_drop": {
            col: int(df[col].isna().sum())
            for col in COLUMNS
        },
    }

    return df, metadata


def normalize_target_series(y: pd.Series) -> pd.Series:
    """
    Normalize UCI/OpenML target encodings to {'good', 'bad'}.
    """
    normalized = y.astype(str).str.strip().str.lower()

    label_map = {
        "1": "good",
        "good": "good",
        "2": "bad",
        "bad": "bad",
    }

    normalized = normalized.map(label_map)

    if normalized.isna().any():
        bad_values = sorted(set(y.astype(str)) - set(label_map.keys()))
        raise ValueError(f"Unexpected credit labels found: {bad_values}")

    return normalized


def encode_features_and_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Encode categorical features as deterministic integer codes.

    Numeric features are preserved as numeric values.
    Categorical features are encoded alphabetically by category.
    """
    encoded = df[FEATURE_NAMES].copy()

    category_maps: dict[str, dict[str, int]] = {}

    for col in CATEGORICAL_FEATURES:
        categories = sorted(encoded[col].astype(str).unique().tolist())
        mapping = {category: idx for idx, category in enumerate(categories)}

        encoded[col] = encoded[col].map(mapping).astype(np.int64)
        category_maps[col] = {
            category: int(idx)
            for category, idx in mapping.items()
        }

    for col in NUMERIC_FEATURES:
        encoded[col] = pd.to_numeric(encoded[col], errors="raise")

    label_map = {
        "good": 0,
        "bad": 1,
    }

    X = encoded[FEATURE_NAMES].astype(np.float32).to_numpy()
    y = df[TARGET_COLUMN].map(label_map).astype(np.int64).to_numpy()

    metadata = {
        "feature_encoding": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "categorical_feature_indices": CATEGORICAL_FEATURE_INDICES.tolist(),
            "categorical_encoding": "alphabetical integer codes per feature",
            "category_maps": category_maps,
        },
        "target_encoding": label_map,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES.tolist(),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(TARGET_NAMES)),
        "feature_dtype": str(X.dtype),
        "target_dtype": str(y.dtype),
    }

    return X, y, metadata


def save_arrays(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    train_size: int,
    random_state: int,
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if train_size <= 0 or train_size >= len(y):
        raise ValueError(
            f"train_size must be between 1 and {len(y) - 1}; got {train_size}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        stratify=y,
        random_state=random_state,
    )

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "feature_names.npy", np.array(FEATURE_NAMES))
    np.save(out_dir / "target_names.npy", TARGET_NAMES)
    np.save(out_dir / "categorical_feature_indices.npy", CATEGORICAL_FEATURE_INDICES)

    class_counts_full = np.bincount(y, minlength=2).astype(int)
    class_counts_train = np.bincount(y_train, minlength=2).astype(int)
    class_counts_test = np.bincount(y_test, minlength=2).astype(int)

    metadata = {
        **metadata,
        "split": {
            "method": "sklearn.model_selection.train_test_split",
            "stratified": True,
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "random_state": int(random_state),
        },
        "shapes": {
            "X": list(X.shape),
            "y": list(y.shape),
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.shape),
            "y_test": list(y_test.shape),
        },
        "class_counts": {
            "full": {
                TARGET_NAMES[i]: int(class_counts_full[i])
                for i in range(len(TARGET_NAMES))
            },
            "train": {
                TARGET_NAMES[i]: int(class_counts_train[i])
                for i in range(len(TARGET_NAMES))
            },
            "test": {
                TARGET_NAMES[i]: int(class_counts_test[i])
                for i in range(len(TARGET_NAMES))
            },
        },
        "label_balancing": {
            "applied": False,
            "reason": "preserve natural German Credit class distribution",
        },
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved cleaned German Credit arrays to: {out_dir.resolve()}")
    print(f"X:       {X.shape}, y:       {y.shape}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print("Class counts full:")
    print(f"  good: {int(class_counts_full[0])}")
    print(f"  bad:  {int(class_counts_full[1])}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=["openml", "uci"],
        default="openml",
        help="Dataset source when --raw-path is not provided. Default: openml.",
    )
    
    parser.add_argument(
        "--openml-data-id",
        type=int,
        default=OPENML_DATA_ID,
        help="OpenML data_id for credit-g. Default: 31.",
    )
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Optional local path to UCI-format german.data. Overrides --source.",
    )
    parser.add_argument(
        "--out-dir",
        default="credit_npy",
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=800,
        help="Number of training rows. Default: 800.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    raw_df, source_metadata = load_raw_data(
        source=args.source,
        raw_path=args.raw_path,
        openml_data_id=args.openml_data_id,
    )

    clean_df, clean_metadata = clean_no_missing(raw_df)

    X, y, encoding_metadata = encode_features_and_target(clean_df)

    label_counts = (
        clean_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    metadata = {
        "dataset": "Statlog German Credit Data / OpenML credit-g",
        "alias": "credit-g",
        "official_expected_rows": 1000,
        "official_expected_features": 20,
        "class_distribution_preserved": True,
        "raw_rows": int(len(raw_df)),
        "class_counts_before_split": {
            str(k): int(v) for k, v in label_counts.items()
        },
        **source_metadata,
        **clean_metadata,
        **encoding_metadata,
    }

    save_arrays(
        X=X,
        y=y,
        out_dir=Path(args.out_dir),
        train_size=args.train_size,
        random_state=args.random_state,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
