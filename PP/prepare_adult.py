#!/usr/bin/env python3
"""
Prepare cleaned + balanced UCI Adult / Census Income dataset for TabPFN.

Dataset:
  UCI Adult / Census Income

Processing:
  1. Download official UCI train + test files.
  2. Combine them into one dataset.
  3. Remove rows with missing categorical values marked as '?'.
  4. Encode categorical features as integer category codes.
  5. Encode income labels:
       <=50K = 0
       >50K  = 1
  6. Balance labels by downsampling the majority class.
  7. Save full balanced arrays and a stratified train/test split.

Expected cleaned + balanced size:
  22,416 total rows
    11,208 <=50K
    11,208 >50K

Default split:
  10,000 train rows
  12,416 test rows

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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


UCI_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)

UCI_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

FEATURE_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]

TARGET_COLUMN = "income"

TARGET_NAMES = np.array(["<=50K", ">50K"])

CATEGORICAL_FEATURE_INDICES = np.array(
    [FEATURE_NAMES.index(col) for col in CATEGORICAL_FEATURES],
    dtype=np.int64,
)


def download_bytes(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 adult-tabpfn-prep"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def read_adult_csv(raw: bytes | str, is_test: bool) -> pd.DataFrame:
    """
    Read adult.data or adult.test.

    adult.test has a first metadata line beginning with '|', and labels end
    with a period, e.g. '<=50K.'.
    """
    if isinstance(raw, bytes):
        source = io.BytesIO(raw)
    else:
        source = raw

    df = pd.read_csv(
        source,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        comment="|",
        na_values=["?"],
    )

    # Strip whitespace from string columns.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # adult.test has labels with trailing periods.
    if is_test:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].str.replace(
            r"\.$",
            "",
            regex=True,
        )

    return df


def load_raw_data(
    raw_train_path: str | None = None,
    raw_test_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Adult train/test files from either local paths or official UCI URLs.
    """
    if raw_train_path is not None and raw_test_path is not None:
        train_df = read_adult_csv(raw_train_path, is_test=False)
        test_df = read_adult_csv(raw_test_path, is_test=True)
        return train_df, test_df

    if raw_train_path is not None or raw_test_path is not None:
        raise ValueError(
            "Provide both --raw-train-path and --raw-test-path, or neither."
        )

    train_bytes = download_bytes(UCI_TRAIN_URL)
    test_bytes = download_bytes(UCI_TEST_URL)

    train_df = read_adult_csv(train_bytes, is_test=False)
    test_df = read_adult_csv(test_bytes, is_test=True)

    return train_df, test_df


def clean_no_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop rows with missing values and validate labels/features.
    """
    n_before = len(df)

    # Convert '?' and accidental string 'nan' cases to actual missing.
    df = df.replace("?", np.nan)
    df = df.replace("nan", np.nan)

    missing_before_drop = {
        col: int(df[col].isna().sum())
        for col in COLUMNS
    }

    df = df.dropna(axis=0).reset_index(drop=True)

    n_after = len(df)

    # Normalize label values.
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.replace(r"\.$", "", regex=True)
    )

    valid_labels = set(TARGET_NAMES.tolist())
    bad_labels = sorted(set(df[TARGET_COLUMN]) - valid_labels)

    if bad_labels:
        raise ValueError(f"Unexpected income labels found: {bad_labels}")

    # Numeric conversion.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Categorical cleanup.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

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


def balance_labels(
    df: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Downsample majority class to the minority class count.
    """
    label_counts_before = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    min_count = int(df[TARGET_COLUMN].value_counts().min())

    balanced_df = (
        df.groupby(TARGET_COLUMN, group_keys=False)
        .sample(n=min_count, random_state=random_state)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    label_counts_after = (
        balanced_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    metadata = {
        "balance_method": "downsample majority class to minority class count",
        "class_counts_before_balance": {
            str(k): int(v) for k, v in label_counts_before.items()
        },
        "minority_class_count": int(min_count),
        "class_counts_after_balance": {
            str(k): int(v) for k, v in label_counts_after.items()
        },
        "n_rows_after_balance": int(len(balanced_df)),
    }

    return balanced_df, metadata


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
        "<=50K": 0,
        ">50K": 1,
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
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved cleaned balanced Adult arrays to: {out_dir.resolve()}")
    print(f"X:       {X.shape}, y:       {y.shape}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print("Class counts full:")
    print(f"  <=50K: {int(class_counts_full[0])}")
    print(f"  >50K:  {int(class_counts_full[1])}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-train-path",
        default=None,
        help="Optional local path to adult.data.",
    )
    parser.add_argument(
        "--raw-test-path",
        default=None,
        help="Optional local path to adult.test.",
    )
    parser.add_argument(
        "--out-dir",
        default="adult_balanced_npy",
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=10_000,
        help="Number of training rows. Default: 10000.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    raw_train_df, raw_test_df = load_raw_data(
        raw_train_path=args.raw_train_path,
        raw_test_path=args.raw_test_path,
    )

    full_df = pd.concat([raw_train_df, raw_test_df], axis=0, ignore_index=True)

    clean_df, clean_metadata = clean_no_missing(full_df)

    balanced_df, balance_metadata = balance_labels(
        clean_df,
        random_state=args.random_state,
    )

    X, y, encoding_metadata = encode_features_and_target(balanced_df)

    metadata = {
        "dataset": "UCI Adult / Census Income",
        "alias": "adult",
        "official_raw_rows": {
            "train": int(len(raw_train_df)),
            "test": int(len(raw_test_df)),
            "total": int(len(full_df)),
        },
        **clean_metadata,
        **balance_metadata,
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