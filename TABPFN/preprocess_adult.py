"""Combine the downloaded Adult splits into a csv for TabPFN."""

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]


def main():
    raw_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=raw_dir, help="Directory containing adult.data and adult.test.")
    parser.add_argument("--output", type=Path, default=raw_dir.parent / "data" / "adult" / "adult.csv", help="Output CSV used by the TabPFN experiments.")
    args = parser.parse_args()

    splits = [
        pd.read_csv(args.input_dir / name, names=COLUMNS, skipinitialspace=True, na_values="?", comment="|")
        for name in ("adult.data", "adult.test")
    ]
    data = pd.concat(splits, ignore_index=True)
    n_raw = len(data)
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].str.strip()
    data["income"] = data["income"].str.removesuffix(".")
    data = data.dropna().reset_index(drop=True)

    # Preserve source order, all 14 features, duplicates, and natural class counts.
    # Train/test splitting for the experiments is performed downstream.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Read {n_raw} rows; removed {n_raw - len(data)} rows with missing values.")
    print(f"Saved {len(data)} rows to {args.output}")
    print(data["income"].value_counts().to_string())


if __name__ == "__main__":
    main()
