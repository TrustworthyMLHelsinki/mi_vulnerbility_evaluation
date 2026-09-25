import argparse
import os
import pickle
import warnings
from typing import Union

# import time
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from tqdm import tqdm
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prob_to_score(prob: Union[np.ndarray, float]):
    return np.log(prob / (1 - prob))


def load_adult_csv(csv_file):
    data = pd.read_csv(csv_file, na_values="?")

    feature_columns = [c for c in data.columns if c not in {"income", "index"}]
    X = data[feature_columns]
    y = data["income"].map({"<=50K": 0, ">50K": 1}).to_numpy(dtype=np.int64)

    return X, y


def find_frame_for_nplus(subsets, N_plus):
    """Return the saved finite frame whose size is exactly N_plus."""
    for ratio, size in subsets["frame_sizes_by_ratio"].items():
        if int(size) == int(N_plus):
            return float(ratio), np.asarray(
                subsets["X_superset_indices_by_ratio"][ratio], dtype=np.int64
            )

    available = sorted(int(v) for v in subsets["frame_sizes_by_ratio"].values())
    raise ValueError(
        f"N_plus={N_plus} was not saved. Available N_plus values: {available}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results", required=True, help="Directory for model statistics/results.")
    parser.add_argument("--fpc_data_dir", required=True, help="Directory containing the saved fpc_data_subsets.pkl.")
    parser.add_argument("--dataset", required=True, help="Dataset name used in the subset/result directory structure.")
    parser.add_argument("--csv_file", required=True, help="Path to the Adult CSV file.")
    parser.add_argument("--training_sample_size", type=int, default=200, help="Per-model training-set size N.")
    parser.add_argument("--target_ratio", type=float, required=True, help="Finite-frame N/N_plus for this run.")
    parser.add_argument("--M_shadow", type=int, default=1000, help="Total number of shadow models for this N_plus.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--stop_idx", type=int, default=None)

    args = parser.parse_args()

    subset_path = os.path.join(
        args.fpc_data_dir,
        args.dataset,
        f"Seed={args.seed}",
        f"N={args.training_sample_size}",
        "fpc_data_subsets.pkl",
    )
    with open(subset_path, "rb") as f:
        subsets = pickle.load(f)

    if args.training_sample_size != int(subsets["training_sample_size"]):
        raise ValueError(
            "training_sample_size does not match the value used to create the saved subsets."
        )

    N_plus = int(round(args.training_sample_size / args.target_ratio))
    ratio, frame_indices = find_frame_for_nplus(subsets, N_plus)
    eval_indices = np.asarray(subsets["X_eval_indices"], dtype=np.int64)

    if args.training_sample_size > len(frame_indices):
        raise ValueError("training_sample_size cannot exceed N_plus.")

    stop_idx = args.M_shadow if args.stop_idx is None else args.stop_idx
    if not (0 <= args.start_idx < stop_idx <= args.M_shadow):
        raise ValueError("Require 0 <= start_idx < stop_idx <= M_shadow.")

    X, y = load_adult_csv(args.csv_file)

    X_eval = X.iloc[eval_indices]
    y_eval = y[eval_indices]

    N_models = stop_idx - args.start_idx
    N_eval = len(eval_indices)

    target_stats = np.empty((N_models, N_eval), dtype=np.float32)
    target_membership = np.zeros((N_models, N_eval), dtype=bool)

    target_lookup = np.full(len(X), -1, dtype=np.int64)
    target_lookup[eval_indices] = np.arange(N_eval)

    for local_i, model_idx in enumerate(
        tqdm(range(args.start_idx, stop_idx), desc="Shadow models")
    ):
        rng = np.random.default_rng(args.seed + model_idx)
        train_indices = rng.choice(
            frame_indices,
            size=args.training_sample_size,
            replace=False,
        )
        train_indices = np.sort(train_indices)

        target_cols = target_lookup[train_indices]
        target_cols = target_cols[target_cols >= 0]
        target_membership[local_i, target_cols] = True

        # t0 = time.perf_counter()

        model = TabPFNClassifier(device=DEVICE, n_estimators=1)
        
        # t1 = time.perf_counter()

        model.fit(X.iloc[train_indices], y[train_indices])

        # t2 = time.perf_counter()

        predicted_probs = model.predict_proba(X_eval)
        
        # t3 = time.perf_counter()

        true_class_probs = np.take_along_axis(
            predicted_probs,
            y_eval.reshape((-1, 1)),
            axis=1,
        ).reshape(-1)
        target_stats[local_i] = prob_to_score(true_class_probs)

        # print(
        #     f"Model {model_idx}: "
        #     f"init={t1 - t0:.3f}s, "
        #     f"fit={t2 - t1:.3f}s, "
        #     f"predict={t3 - t2:.3f}s, "
        #     f"total={t3 - t0:.3f}s"
        # )
    output_dir = os.path.join(
        args.results,
        args.dataset,
        f"Seed={args.seed}",
        f"N={args.training_sample_size}",
        f"ratio={args.target_ratio}",
    )
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "dataset": args.dataset,
        "model": "TabPFN",
        "csv_file": args.csv_file,
        "seed": args.seed,
        "N": args.training_sample_size,
        "N_plus": N_plus,
        "ratio": ratio,
        "M_shadow": args.M_shadow,
        "start_idx": args.start_idx,
        "stop_idx": stop_idx,
        "X_eval_indices": eval_indices,
        "target_stats": target_stats,
        "target_membership": target_membership,
    }

    output_path = os.path.join(
        output_dir,
        f"shadow_{args.start_idx}_{stop_idx}.pkl",
    )
    with open(output_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {output_path}")
    print(f"N_plus={N_plus}, N/N_plus={ratio:.4f}")
    print(f"Models trained: {N_models}")
    print(f"Mean target IN fraction in this chunk: {target_membership.mean():.4f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
