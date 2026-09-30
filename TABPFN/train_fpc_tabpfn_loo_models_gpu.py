import argparse
import os
import pickle
import warnings
from typing import Union

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from tqdm import tqdm


def prob_to_score(prob: Union[np.ndarray, float]):
    return np.log(prob / (1 - prob))


def load_adult_csv(csv_file):
    data = pd.read_csv(csv_file, na_values="?")

    feature_columns = [c for c in data.columns if c not in {"income", "index"}]
    X = data[feature_columns]
    y = data["income"].map({"<=50K": 0, ">50K": 1}).to_numpy(dtype=np.int64)

    return X, y


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results", required=True, help="Directory for model statistics/results.")
    parser.add_argument("--data_directory", required=True, help="Directory containing fpc_data_subsets.pkl.")
    parser.add_argument("--dataset", required=True, help="Dataset name used in the subset/result directory structure.")
    parser.add_argument("--csv_file", required=True, help="Path to the Adult CSV file.")

    parser.add_argument("--training_sample_size", type=int, default=200, help="Per-model training-set size N.")
    parser.add_argument("--M_pop", type=int, default=1000, help="Total population LOO models per evaluation target. Must be even.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--target_start_idx", type=int, default=0)
    parser.add_argument("--target_stop_idx", type=int, default=None)
    parser.add_argument("--model_start_idx", type=int, default=0)
    parser.add_argument("--model_stop_idx", type=int, default=None)

    args = parser.parse_args()

    if args.M_pop % 2 != 0:
        raise ValueError("M_pop must be even so that half the models are IN and half are OUT.")

    subset_path = os.path.join(
        args.data_directory,
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

    eval_indices = np.asarray(subsets["X_eval_indices"], dtype=np.int64)
    population_indices = np.asarray(subsets["X_population_indices"], dtype=np.int64)

    N_eval = len(eval_indices)
    target_stop_idx = N_eval if args.target_stop_idx is None else args.target_stop_idx
    if not (0 <= args.target_start_idx < target_stop_idx <= N_eval):
        raise ValueError("Require 0 <= target_start_idx < target_stop_idx <= len(X_eval).")

    model_stop_idx = args.M_pop if args.model_stop_idx is None else args.model_stop_idx
    if not (0 <= args.model_start_idx < model_stop_idx <= args.M_pop):
        raise ValueError("Require 0 <= model_start_idx < model_stop_idx <= M_pop.")

    X, y = load_adult_csv(args.csv_file)

    selected_target_positions = np.arange(args.target_start_idx, target_stop_idx)
    selected_model_indices = np.arange(args.model_start_idx, model_stop_idx)

    n_targets_job = len(selected_target_positions)
    n_models_job = len(selected_model_indices)

    target_stats = np.empty((n_targets_job, n_models_job), dtype=np.float32)
    target_membership = np.zeros((n_targets_job, n_models_job), dtype=bool)

    M_pop_in = args.M_pop // 2

    for local_t, target_pos in enumerate(
        tqdm(selected_target_positions, desc="Targets")
    ):
        target_dataset_idx = int(eval_indices[target_pos])
        target_x = X.iloc[[target_dataset_idx]]
        target_y = int(y[target_dataset_idx])

        # LOO population for this target: the target itself must not be among
        # the background samples used by either IN or OUT models.
        target_population_indices = population_indices[
            population_indices != target_dataset_idx
        ]

        if args.training_sample_size > len(target_population_indices):
            raise ValueError("Population pool is too small for an OUT training set of size N.")

        for local_m, model_idx in enumerate(selected_model_indices):
            is_in = model_idx < M_pop_in
            rng = np.random.default_rng(
                args.seed + target_pos * args.M_pop + model_idx
            )

            n_fresh = args.training_sample_size - int(is_in)
            train_indices = rng.choice(
                target_population_indices,
                size=n_fresh,
                replace=False,
            )

            if is_in:
                train_indices = np.append(train_indices, target_dataset_idx)

            train_indices = np.sort(train_indices)

            model = TabPFNClassifier(device="cuda", n_estimators=1)
            model.fit(X.iloc[train_indices], y[train_indices])

            predicted_probs = model.predict_proba(target_x)
            true_class_prob = predicted_probs[0, target_y]

            target_stats[local_t, local_m] = prob_to_score(true_class_prob)
            target_membership[local_t, local_m] = is_in

    output_dir = os.path.join(
        args.results,
        args.dataset,
        f"Seed={args.seed}",
        f"N={args.training_sample_size}",
    )
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "dataset": args.dataset,
        "model": "TabPFN",
        "csv_file": args.csv_file,
        "seed": args.seed,
        "N": args.training_sample_size,
        "M_pop": args.M_pop,
        "M_pop_in": M_pop_in,
        "target_start_idx": args.target_start_idx,
        "target_stop_idx": target_stop_idx,
        "model_start_idx": args.model_start_idx,
        "model_stop_idx": model_stop_idx,
        "target_positions": selected_target_positions,
        "X_eval_indices": eval_indices,
        "target_stats": target_stats,
        "target_membership": target_membership,
    }

    output_path = os.path.join(
        output_dir,
        (
            f"population_targets_{args.target_start_idx}_{target_stop_idx}"
            f"_models_{args.model_start_idx}_{model_stop_idx}.pkl"
        ),
    )
    with open(output_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {output_path}")
    print(f"Targets processed: {n_targets_job}")
    print(f"Models per target in this job: {n_models_job}")
    print(f"Global split: {M_pop_in} IN / {args.M_pop - M_pop_in} OUT")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        main()
