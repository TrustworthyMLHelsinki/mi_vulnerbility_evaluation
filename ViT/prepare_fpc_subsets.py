import argparse
import os
import pickle

import numpy as np

from cached_data_loader import CachedFeatureLoader
from dataset import dataset_map


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, help="Dataset to use.")
    parser.add_argument("--dataset_dir", default=".", help="Directory containing cached dataset/features.")
    parser.add_argument("--data_directory", required=True, help="Directory in which to save the FPC data subsets.")
    parser.add_argument("--feature_extractor", choices=["vit-b-16", "BiT-M-R50x1"], default="vit-b-16", help="Feature extractor used for cached features.")
    parser.add_argument("--training_sample_size", type=int, default=1000, help="Per-model training-set size N.")
    parser.add_argument("--frame_size_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75], help="Values of N / N_plus.")
    parser.add_argument("--eval_sample_size", type=int, default=50, help="Number of fixed evaluation samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    if any(r <= 0 or r > 1 for r in args.frame_size_ratios):
        raise ValueError("All frame_size_ratios must lie in (0, 1].")

    rng = np.random.default_rng(args.seed)

    dataset_reader = CachedFeatureLoader(
        path_to_cache_dir=args.dataset_dir,
        dataset=args.dataset,
        feature_extractor=args.feature_extractor,
        random_seed=args.seed,
    )

    num_classes = dataset_map[args.dataset][0]["num_classes"]
    train_features, train_labels, class_mapping = dataset_reader.load_train_data(
        shots=-1,
        n_classes=num_classes,
    )

    n_total = len(train_features)

    # Largest finite frame required by the smallest N / N_plus ratio.
    N_superset_max_size = int(round(
        args.training_sample_size / min(args.frame_size_ratios)
    ))

    if N_superset_max_size > n_total:
        raise ValueError(
            f"Largest N_plus ({N_superset_max_size}) exceeds available samples ({n_total})."
        )

    if args.eval_sample_size > N_superset_max_size:
        raise ValueError("eval_sample_size cannot exceed the smallest available finite frame.")

    # Sample one maximum frame. All smaller frames will be nested inside it.
    X_superset_max_indices = rng.choice(
        n_total,
        size=N_superset_max_size,
        replace=False,
    )

    # Fix X_eval once so the same targets are used for every N_plus.
    X_eval_indices = rng.choice(
        X_superset_max_indices,
        size=args.eval_sample_size,
        replace=False,
    )

    # Candidates used to enlarge/shrink the finite frame while keeping X_eval fixed.
    is_eval = np.isin(X_superset_max_indices, X_eval_indices)
    X_superset_non_eval_indices = X_superset_max_indices[~is_eval]

    # Population pool: all available samples, including samples in the finite frame.
    X_population_indices = np.arange(n_total)

    # Construct one nested finite frame for every N / N_plus ratio.
    frame_indices_by_ratio = {}
    frame_sizes_by_ratio = {}

    for ratio in args.frame_size_ratios:
        N_plus = int(round(args.training_sample_size / ratio))

        if N_plus < args.eval_sample_size:
            raise ValueError(
                f"N_plus={N_plus} for ratio={ratio} is smaller than eval_sample_size."
            )

        n_extra = N_plus - args.eval_sample_size
        X_superset_indices = np.concatenate([
            X_eval_indices,
            X_superset_non_eval_indices[:n_extra],
        ])

        frame_indices_by_ratio[ratio] = X_superset_indices
        frame_sizes_by_ratio[ratio] = N_plus

    subsets = {
        "dataset": args.dataset,
        "feature_extractor": args.feature_extractor,
        "seed": args.seed,
        "training_sample_size": args.training_sample_size,
        "eval_sample_size": args.eval_sample_size,
        "frame_size_ratios": list(args.frame_size_ratios),
        "frame_sizes_by_ratio": frame_sizes_by_ratio,
        "X_eval_indices": X_eval_indices,
        "X_superset_max_indices": X_superset_max_indices,
        "X_population_indices": X_population_indices,
        "X_superset_indices_by_ratio": frame_indices_by_ratio,
    }

    output_dir = os.path.join(
        args.data_directory,
        args.dataset,
        f"Seed={args.seed}",
        f"N={args.training_sample_size}"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "fpc_data_subsets.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(subsets, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved subsets to: {output_path}")
    print(f"Total samples: {n_total}")
    print(f"Fixed X_eval size: {len(X_eval_indices)}")
    print(f"Population pool size: {len(X_population_indices)}")
    for ratio in args.frame_size_ratios:
        print(f"N/N_plus={ratio:.3f} -> N_plus={frame_sizes_by_ratio[ratio]}")


if __name__ == "__main__":
    main()
