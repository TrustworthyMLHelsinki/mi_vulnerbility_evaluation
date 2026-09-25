import argparse
import os
import pickle
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cached_data_loader import CachedFeatureLoader
from dataset import dataset_map
from lira import calculate_statistic, convert_logit_to_prob, log_loss

from ViT.fpc_linear_training import (
    DEVICE, BatchedLinearHeads, fine_tune_batched,
    add_lbfgs_arguments, lbfgs_options,
)


def compute_batched_accuracy(model, features, labels, batch_size=256):
    model.eval()
    K = model.weight.shape[0]
    correct = torch.zeros(K, device=DEVICE)
    total = 0

    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.long().to(DEVICE)
            xb = x.unsqueeze(0).expand(K, -1, -1)
            pred = model(xb).argmax(dim=2)
            correct += (pred == y.unsqueeze(0)).sum(dim=1)
            total += y.numel()

    return (correct / total).cpu().numpy()


def find_frame_for_nplus(subsets, N_plus):
    """Return the saved finite frame whose size is exactly N_plus."""
    for ratio, size in subsets["frame_sizes_by_ratio"].items():
        if int(size) == int(N_plus):
            return float(ratio), np.asarray(subsets["X_superset_indices_by_ratio"][ratio])

    available = sorted(int(v) for v in subsets["frame_sizes_by_ratio"].values())
    raise ValueError(f"N_plus={N_plus} was not saved. Available N_plus values: {available}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results", required=True, help="Directory for model statistics/results.")
    parser.add_argument("--fpc_data_dir", required=True, help="Directory containing the saved fpc_data_subsets.pkl.")
    parser.add_argument("--dataset", required=True, help="Dataset to use.")
    parser.add_argument("--dataset_dir", default=".", help="Directory containing cached dataset/features.")
    parser.add_argument("--feature_extractor", choices=["vit-b-16", "BiT-M-R50x1"], default="vit-b-16")

    parser.add_argument("--training_sample_size", type=int, default=1000, help="Per-model training-set size N.")
    parser.add_argument("--target_ratio", type=float, required=True, help="Finite-frame N/N_plus for this run.")
    parser.add_argument("--M_shadow", type=int, default=1000, help="Total number of shadow models for this N_plus.")

    parser.add_argument("--train_batch_size", "-b", type=int, default=128)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0025)
    parser.add_argument("--epochs", "-e", type=int, default=40)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD", "LBFGS"], help="Optimizer to use for training.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize_features", action="store_true", help="Whether to normalize features before training.")
    parser.add_argument("--weight_decay", "--l2_reg", type=float, default=0.0001, help="L2 penalty on the linear-head weights.")
    # Enables parallel/chunked jobs without changing the model-specific subsets.
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--stop_idx", type=int, default=None)
    parser.add_argument("--models_per_gpu_batch", type=int, default=32, help="Number of shadow models trained simultaneously on the GPU.")

    add_lbfgs_arguments(parser)
    args = parser.parse_args()

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    torch.set_num_threads(n_cpus)
    torch.set_num_interop_threads(1)

    print("Torch threads:", torch.get_num_threads())
    print("Interop threads:", torch.get_num_interop_threads())

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
    eval_indices = np.asarray(subsets["X_eval_indices"])

    if args.training_sample_size > len(frame_indices):
        raise ValueError("training_sample_size cannot exceed N_plus.")

    stop_idx = args.M_shadow if args.stop_idx is None else args.stop_idx
    if not (0 <= args.start_idx < stop_idx <= args.M_shadow):
        raise ValueError("Require 0 <= start_idx < stop_idx <= M_shadow.")

    dataset_reader = CachedFeatureLoader(
        path_to_cache_dir=args.dataset_dir,
        dataset=args.dataset,
        feature_extractor=args.feature_extractor,
        random_seed=args.seed,
    )

    # Fetch the dataset features/labels and class mapping.
    feature_dim = dataset_reader.obtain_feature_dim()
    num_classes = dataset_map[args.dataset][0]["num_classes"]
    train_features, train_labels, class_mapping = dataset_reader.load_train_data(shots=-1,
                                                                    n_classes=num_classes)
    test_features, test_labels = dataset_reader.load_test_data(class_mapping=class_mapping)

    if args.normalize_features:
        train_features = torch.nn.functional.normalize(
            train_features, p=2, dim=1
        )
        test_features = torch.nn.functional.normalize(
            test_features, p=2, dim=1
        )

    print("Training features shape:", train_features.shape)
    
    # Selecting X_eval features/labels from the data.
    eval_features = train_features[eval_indices]
    eval_labels = train_labels[eval_indices]

    N_models = stop_idx - args.start_idx
    N_eval = len(eval_indices)

    target_stats = np.empty((N_models, N_eval), dtype=np.float32)
    target_losses = np.empty((N_models, N_eval), dtype=np.float32)
    target_membership = np.zeros((N_models, N_eval), dtype=bool)


    # map the original dataset index -> target column; -1 means not in X_eval.
    target_lookup = np.full(len(train_features), -1, dtype=np.int64)
    target_lookup[eval_indices] = np.arange(N_eval)

    selected_model_indices = np.arange(args.start_idx, stop_idx)
    K_max = 1 if args.optim == "LBFGS" else args.models_per_gpu_batch

    for chunk_start in tqdm(range(0, N_models, K_max), desc="Model batches"):
        chunk_model_indices = selected_model_indices[chunk_start:chunk_start + K_max]
        K = len(chunk_model_indices)
        x_list, y_list = [], []

        for j, model_idx in enumerate(chunk_model_indices):
            # Model-specific RNG keeps exactly the same finite-frame subset
            # as the original one-model-at-a-time implementation.
            rng = np.random.default_rng(args.seed + model_idx)
            train_indices = rng.choice(
                frame_indices,
                size=args.training_sample_size,
                replace=False,
            )
            if args.optim != "LBFGS":
                train_indices = np.sort(train_indices)

            local_i = chunk_start + j
            target_cols = target_lookup[train_indices]
            target_cols = target_cols[target_cols >= 0]
            target_membership[local_i, target_cols] = True

            x_list.append(train_features[train_indices])
            y_list.append(train_labels[train_indices])

        x_batch = torch.stack(x_list).to(DEVICE)
        y_batch = torch.stack(y_list).long().to(DEVICE)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        model = BatchedLinearHeads(K, feature_dim, num_classes)
        fine_tune_batched(
            model,
            x_batch,
            y_batch,
            lr=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            optimizer_name=args.optim,
            weight_decay=args.weight_decay,
            seed=args.seed,
            **lbfgs_options(args),
        )

        test_accuracies = compute_batched_accuracy(
            model,
            test_features,
            test_labels,
            batch_size=256,
        )
        for j, model_idx in enumerate(chunk_model_indices):
            print(f"Model {model_idx} with test accuracy = {100 * test_accuracies[j]:.2f}")

        with torch.no_grad():
            ex = eval_features.to(DEVICE).unsqueeze(0).expand(K, -1, -1)
            logits_np = model(ex).cpu().numpy()

        eval_labels_np = eval_labels.cpu().numpy()
        for j in range(K):
            prob = convert_logit_to_prob(logits_np[j])
            local_i = chunk_start + j
            target_stats[local_i] = np.asarray(
                calculate_statistic(prob, eval_labels_np, is_logits=False)
            ).reshape(-1)
            target_losses[local_i] = np.asarray(
                log_loss(eval_labels_np, prob)
            ).reshape(-1)

        del model, x_batch, y_batch

    output_dir = os.path.join(
        args.results,
        args.dataset,
        f"Seed={args.seed}",
        f"N={args.training_sample_size}",
        f"ratio={args.target_ratio}",
    )
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "optimizer": args.optim,
        "l2_reg": args.weight_decay,
        "dataset": args.dataset,
        "feature_extractor": args.feature_extractor,
        "seed": args.seed,
        "N": args.training_sample_size,
        "N_plus": N_plus,
        "ratio": ratio,
        "M_shadow": args.M_shadow,
        "start_idx": args.start_idx,
        "stop_idx": stop_idx,
        "X_eval_indices": eval_indices,
        "target_stats": target_stats,
        "target_losses": target_losses,
        "target_membership": target_membership,
    }

    if args.optim == "LBFGS":
        output.update(lbfgs_options(args))

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
        warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")
        main()
