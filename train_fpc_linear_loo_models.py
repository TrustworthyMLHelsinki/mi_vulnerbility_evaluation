import argparse
import os
import pickle
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cached_data_loader import CachedFeatureLoader
from dataset import dataset_map
from lira import calculate_statistic, convert_logit_to_prob, log_loss
from utils import cross_entropy_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_head(feature_dim: int, num_classes: int):
    head = nn.Linear(feature_dim, num_classes)
    head.weight.data.fill_(0.0)
    head.bias.data.fill_(0.0)
    return head.to(DEVICE)


def fine_tune_batch(model, train_loader, lr, epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.long().to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = cross_entropy_loss(logits, batch_labels)
            loss.backward()
            optimizer.step()

    return model


def get_stat_and_loss(model, x, y):
    """Compute the LiRA statistic and loss for one evaluation target."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(DEVICE)).cpu().numpy()

    prob = convert_logit_to_prob(logits)
    y_np = y.cpu().numpy()

    stat = calculate_statistic(prob, y_np, is_logits=False)
    loss = log_loss(y_np, prob)

    return float(np.asarray(stat).reshape(-1)[0]), float(np.asarray(loss).reshape(-1)[0])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results", required=True, help="Directory for model statistics/results.")
    parser.add_argument("--data_directory", required=True,
                        help="Directory containing fpc_data_subsets.pkl.")
    parser.add_argument("--dataset", required=True, help="Dataset to use.")
    parser.add_argument("--dataset_dir", default=".", help="Directory containing cached dataset/features.")
    parser.add_argument("--feature_extractor", choices=["vit-b-16", "BiT-M-R50x1"],
                        default="vit-b-16")

    parser.add_argument("--training_sample_size", type=int, default=1000,
                        help="Per-model training-set size N.")
    parser.add_argument("--M_pop", type=int, default=1000,
                        help="Total population LOO models per evaluation target. Must be even.")

    parser.add_argument("--train_batch_size", "-b", type=int, default=128)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0025)
    parser.add_argument("--epochs", "-e", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)

    # Which evaluation targets to process in this job.
    parser.add_argument("--target_start_idx", type=int, default=0)
    parser.add_argument("--target_stop_idx", type=int, default=None)

    # Which population models to process for each selected target.
    parser.add_argument("--model_start_idx", type=int, default=0)
    parser.add_argument("--model_stop_idx", type=int, default=None)

    args = parser.parse_args()

    if args.M_pop % 2 != 0:
        raise ValueError("M_pop must be even so that half the models are IN and half are OUT.")

    subset_path = os.path.join(
        args.data_directory,
        args.dataset,
        f"Seed={args.seed}",
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

    if args.training_sample_size > len(population_indices):
        raise ValueError("Population pool is too small for an OUT training set of size N.")

    if args.training_sample_size - 1 > len(population_indices):
        raise ValueError("Population pool is too small for an IN training set of size N.")

    N_eval = len(eval_indices)

    target_stop_idx = N_eval if args.target_stop_idx is None else args.target_stop_idx
    if not (0 <= args.target_start_idx < target_stop_idx <= N_eval):
        raise ValueError("Require 0 <= target_start_idx < target_stop_idx <= len(X_eval).")

    model_stop_idx = args.M_pop if args.model_stop_idx is None else args.model_stop_idx
    if not (0 <= args.model_start_idx < model_stop_idx <= args.M_pop):
        raise ValueError("Require 0 <= model_start_idx < model_stop_idx <= M_pop.")

    dataset_reader = CachedFeatureLoader(
                            path_to_cache_dir=args.dataset_dir,
                            dataset=args.dataset,
                            feature_extractor=args.feature_extractor,
                            random_seed=args.seed,
                        )

    feature_dim = dataset_reader.obtain_feature_dim()
    num_classes = dataset_map[args.dataset][0]["num_classes"]

    train_features, train_labels, _ = dataset_reader.load_train_data(shots=-1, n_classes=num_classes)

    selected_target_positions = np.arange(args.target_start_idx, target_stop_idx)
    selected_model_indices = np.arange(args.model_start_idx, model_stop_idx)

    n_targets_job = len(selected_target_positions)
    n_models_job = len(selected_model_indices)

    target_stats = np.empty((n_targets_job, n_models_job), dtype=np.float32)
    target_losses = np.empty((n_targets_job, n_models_job), dtype=np.float32)
    target_membership = np.zeros((n_targets_job, n_models_job), dtype=bool)

    # First M_pop/2 models are IN, second M_pop/2 are OUT for every target.
    M_pop_in = args.M_pop // 2
    for local_t, target_pos in enumerate(tqdm(selected_target_positions, desc="Targets")):
        target_dataset_idx = int(eval_indices[target_pos])

        target_x = train_features[target_dataset_idx:target_dataset_idx + 1]
        target_y = train_labels[target_dataset_idx:target_dataset_idx + 1]

        for local_m, model_idx in enumerate(selected_model_indices):
            is_in = model_idx < M_pop_in

            # Deterministic seed for each (target, model) pair.
            pair_seed = int(args.seed + target_pos * args.M_pop + model_idx)
            rng = np.random.default_rng(pair_seed)
            if is_in:
                # LOO-IN: x + N-1 fresh samples from the disjoint population pool.
                fresh_indices = rng.choice(population_indices, size=args.training_sample_size - 1,
                                    replace=False)
                train_indices = np.concatenate([np.array([target_dataset_idx], dtype=np.int64),
                                                fresh_indices])
            else:
                # LOO-OUT: N fresh samples from the disjoint population pool.
                train_indices = rng.choice(population_indices, size=args.training_sample_size,
                                            replace=False)

            x, y = train_features[train_indices], train_labels[train_indices]

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            loader_generator = torch.Generator()
            loader_generator.manual_seed(args.seed)
            train_loader = DataLoader(
                TensorDataset(x, y),
                batch_size=args.train_batch_size,
                shuffle=True,
                generator=loader_generator)

            model = create_head(feature_dim=feature_dim, num_classes=num_classes)
            fine_tune_batch(model, train_loader, lr=args.learning_rate, epochs=args.epochs)

            stat, loss = get_stat_and_loss(model, target_x, target_y)

            target_stats[local_t, local_m], target_losses[local_t, local_m] = stat, loss
            target_membership[local_t, local_m] = is_in

            del model, train_loader, x, y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_dir = os.path.join(
        args.results,
        args.dataset,
        f"Seed={args.seed}",
        f"N={args.training_sample_size}"
        )
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "M_pop": args.M_pop,
        "M_pop_in": M_pop_in,
        "target_start_idx": args.target_start_idx,
        "target_stop_idx": target_stop_idx,
        "model_start_idx": args.model_start_idx,
        "model_stop_idx": model_stop_idx,
        "target_positions": selected_target_positions,
        "target_stats": target_stats,
        "target_losses": target_losses,
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

    # print(f"Saved: {output_path}")
    print(f"Targets processed: {n_targets_job}")
    print(f"Models per target in this job: {n_models_job}")
    print(f"Global split: {M_pop_in} IN / {args.M_pop - M_pop_in} OUT")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")
        main()
