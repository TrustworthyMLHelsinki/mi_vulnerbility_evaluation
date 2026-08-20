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

def compute_accuracy(model, features, labels, batch_size=256):
    model.eval()

    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.long().to(DEVICE)

            pred = model(x).argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / total

def get_stat_and_loss(model, x, y):
    """Compute the LiRA statistic and loss for the fixed evaluation targets."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(DEVICE)).cpu().numpy()

    prob = convert_logit_to_prob(logits)
    y_np = y.cpu().numpy()

    stat = calculate_statistic(prob, y_np, is_logits=False)
    loss = log_loss(y_np, prob)
    return np.asarray(stat).reshape(-1), np.asarray(loss).reshape(-1)


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
    parser.add_argument("--fpc_data_dir", required=True,
                        help="Directory containing the saved fpc_data_subsets.pkl.")
    parser.add_argument("--dataset", required=True, help="Dataset to use.")
    parser.add_argument("--dataset_dir", default=".", help="Directory containing cached dataset/features.")
    parser.add_argument("--feature_extractor", choices=["vit-b-16", "BiT-M-R50x1"],
                        default="vit-b-16")

    parser.add_argument("--training_sample_size", type=int, default=1000,
                        help="Per-model training-set size N.")
    parser.add_argument("--target_ratio", type=float, required=True,
                        help="Finite-frame N/N_plus for this run.")
    parser.add_argument("--M_shadow", type=int, default=1000,
                        help="Total number of shadow models for this N_plus.")

    parser.add_argument("--train_batch_size", "-b", type=int, default=128)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0025)
    parser.add_argument("--epochs", "-e", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)

    # Enables parallel/chunked jobs without changing the model-specific subsets.
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--stop_idx", type=int, default=None)

    args = parser.parse_args()

    subset_path = os.path.join(
        args.fpc_data_dir,
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

    for local_i, model_idx in enumerate(tqdm(range(args.start_idx, stop_idx))):
        # Model-specific RNG makes subsets reproducible even when jobs are chunked differently.
        rng = np.random.default_rng(args.seed + model_idx)
        # Natural finite-frame sampling: N records uniformly without replacement from D_N_plus.
        train_indices = rng.choice(
            frame_indices,
            size=args.training_sample_size,
            replace=False)
        # record which fixed evaluation targets are members of this model's training set.
        target_cols = target_lookup[train_indices]
        target_cols = target_cols[target_cols >= 0]
        target_membership[local_i, target_cols] = True

        x,y = train_features[train_indices], train_labels[train_indices]
        
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)
        train_loader = DataLoader(
            TensorDataset(x, y),
            batch_size=args.train_batch_size,
            shuffle=True,
            generator=loader_generator,
        )

        model = create_head(feature_dim=feature_dim, num_classes=num_classes)
        fine_tune_batch(
            model,
            train_loader,
            lr=args.learning_rate,
            epochs=args.epochs,
        )

        test_accuracy = compute_accuracy(
            model,
            test_features,
            test_labels,
            batch_size=256,
        )

        print(f"Model {model_idx} with test accuracy = {100 * test_accuracy:.2f}")

        stats, losses = get_stat_and_loss(model, eval_features, eval_labels)
        target_stats[local_i] = stats
        target_losses[local_i] = losses


        del model, train_loader, x, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    output_path = os.path.join(
        output_dir,
        f"shadow_{args.start_idx}_{stop_idx}.pkl",
    )
    with open(output_path, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print(f"Saved: {output_path}")
    print(f"N_plus={N_plus}, N/N_plus={ratio:.4f}")
    print(f"Models trained: {N_models}")
    print(f"Mean target IN fraction in this chunk: {target_membership.mean():.4f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")
        main()
