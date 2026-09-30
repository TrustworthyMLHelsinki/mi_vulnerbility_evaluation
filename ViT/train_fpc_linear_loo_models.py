import argparse
import os
import pickle
import warnings

import numpy as np
import torch
from tqdm import tqdm

from cached_data_loader import CachedFeatureLoader
from dataset import dataset_map
from lira import calculate_statistic, convert_logit_to_prob, log_loss

from ViT.fpc_linear_training import (
    DEVICE, BatchedLinearHeads, fine_tune_batched,
    add_lbfgs_arguments, lbfgs_options,
)

import time
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results", required=True, help="Directory for model statistics/results.")
    parser.add_argument("--data_directory", required=True, help="Directory containing fpc_data_subsets.pkl.")
    parser.add_argument("--dataset", required=True, help="Dataset to use.")
    parser.add_argument("--dataset_dir", default=".", help="Directory containing cached dataset/features.")
    parser.add_argument("--feature_extractor", choices=["vit-b-16", "BiT-M-R50x1"], default="vit-b-16")

    parser.add_argument("--training_sample_size", type=int, default=1000, help="Per-model training-set size N.")
    parser.add_argument("--M_pop", type=int, default=1000, help="Total population LOO models per evaluation target. Must be even.")

    parser.add_argument("--train_batch_size", "-b", type=int, default=128)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0025)
    parser.add_argument("--epochs", "-e", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize_features", action="store_true", help="Whether to normalize features before training.")
    parser.add_argument("--weight_decay", "--l2_reg", type=float, default=0.0001, help="L2 penalty on the linear-head weights.")
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD", "LBFGS"], help="Optimizer to use for training.")
    # Which evaluation targets to process in this job.
    parser.add_argument("--target_start_idx", type=int, default=0)
    parser.add_argument("--target_stop_idx", type=int, default=None)

    # Which population models to process for each selected target.
    parser.add_argument("--model_start_idx", type=int, default=0)
    parser.add_argument("--model_stop_idx", type=int, default=None)

    # batch training of models
    parser.add_argument("--models_per_gpu_batch", type=int, default=32)
    add_lbfgs_arguments(parser)
    args = parser.parse_args()

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    torch.set_num_threads(n_cpus)
    torch.set_num_interop_threads(1)

    print("Torch threads:", torch.get_num_threads())
    print("Interop threads:", torch.get_num_interop_threads())
    
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

    if args.normalize_features:
        train_features = torch.nn.functional.normalize(
            train_features, p=2, dim=1
        )
        
    selected_target_positions = np.arange(args.target_start_idx, target_stop_idx)
    selected_model_indices = np.arange(args.model_start_idx, model_stop_idx)

    n_targets_job = len(selected_target_positions)
    n_models_job = len(selected_model_indices)

    target_stats = np.empty((n_targets_job, n_models_job), dtype=np.float32)
    target_losses = np.empty((n_targets_job, n_models_job), dtype=np.float32)
    target_membership = np.zeros((n_targets_job, n_models_job), dtype=bool)

    M_pop_in = args.M_pop // 2
    K_max = 1 if args.optim == "LBFGS" else args.models_per_gpu_batch

    for local_t, target_pos in enumerate(tqdm(selected_target_positions, desc="Targets")):
        target_dataset_idx = int(eval_indices[target_pos])
        target_x = train_features[target_dataset_idx:target_dataset_idx + 1]
        target_y = train_labels[target_dataset_idx:target_dataset_idx + 1]
        target_population_indices = population_indices[population_indices != target_dataset_idx]

        t_sample = t_setup = t_train = t_eval = 0.0

        for chunk_start in range(0, n_models_job, K_max):
            chunk_model_indices = selected_model_indices[chunk_start:chunk_start + K_max]
            K = len(chunk_model_indices)

            # Sampling
            sync(); t0 = time.perf_counter()
            x_list, y_list, memberships = [], [], []

            for model_idx in chunk_model_indices:
                is_in = model_idx < M_pop_in
                rng = np.random.default_rng(args.seed + target_pos * args.M_pop + model_idx)

                n_fresh = args.training_sample_size - int(is_in)
                train_indices = rng.choice(target_population_indices, size=n_fresh, replace=False)

                if is_in:
                    if args.optim == "LBFGS":
                        train_indices = np.concatenate(([target_dataset_idx], train_indices))
                    else:
                        train_indices = np.append(train_indices, target_dataset_idx)

                if args.optim != "LBFGS":
                    train_indices = np.sort(train_indices)
                x_list.append(train_features[train_indices])
                y_list.append(train_labels[train_indices])
                memberships.append(is_in)

            sync(); t1 = time.perf_counter()
            t_sample += t1 - t0

            # Setup
            x_batch = torch.stack(x_list).to(DEVICE)
            y_batch = torch.stack(y_list).long().to(DEVICE)
            model = BatchedLinearHeads(K, feature_dim, num_classes)

            sync(); t2 = time.perf_counter()
            t_setup += t2 - t1

            # Training
            fine_tune_batched(
                model, x_batch, y_batch,
                args.learning_rate, args.epochs, args.train_batch_size,
                args.optim, args.weight_decay, args.seed,
                **lbfgs_options(args),
            )

            sync(); t3 = time.perf_counter()
            t_train += t3 - t2

            # Evaluation
            with torch.no_grad():
                tx = target_x.to(DEVICE).unsqueeze(0).expand(K, -1, -1)
                logits_np = model(tx)[:, 0].cpu().numpy()

            target_y_np = target_y.cpu().numpy()

            for j in range(K):
                prob = convert_logit_to_prob(logits_np[j:j + 1])
                local_m = chunk_start + j

                target_stats[local_t, local_m] = float(
                    np.asarray(calculate_statistic(prob, target_y_np, is_logits=False)).ravel()[0]
                )
                target_losses[local_t, local_m] = float(
                    np.asarray(log_loss(target_y_np, prob)).ravel()[0]
                )
                target_membership[local_t, local_m] = memberships[j]

            sync(); t4 = time.perf_counter()
            t_eval += t4 - t3

            del model, x_batch, y_batch

        # print(
        #     f"Target {target_pos}: "
        #     f"Sampling={t_sample:.2f}s, "
        #     f"Setup={t_setup:.2f}s, "
        #     f"Training={t_train:.2f}s, "
        #     f"Evaluation={t_eval:.2f}s"
        # )

    # -----------------------------------------------------------------------------------------
    output_dir = os.path.join(
        args.results,
        args.dataset,
        f"Seed={args.seed}",
        f"N={args.training_sample_size}"
        )
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "optimizer": args.optim,
        "l2_reg": args.weight_decay,
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

    if args.optim == "LBFGS":
        output.update(lbfgs_options(args))

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
        warnings.filterwarnings("ignore", message=r".*Using a non-full backward hook.*")
        main()
