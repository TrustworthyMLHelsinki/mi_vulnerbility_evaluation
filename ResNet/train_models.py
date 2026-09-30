# UPDATED VERSION: loads BiT-M-R50x1 through timm; no bit_resnet.py dependency.
import argparse
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "This script requires timm. Install it with: "
        "pip install timm huggingface_hub safetensors"
    ) from exc
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


class R50FiLM(nn.Module):
    """BiT-M-R50x1 with the FiLM parameters used by dp-few-shot."""

    MODEL_NAME = "resnetv2_50x1_bit.goog_in21k"

    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = timm.create_model(
            self.MODEL_NAME,
            pretrained=True,
            num_classes=0,
        )

        # dp-few-shot selects the third GroupNorm in every bottleneck and the
        # final GroupNorm. In timm these are named *.norm3.* and norm.*.
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        for name, parameter in self.feature_extractor.named_parameters():
            if ".norm3." in name or name.startswith("norm."):
                parameter.requires_grad = True

        self.head = nn.Linear(self.feature_extractor.num_features, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset_dir", default=".")
    parser.add_argument("--examples_per_class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--test_batch_size", type=int, default=600)
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.003)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--stop_idx", type=int, default=1)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    directory = os.path.join(
        args.results,
        "cifar10",
        f"Seed={args.seed}",
        f"EPC={args.examples_per_class}",
    )
    os.makedirs(directory, exist_ok=True)

    # This matches the dp-few-shot BiT preprocessing: resize and map [0,1] to [-1,1].
    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    cifar10 = datasets.CIFAR10(
        root=args.dataset_dir,
        train=True,
        download=args.download,
        transform=transform,
    )

    target_indices_file = os.path.join(directory, "target_indices.pkl")
    validation_indices_file = os.path.join(directory, "validation_indices.pkl")
    in_indices_file = os.path.join(directory, "in_indices_target.pkl")

    if not args.train:
        # Select k target examples and a separate k validation examples per class.
        target_indices, validation_indices = select_disjoint_subsets(
            np.asarray(cifar10.targets), args.examples_per_class, args.seed
        )
        N = len(target_indices)

        rng = np.random.default_rng(args.seed)
        target_in_indices = rng.binomial(
            1, 0.5, size=(args.num_models + 1, N)
        ).astype(bool)

        with open(target_indices_file, "wb") as f:
            pickle.dump(target_indices, f)
        with open(validation_indices_file, "wb") as f:
            pickle.dump(validation_indices, f)
        with open(in_indices_file, "wb") as f:
            pickle.dump(target_in_indices, f)

        print("Using target dataset of size:", N)
        print("Using validation dataset of size:", len(validation_indices))
        print("Membership matrix shape:", target_in_indices.shape)

    else:
        with open(target_indices_file, "rb") as f:
            target_indices = pickle.load(f)
        with open(validation_indices_file, "rb") as f:
            validation_indices = pickle.load(f)
        with open(in_indices_file, "rb") as f:
            target_in_indices = pickle.load(f)

        target_dataset = Subset(cifar10, target_indices.tolist())
        validation_dataset = Subset(cifar10, validation_indices.tolist())
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        target_loader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        N = len(target_dataset)
        target_stats = np.zeros((args.stop_idx - args.start_idx, N))
        train_accuracies = np.zeros(args.stop_idx - args.start_idx)
        validation_accuracies = np.zeros(args.stop_idx - args.start_idx)

        for i in tqdm(range(args.start_idx, args.stop_idx)):
            set_seeds(args.seed + i)
            D_in = target_in_indices[i]
            train_dataset = Subset(target_dataset, np.flatnonzero(D_in).tolist())
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                generator=torch.Generator().manual_seed(args.seed + i),
            )
            train_eval_loader = DataLoader(
                train_dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

            model = R50FiLM().to(device)
            train_model(
                model,
                train_loader,
                device,
                args.epochs,
                args.learning_rate,
                args.optimizer,
            )

            train_acc = compute_accuracy(model, train_eval_loader, device)
            validation_acc = compute_accuracy(model, validation_loader, device)
            train_accuracies[i - args.start_idx] = train_acc
            validation_accuracies[i - args.start_idx] = validation_acc

            print(
                f"Model {i}: trained on {D_in.sum()} samples | "
                f"train accuracy = {100 * train_acc:.2f}% | "
                f"validation accuracy = {100 * validation_acc:.2f}%"
            )

            logits, labels = predict_logits_and_labels(model, target_loader, device)
            target_stats[i - args.start_idx, :] = calculate_statistic(
                logits, labels, is_logits=True, option="logit"
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        target_stats = target_stats.reshape((args.stop_idx - args.start_idx, N, 1))
        # print(target_stats.shape)

        with open(
            os.path.join(
                directory, 
                f"stats_target_m_in_{args.start_idx}_{args.stop_idx}.pkl"), 
                "wb") as f:
            pickle.dump(target_stats, f)

        with open(
            os.path.join(
                directory,
                f"accuracies_target_m_in_{args.start_idx}_{args.stop_idx}.pkl"),
                "wb") as f:
            pickle.dump(
                {
                    "train": train_accuracies,
                    "validation": validation_accuracies,
                },
                f,
            )


def select_disjoint_subsets(targets, examples_per_class, seed):
    rng = np.random.default_rng(seed)
    target_indices, validation_indices = [], []

    for c in range(10):
        class_indices = np.flatnonzero(targets == c)
        if len(class_indices) < 2 * examples_per_class:
            raise ValueError(
                f"Class {c} needs at least {2 * examples_per_class} samples."
            )
        selected = rng.choice(
            class_indices, size=2 * examples_per_class, replace=False
        )
        target_indices.extend(selected[:examples_per_class])
        validation_indices.extend(selected[examples_per_class:])

    target_indices = np.asarray(target_indices)
    validation_indices = np.asarray(validation_indices)
    rng.shuffle(target_indices)
    rng.shuffle(validation_indices)
    return target_indices, validation_indices


def train_model(model, train_loader, device, epochs, learning_rate, optimizer_name):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    model.train()
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total


def predict_logits_and_labels(model, data_loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            all_logits.append(model(images).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def convert_logit_to_prob(logit: np.ndarray) -> np.ndarray:
    """Stable float64 softmax, matching dp-few-shot's LiRA code."""
    prob = logit - np.max(logit, axis=1, keepdims=True)
    prob = np.asarray(np.exp(prob), dtype=np.float64)
    return prob / np.sum(prob, axis=1, keepdims=True)


def calculate_statistic(
    pred: np.ndarray,
    labels: np.ndarray,
    is_logits: bool = True,
    option: str = "logit",
    small_value: float = 1e-45,
) -> np.ndarray:
    """Compute the true-class statistic as in dp-few-shot."""
    if option != "logit":
        raise ValueError('This training script supports option="logit" only.')

    pred = np.asarray(pred)
    labels = np.asarray(labels, dtype=np.int64)
    if pred.ndim != 2 or pred.shape[0] != labels.size:
        raise ValueError(
            "pred must have shape (num_samples, num_classes) and match labels."
        )

    if is_logits:
        pred = convert_logit_to_prob(pred)
    else:
        pred = np.asarray(pred, dtype=np.float64).copy()

    n = labels.size
    p_true = pred[np.arange(n), labels].copy()
    pred[np.arange(n), labels] = 0.0
    p_other = pred.sum(axis=1)
    return np.log(p_true + small_value) - np.log(p_other + small_value)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*Using a non-full backward hook.*"
        )
        main()