"""Training and target-statistic inference for BiT-FiLM models."""

from __future__ import annotations

import random
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .model import build_bit_m_r50x1_film


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    if (
        requested == "mps"
        and not torch.backends.mps.is_available()
    ):
        raise RuntimeError(
            "MPS was requested but is unavailable."
        )

    return torch.device(requested)


def make_optimizer(
    *,
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return torch.optim.SGD(
        trainable_parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def train_film_model(
    *,
    train_dataset: Dataset,
    membership: np.ndarray,
    bit_weights: Mapping[str, np.ndarray],
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    momentum: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
) -> nn.Module:
    if epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if batch_size <= 0:
        raise ValueError(
            "--batch_size must be positive."
        )
    if learning_rate <= 0:
        raise ValueError(
            "--learning_rate must be positive."
        )

    seed_everything(seed)

    included_indices = np.flatnonzero(
        membership
    ).tolist()
    training_subset = Subset(
        train_dataset,
        included_indices,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(
        training_subset,
        **loader_kwargs,
    )

    model = build_bit_m_r50x1_film(
        bit_weights=bit_weights,
        num_classes=num_classes,
    ).to(device)

    optimizer = make_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    non_blocking = device.type == "cuda"

    model.train()

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=non_blocking,
            )
            labels = labels.to(
                device,
                dtype=torch.long,
                non_blocking=non_blocking,
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return model


@torch.inference_mode()
def predict_true_class_probabilities(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError(
            "--eval_batch_size must be positive."
        )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    model.eval()
    true_class_probability_batches: list[np.ndarray] = []
    non_blocking = device.type == "cuda"

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=non_blocking,
        )
        labels = labels.to(
            device,
            dtype=torch.long,
            non_blocking=non_blocking,
        )

        probabilities = torch.softmax(
            model(images),
            dim=1,
        )
        true_class_probabilities = probabilities.gather(
            1,
            labels[:, None],
        )

        true_class_probability_batches.append(
            true_class_probabilities[:, 0]
            .cpu()
            .numpy()
        )

    return np.concatenate(
        true_class_probability_batches
    )
