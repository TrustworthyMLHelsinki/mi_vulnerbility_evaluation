"""CIFAR-10 target-pool construction and balanced memberships."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


CIFAR10_NUM_CLASSES = 10
CIFAR10_TRAIN_EXAMPLES_PER_CLASS = 5000
BIT_MEAN = (0.5, 0.5, 0.5)
BIT_STD = (0.5, 0.5, 0.5)


def validate_examples_per_class(
    examples_per_class: int,
) -> None:
    maximum = CIFAR10_TRAIN_EXAMPLES_PER_CLASS // 2

    if not 1 <= examples_per_class <= maximum:
        raise ValueError(
            "--examples_per_class must be in "
            f"[1, {maximum}]. The upper bound leaves a target pool "
            "twice the per-model training size."
        )


def build_bit_transforms(
    image_size: int,
    use_augmentation: bool,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Create stochastic training and deterministic scoring transforms."""
    if image_size < 32:
        raise ValueError("--image_size must be at least 32.")

    normalize = transforms.Normalize(
        mean=BIT_MEAN,
        std=BIT_STD,
    )
    resize = transforms.Resize(
        (image_size, image_size),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    if use_augmentation:
        training_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.80, 1.0),
                    ratio=(0.90, 1.10),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        training_transform = transforms.Compose(
            [
                resize,
                transforms.ToTensor(),
                normalize,
            ]
        )

    evaluation_transform = transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return training_transform, evaluation_transform


def select_balanced_target_indices(
    labels: np.ndarray,
    examples_per_class_in_pool: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for class_id in range(CIFAR10_NUM_CLASSES):
        class_indices = np.flatnonzero(labels == class_id)

        if len(class_indices) < examples_per_class_in_pool:
            raise ValueError(
                f"Class {class_id} contains only {len(class_indices)} "
                f"examples, but {examples_per_class_in_pool} are required."
            )

        selected = rng.choice(
            class_indices,
            size=examples_per_class_in_pool,
            replace=False,
        )
        selected_indices.extend(selected.tolist())

    rng.shuffle(selected_indices)
    return np.asarray(selected_indices, dtype=np.int64)


def build_cifar10_target_pool(
    *,
    data_dir: Path,
    examples_per_class: int,
    image_size: int,
    seed: int,
    use_augmentation: bool,
) -> tuple[Dataset, Dataset, np.ndarray, np.ndarray]:
    """
    Return training and deterministic views of the same target examples.

    Local indices are identical in both datasets and in the membership matrix.
    """
    validate_examples_per_class(examples_per_class)

    training_transform, evaluation_transform = (
        build_bit_transforms(
            image_size=image_size,
            use_augmentation=use_augmentation,
        )
    )

    training_base = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=training_transform,
        download=True,
    )
    evaluation_base = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=evaluation_transform,
        download=True,
    )

    all_labels = np.asarray(
        training_base.targets,
        dtype=np.int64,
    )
    original_indices = select_balanced_target_indices(
        labels=all_labels,
        examples_per_class_in_pool=2 * examples_per_class,
        seed=seed,
    )

    target_labels = all_labels[original_indices]
    training_pool = Subset(
        training_base,
        original_indices.tolist(),
    )
    evaluation_pool = Subset(
        evaluation_base,
        original_indices.tolist(),
    )

    expected_size = (
        2 * CIFAR10_NUM_CLASSES * examples_per_class
    )
    if len(training_pool) != expected_size:
        raise RuntimeError(
            f"Target pool has {len(training_pool)} examples; "
            f"expected {expected_size}."
        )

    return (
        training_pool,
        evaluation_pool,
        target_labels,
        original_indices,
    )


def build_balanced_memberships(
    *,
    labels: np.ndarray,
    num_membership_rows: int,
    included_examples_per_class: int,
    seed: int,
) -> np.ndarray:
    """
    Select exactly half of each class-specific target pool for every model.
    """
    if num_membership_rows <= 0:
        raise ValueError(
            "num_membership_rows must be positive."
        )

    labels = np.asarray(labels, dtype=np.int64)
    memberships = np.zeros(
        (num_membership_rows, labels.shape[0]),
        dtype=bool,
    )
    rng = np.random.default_rng(seed)

    expected_candidates_per_class = (
        2 * included_examples_per_class
    )

    for row_idx in range(num_membership_rows):
        for class_id in range(CIFAR10_NUM_CLASSES):
            class_indices = np.flatnonzero(
                labels == class_id
            )

            if (
                len(class_indices)
                != expected_candidates_per_class
            ):
                raise ValueError(
                    f"Class {class_id} has {len(class_indices)} "
                    "target candidates; expected "
                    f"{expected_candidates_per_class}."
                )

            included_indices = rng.choice(
                class_indices,
                size=included_examples_per_class,
                replace=False,
            )
            memberships[
                row_idx,
                included_indices,
            ] = True

    expected_row_size = (
        CIFAR10_NUM_CLASSES
        * included_examples_per_class
    )
    row_sizes = memberships.sum(axis=1)

    if not np.all(row_sizes == expected_row_size):
        raise RuntimeError(
            "At least one membership row has the wrong size."
        )

    return memberships
