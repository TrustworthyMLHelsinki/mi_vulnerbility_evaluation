# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0.
#
# Adapted from Google Research Big Transfer and the BiT FiLM configuration
# used by cambridge-mlg/dp-few-shot.

"""BiT-M-R50x1 feature extractor and FiLM classifier."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    """Convolution with per-output-channel weight standardization."""

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        variance, mean = torch.var_mean(
            self.weight,
            dim=(1, 2, 3),
            keepdim=True,
            unbiased=False,
        )
        standardized_weight = (
            self.weight - mean
        ) / torch.sqrt(variance + 1e-10)

        return F.conv2d(
            inputs,
            standardized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> StdConv2d:
    return StdConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> StdConv2d:
    return StdConv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


def tf_to_torch(array: np.ndarray) -> torch.Tensor:
    """Convert TensorFlow HWIO convolution weights to PyTorch OIHW."""
    if array.ndim == 4:
        array = array.transpose(3, 2, 0, 1)
    return torch.from_numpy(array)


class PreActBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        middle_channels: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.gn1 = nn.GroupNorm(32, in_channels)
        self.conv1 = conv1x1(
            in_channels,
            middle_channels,
        )

        self.gn2 = nn.GroupNorm(32, middle_channels)
        self.conv2 = conv3x3(
            middle_channels,
            middle_channels,
            stride=stride,
        )

        self.gn3 = nn.GroupNorm(32, middle_channels)
        self.conv3 = conv1x1(
            middle_channels,
            out_channels,
        )
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = conv1x1(
                in_channels,
                out_channels,
                stride=stride,
            )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        output = self.relu(self.gn1(inputs))

        residual = inputs
        if hasattr(self, "downsample"):
            residual = self.downsample(output)

        output = self.conv1(output)
        output = self.conv2(
            self.relu(self.gn2(output))
        )
        output = self.conv3(
            self.relu(self.gn3(output))
        )
        return output + residual

    def load_from(
        self,
        weights: Mapping[str, np.ndarray],
        prefix: str,
    ) -> None:
        convolution_name = "standardized_conv2d"

        with torch.no_grad():
            self.conv1.weight.copy_(
                tf_to_torch(
                    weights[
                        f"{prefix}a/{convolution_name}/kernel"
                    ]
                )
            )
            self.conv2.weight.copy_(
                tf_to_torch(
                    weights[
                        f"{prefix}b/{convolution_name}/kernel"
                    ]
                )
            )
            self.conv3.weight.copy_(
                tf_to_torch(
                    weights[
                        f"{prefix}c/{convolution_name}/kernel"
                    ]
                )
            )

            for module_name, branch_name in (
                ("gn1", "a"),
                ("gn2", "b"),
                ("gn3", "c"),
            ):
                group_norm = getattr(self, module_name)
                group_norm.weight.copy_(
                    tf_to_torch(
                        weights[
                            f"{prefix}{branch_name}/"
                            "group_norm/gamma"
                        ]
                    )
                )
                group_norm.bias.copy_(
                    tf_to_torch(
                        weights[
                            f"{prefix}{branch_name}/"
                            "group_norm/beta"
                        ]
                    )
                )

            if hasattr(self, "downsample"):
                self.downsample.weight.copy_(
                    tf_to_torch(
                        weights[
                            f"{prefix}a/proj/"
                            f"{convolution_name}/kernel"
                        ]
                    )
                )


class ResNetV2FeatureExtractor(nn.Module):
    """BiT-M-R50x1 without its pretraining classifier."""

    def __init__(self) -> None:
        super().__init__()

        self.output_dim = 2048

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("pad", nn.ConstantPad2d(1, 0.0)),
                    (
                        "pool",
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=2,
                            padding=0,
                        ),
                    ),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        self._make_block(
                            units=3,
                            in_channels=64,
                            out_channels=256,
                            middle_channels=64,
                            stride=1,
                        ),
                    ),
                    (
                        "block2",
                        self._make_block(
                            units=4,
                            in_channels=256,
                            out_channels=512,
                            middle_channels=128,
                            stride=2,
                        ),
                    ),
                    (
                        "block3",
                        self._make_block(
                            units=6,
                            in_channels=512,
                            out_channels=1024,
                            middle_channels=256,
                            stride=2,
                        ),
                    ),
                    (
                        "block4",
                        self._make_block(
                            units=3,
                            in_channels=1024,
                            out_channels=2048,
                            middle_channels=512,
                            stride=2,
                        ),
                    ),
                ]
            )
        )

        self.head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "gn",
                        nn.GroupNorm(32, self.output_dim),
                    ),
                    ("relu", nn.ReLU(inplace=True)),
                    (
                        "avg",
                        nn.AdaptiveAvgPool2d(1),
                    ),
                ]
            )
        )

    @staticmethod
    def _make_block(
        units: int,
        in_channels: int,
        out_channels: int,
        middle_channels: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [
            (
                "unit01",
                PreActBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    middle_channels=middle_channels,
                    stride=stride,
                ),
            )
        ]

        layers.extend(
            (
                f"unit{unit_idx:02d}",
                PreActBottleneck(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    middle_channels=middle_channels,
                    stride=1,
                ),
            )
            for unit_idx in range(2, units + 1)
        )
        return nn.Sequential(OrderedDict(layers))

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        features = self.head(
            self.body(self.root(inputs))
        )
        return features[..., 0, 0]

    def load_from(
        self,
        weights: Mapping[str, np.ndarray],
        prefix: str = "resnet/",
    ) -> None:
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf_to_torch(
                    weights[
                        f"{prefix}root_block/"
                        "standardized_conv2d/kernel"
                    ]
                )
            )
            self.head.gn.weight.copy_(
                tf_to_torch(
                    weights[f"{prefix}group_norm/gamma"]
                )
            )
            self.head.gn.bias.copy_(
                tf_to_torch(
                    weights[f"{prefix}group_norm/beta"]
                )
            )

            for block_name, block in (
                self.body.named_children()
            ):
                for unit_name, unit in (
                    block.named_children()
                ):
                    unit.load_from(
                        weights,
                        prefix=(
                            f"{prefix}{block_name}/"
                            f"{unit_name}/"
                        ),
                    )


class BiTFiLMClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.feature_extractor = (
            ResNetV2FeatureExtractor()
        )
        self.classifier = nn.Linear(
            self.feature_extractor.output_dim,
            num_classes,
        )

        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        return self.classifier(features)


def configure_film(
    model: BiTFiLMClassifier,
) -> list[str]:
    """
    Train bottleneck ``gn3`` affine parameters, final GroupNorm affine
    parameters, and the task classifier.
    """
    for parameter in (
        model.feature_extractor.parameters()
    ):
        parameter.requires_grad = False

    film_names: list[str] = []

    for name, parameter in (
        model.feature_extractor.named_parameters()
    ):
        if ".gn3." in name or name.startswith(
            "head.gn."
        ):
            parameter.requires_grad = True
            film_names.append(name)

    if not film_names:
        raise RuntimeError(
            "No FiLM parameters were selected."
        )

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return film_names


def load_bit_checkpoint(
    checkpoint_path: Path,
) -> dict[str, np.ndarray]:
    """Load the NPZ once and reuse it for all target models."""
    with np.load(
        checkpoint_path,
        allow_pickle=False,
    ) as archive:
        return {
            key: archive[key]
            for key in archive.files
        }


def build_bit_m_r50x1_film(
    *,
    bit_weights: Mapping[str, np.ndarray],
    num_classes: int,
) -> BiTFiLMClassifier:
    model = BiTFiLMClassifier(
        num_classes=num_classes
    )
    model.feature_extractor.load_from(bit_weights)
    configure_film(model)
    return model
