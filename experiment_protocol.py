#!/usr/bin/env python3
"""Comparison keys and the compatibility rules between two runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Sequence, Tuple

from project_config import EXPERIMENT_SCHEMA, dict_sha256

# Every field that can change what a comparison means. A run writes all of
# them; a comparison checks all of them.
COMPARISON_FIELDS: Tuple[str, ...] = (
    "experiment_schema",
    "model_variant",
    "build_config_hash",
    "split_manifest_hashes",
    "data_stream",
    "active_objectives",
    "seed",
    "random_seed",
    "initialization_mode",
    "initialization_checkpoint_sha256",
    "max_disparity",
    "hidden_dims",
    "correlation_levels",
    "correlation_radius",
    "train_iterations",
    "evaluation_iterations",
    "backbone_normalization",
    "head_normalization",
    "semantic_enabled",
    "temporal_crash_mode",
    "previous_frame_spacing",
    "previous_frame_source",
    "crash_supervision_scope",
    "crash_class_weights",
    "ccd_balance_mode",
    "ccd_samples_per_clip",
    "ccd_frame_jitter_policy",
    "camera_transform",
    "photometric_masks",
    "rig_disparity_scaling",
    "sensor_fusion_enabled",
    "fusion_mode",
    "positional_encoding",
    "radar_filter_mode",
    "radar_modality_dropout",
    "nuscenes_version",
    "batch_size",
    "gradient_accumulation",
    "optimizer",
    "learning_rates",
    "weight_decay",
    "scheduler",
    "epoch_budget",
    "validation_interval",
    "checkpoint_selection_metric",
    "threshold_selection_method",
    "loss_weights",
    "samples_per_epoch",
    "nuscenes_subset",
    "train_sizes",
)

TEMPORAL_CRASH_MODES = ("none", "difference", "concat")
PREVIOUS_FRAME_SOURCES = ("clip", "other_clip")
CRASH_SUPERVISION_SCOPES = ("labelled", "all")
FUSION_MODES = ("attention", "concat")
RADAR_MODES = ("true", "shuffled", "synthetic", "zero", "absent")
SELECTION_METRICS = ("average_precision", "epe", "mae", "miou")


@dataclass
class ExperimentProtocol:
    """Quantities held constant across every run of the campaign.
    """

    batch_size: int = 1
    gradient_accumulation: int = 8
    optimizer: str = "adamw"
    learning_rate: float = 2.0e-4
    learning_rate_backbone: float = 4.0e-5
    weight_decay: float = 1.0e-5
    scheduler: str = "onecycle"
    epoch_budget: int = 60
    validation_interval: int = 1
    crop_height: int = 320
    crop_width: int = 720
    image_height: int = 376
    image_width: int = 1240
    train_iterations: int = 22
    evaluation_iterations: int = 32
    max_disparity: int = 192
    hidden_dims: Tuple[int, int, int] = (128, 128, 128)
    correlation_levels: int = 4
    correlation_radius: int = 4
    backbone_normalization: str = "group"
    head_normalization: str = "group"
    crash_supervision_scope: str = "labelled"
    crash_class_weights: Tuple[float, float] = (1.0, 2.0)
    ccd_balance_mode: str = "weight"
    ccd_samples_per_clip: int = 5
    ccd_frame_jitter_policy: str = "train_only"
    checkpoint_selection_metric: str = "average_precision"
    threshold_selection_method: str = "validation_f1_grid"
    data_stream: Tuple[str, ...] = ("kitti", "ccd", "cityscapes", "nuscenes")
    nuscenes_version: str = "v1.0-trainval"

    # Registered resource budget.
    #
    # The full protocol as specified is not executable on a single 8 GB
    # laptop GPU within a doctoral schedule, so the budget is declared
    # rather than left implicit in whatever the datasets happen to sum to.
    # Every field here enters the comparison key, so two runs are only
    # comparable when they drew from the same budget, and a reduction is
    # visible in the artifact instead of hidden in a sampler default.
    samples_per_epoch: Optional[int] = 3000
    seeds_screening: Tuple[int, ...] = (42,)
    seeds_confirmatory: Tuple[int, ...] = (42, 43, 44)

    def __post_init__(self) -> None:
        if self.checkpoint_selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"checkpoint_selection_metric must be one of "
                f"{SELECTION_METRICS}, got "
                f"{self.checkpoint_selection_metric!r}."
            )
        if self.gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be at least 1.")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be at least 1.")

    def learning_rates(self) -> Dict[str, float]:
        return {
            "head": float(self.learning_rate),
            "backbone": float(self.learning_rate_backbone),
        }

    def camera_transform(self) -> Dict[str, object]:
        """The exact geometric transform every image passes through."""
        return {
            "resize": [int(self.image_height), int(self.image_width)],
            "resize_mode": "aspect_preserving_then_crop",
            "crop": [int(self.crop_height), int(self.crop_width)],
            "crop_shared_across_temporal_pair": True,
            "photometric_jitter": "shared_across_temporal_pair",
            "normalization": "imagenet_statistics",
        }

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        payload["crash_class_weights"] = list(self.crash_class_weights)
        payload["data_stream"] = list(self.data_stream)
        payload["seeds_screening"] = list(self.seeds_screening)
        payload["seeds_confirmatory"] = list(self.seeds_confirmatory)
        return payload


@dataclass
class ComparisonKey:
    """Everything that must match for two runs to be compared.
    """

    model_variant: str
    build_config_hash: str
    split_manifest_hashes: Dict[str, str]
    data_stream: List[str]
    active_objectives: List[str]
    seed: int
    initialization_mode: str
    initialization_checkpoint_sha256: Optional[str]
    max_disparity: int
    hidden_dims: List[int]
    correlation_levels: int
    correlation_radius: int
    train_iterations: int
    evaluation_iterations: int
    backbone_normalization: str
    head_normalization: str
    semantic_enabled: bool
    temporal_crash_mode: str
    previous_frame_spacing: int
    previous_frame_source: str
    crash_supervision_scope: str
    crash_class_weights: List[float]
    ccd_balance_mode: str
    ccd_samples_per_clip: int
    ccd_frame_jitter_policy: str
    camera_transform: Dict[str, object]
    photometric_masks: Dict[str, object]
    rig_disparity_scaling: Dict[str, object]
    sensor_fusion_enabled: bool
    fusion_mode: Optional[str]
    positional_encoding: Optional[bool]
    radar_filter_mode: Optional[str]
    radar_modality_dropout: Optional[float]
    nuscenes_version: str
    batch_size: int
    gradient_accumulation: int
    optimizer: str
    learning_rates: Dict[str, float]
    weight_decay: float
    scheduler: str
    epoch_budget: int
    validation_interval: int
    checkpoint_selection_metric: str
    threshold_selection_method: str
    loss_weights: Dict[str, float]
    samples_per_epoch: Optional[int]
    nuscenes_subset: Dict[str, object]
    random_seed: int
    train_sizes: Dict[str, int]
    experiment_schema: str = field(default=EXPERIMENT_SCHEMA)

    def __post_init__(self) -> None:
        if self.temporal_crash_mode not in TEMPORAL_CRASH_MODES:
            raise ValueError(
                f"temporal_crash_mode must be one of {TEMPORAL_CRASH_MODES}, "
                f"got {self.temporal_crash_mode!r}."
            )
        if self.previous_frame_source not in PREVIOUS_FRAME_SOURCES:
            raise ValueError(
                f"previous_frame_source must be one of "
                f"{PREVIOUS_FRAME_SOURCES}, got {self.previous_frame_source!r}."
            )
        if self.crash_supervision_scope not in CRASH_SUPERVISION_SCOPES:
            raise ValueError(
                f"crash_supervision_scope must be one of "
                f"{CRASH_SUPERVISION_SCOPES}, got "
                f"{self.crash_supervision_scope!r}."
            )
        if self.fusion_mode is not None and self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {FUSION_MODES} or None, got "
                f"{self.fusion_mode!r}."
            )
        if self.experiment_schema != EXPERIMENT_SCHEMA:
            raise ValueError(
                f"ComparisonKey carries schema {self.experiment_schema!r}; "
                f"this study is {EXPERIMENT_SCHEMA!r}."
            )
        missing = set(COMPARISON_FIELDS) - set(self.to_dict())
        if missing:
            raise RuntimeError(
                f"ComparisonKey is missing enforced fields: "
                f"{sorted(missing)}."
            )

    def to_dict(self) -> Dict[str, object]:
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["hidden_dims"] = list(self.hidden_dims)
        payload["crash_class_weights"] = list(self.crash_class_weights)
        payload["data_stream"] = list(self.data_stream)
        payload["active_objectives"] = sorted(self.active_objectives)
        return payload

    def digest(self) -> str:
        return dict_sha256(self.to_dict())


class IncomparableRuns(RuntimeError):
    """Two runs differ in a field that is not the declared treatment."""


def assert_comparable(
    key_a: Dict[str, object],
    key_b: Dict[str, object],
    name_a: str = "run A",
    name_b: str = "run B",
    treatment: Sequence[str] = (),
) -> Dict[str, object]:
    """Refuse a comparison in which anything but the treatment differs.
    """
    treatment = tuple(treatment)
    unknown = set(treatment) - set(COMPARISON_FIELDS)
    if unknown:
        raise ValueError(
            f"Declared treatment names fields that are not part of the "
            f"comparison key: {sorted(unknown)}."
        )

    missing_a = [f for f in COMPARISON_FIELDS if f not in key_a]
    missing_b = [f for f in COMPARISON_FIELDS if f not in key_b]
    if missing_a or missing_b:
        raise IncomparableRuns(
            f"Comparison key is incomplete: {name_a} is missing "
            f"{missing_a}, {name_b} is missing {missing_b}. A run that did "
            f"not record every field cannot be shown to be controlled."
        )

    differences: Dict[str, Tuple[object, object]] = {}
    for name in COMPARISON_FIELDS:
        if _normalise(key_a[name]) != _normalise(key_b[name]):
            differences[name] = (key_a[name], key_b[name])

    unexpected = {k: v for k, v in differences.items() if k not in treatment}
    if unexpected:
        lines = "\n".join(
            f"    {k}: {name_a}={v[0]!r} {name_b}={v[1]!r}"
            for k, v in sorted(unexpected.items())
        )
        raise IncomparableRuns(
            f"{name_a} and {name_b} differ outside the declared treatment "
            f"{list(treatment)}:\n{lines}"
        )

    unchanged_treatment = [t for t in treatment if t not in differences]
    return {
        "treatment": list(treatment),
        "differed": sorted(differences),
        "treatment_without_difference": unchanged_treatment,
        "controlled": True,
    }


def _normalise(value):
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, float):
        return round(value, 12)
    return value


__all__ = [
    "COMPARISON_FIELDS",
    "TEMPORAL_CRASH_MODES",
    "PREVIOUS_FRAME_SOURCES",
    "CRASH_SUPERVISION_SCOPES",
    "FUSION_MODES",
    "RADAR_MODES",
    "SELECTION_METRICS",
    "ExperimentProtocol",
    "ComparisonKey",
    "IncomparableRuns",
    "assert_comparable",
]
