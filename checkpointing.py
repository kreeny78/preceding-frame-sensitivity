#!/usr/bin/env python3
"""Checkpoint contents, strict loading, and the resume contract.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from project_config import (
    EXPERIMENT_SCHEMA,
    SCHEMA_FIELD,
    SchemaError,
    code_hash,
    environment_record,
    file_sha256,
    git_commit,
)
from reproducibility import capture_rng_state, restore_rng_state

logger = logging.getLogger(__name__)

CHECKPOINT_REQUIRED_KEYS = (
    SCHEMA_FIELD,
    "build_config",
    "comparison_key",
    "code_hash",
    "split_manifest_hashes",
    "initialization_checkpoint_sha256",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "rng_state",
    "epoch",
    "global_step",
    "selected_threshold",
    "selection_metric",
    "selection_value",
)

# Higher is better for these; lower is better for the rest.
_HIGHER_IS_BETTER = {"average_precision", "miou", "delta_1", "auroc"}


SELECTION_RULES = ("final_epoch", "best_epoch")


@dataclass
class SelectionState:
    """Chooses which epoch's parameters the run reports.

    """

    metric: str
    rule: str = "final_epoch"
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    best_threshold: float = 0.5
    peak_value: Optional[float] = None
    peak_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        if self.rule not in SELECTION_RULES:
            raise ValueError(
                f"Selection rule must be one of {SELECTION_RULES}, got "
                f"{self.rule!r}."
            )

    def observe(self, value: Optional[float], epoch: int) -> None:
        """Record the running maximum as a diagnostic, never as a choice.

        The peak is reported beside the selected value so the gap between
        them is visible. A large gap means validation is noisy at this
        sample size, which is a property worth knowing.
        """
        if value is None or value != value:
            return
        if self.peak_value is None or self._better(value, self.peak_value):
            self.peak_value = float(value)
            self.peak_epoch = int(epoch)

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.metric in _HIGHER_IS_BETTER else a < b

    def is_improvement(self, value: Optional[float]) -> bool:
        """Whether this epoch's parameters become the reported ones."""
        if value is None or value != value:  # NaN
            return False
        if self.rule == "final_epoch":
            # Every completed epoch replaces the last, so the run ends
            # holding the final epoch's parameters.
            return True
        if self.best_value is None:
            return True
        return self._better(value, self.best_value)

    def update(self, value: float, epoch: int, threshold: float) -> None:
        self.best_value = float(value)
        self.best_epoch = int(epoch)
        self.best_threshold = float(threshold)

    def to_dict(self) -> Dict[str, object]:
        return {
            "selection_metric": self.metric,
            "selection_rule": self.rule,
            "selected_value": self.best_value,
            "selected_epoch": self.best_epoch,
            "selected_threshold": self.best_threshold,
            "peak_value": self.peak_value,
            "peak_epoch": self.peak_epoch,
            "peak_minus_selected": (
                None
                if self.peak_value is None or self.best_value is None
                else float(self.peak_value - self.best_value)
            ),
        }


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    scaler,
    comparison_key: Dict[str, object],
    split_manifest_hashes: Dict[str, str],
    epoch: int,
    global_step: int,
    selection: SelectionState,
    initialization_checkpoint_sha256: Optional[str],
    sampler_epoch: Optional[int] = None,
    history: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Assemble every field the resume and evaluation contracts require."""
    if not hasattr(model, "build_config"):
        raise AttributeError(
            "The model must expose build_config() so an evaluator can "
            "rebuild the exact network rather than infer it from flags."
        )
    return {
        SCHEMA_FIELD: EXPERIMENT_SCHEMA,
        "build_config": model.build_config(),
        "comparison_key": dict(comparison_key),
        "code_hash": code_hash(),
        "git_commit": git_commit(),
        "environment": environment_record(),
        "split_manifest_hashes": dict(split_manifest_hashes),
        "initialization_checkpoint_sha256": initialization_checkpoint_sha256,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(sampler_epoch=sampler_epoch),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "selected_threshold": float(selection.best_threshold),
        "selection_metric": selection.metric,
        "selection_rule": selection.rule,
        "selection_value": selection.best_value,
        "selection_peak_value": selection.peak_value,
        "selection_peak_epoch": selection.peak_epoch,
        "history": history or {},
    }


def save_checkpoint(payload: Dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [k for k in CHECKPOINT_REQUIRED_KEYS if k not in payload]
    if missing:
        raise KeyError(
            f"Refusing to write an incomplete checkpoint; missing {missing}."
        )
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, object]:
    """Read a checkpoint and require this study's schema and full contents."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise SchemaError(
            f"{path} does not hold a checkpoint mapping; it cannot enter "
            f"{EXPERIMENT_SCHEMA}."
        )
    found = payload.get(SCHEMA_FIELD)
    if found != EXPERIMENT_SCHEMA:
        raise SchemaError(
            f"{path} is stamped {found!r}; this study is "
            f"{EXPERIMENT_SCHEMA!r}. Retrain under the current schema."
        )
    missing = [k for k in CHECKPOINT_REQUIRED_KEYS if k not in payload]
    if missing:
        raise SchemaError(
            f"{path} is missing required checkpoint fields {missing}. "
            f"A partially described checkpoint cannot produce a traceable "
            f"result."
        )
    return payload


def load_model_state(model: torch.nn.Module, payload: Dict[str, object]) -> None:
    """Load parameters strictly. Any key mismatch raises."""
    result = model.load_state_dict(payload["model_state"], strict=True)
    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RuntimeError(
            f"Strict load failed: {len(missing)} missing and "
            f"{len(unexpected)} unexpected parameters. Rebuild the model "
            f"from the checkpoint's build_config."
        )


def assert_build_config_matches(
    model: torch.nn.Module, payload: Dict[str, object]
) -> None:
    """Refuse a model whose construction differs from the checkpoint's.
    """
    stored = payload["build_config"]
    current = model.build_config()
    differences = {
        k: (stored.get(k), current.get(k))
        for k in set(stored) | set(current)
        if stored.get(k) != current.get(k)
    }
    if differences:
        lines = "\n".join(
            f"    {k}: checkpoint={v[0]!r} model={v[1]!r}"
            for k, v in sorted(differences.items())
        )
        raise RuntimeError(
            f"The model was built differently from the checkpoint:\n{lines}"
        )


def restore_training_state(
    payload: Dict[str, object],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    scaler,
) -> Dict[str, object]:
    """Restore everything a resumed run needs to continue identically."""
    assert_build_config_matches(model, payload)
    load_model_state(model, payload)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    sampler_epoch = restore_rng_state(payload["rng_state"])
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "sampler_epoch": sampler_epoch,
        "selected_threshold": float(payload["selected_threshold"]),
        "selection_metric": payload["selection_metric"],
        "selection_value": payload["selection_value"],
    }


def initialization_from(
    model: torch.nn.Module,
    checkpoint_path: Path,
    expected_sha256: Optional[str] = None,
) -> str:
    """Initialize a continuation run from one explicitly named checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    digest = file_sha256(checkpoint_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"{checkpoint_path} hashes to {digest}, not the required "
            f"{expected_sha256}. A continuation must start from the exact "
            f"checkpoint it declares."
        )
    payload = load_checkpoint(checkpoint_path)
    assert_build_config_matches(model, payload)
    load_model_state(model, payload)
    logger.info(
        "Initialized from %s (sha256 %s, epoch %s)",
        checkpoint_path.name,
        digest[:12],
        payload["epoch"],
    )
    return digest


__all__ = [
    "CHECKPOINT_REQUIRED_KEYS",
    "SELECTION_RULES",
    "SelectionState",
    "build_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_state",
    "assert_build_config_matches",
    "restore_training_state",
    "initialization_from",
]
