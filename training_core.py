#!/usr/bin/env python3
"""The training loop every run in this study shares.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from checkpointing import (
    SelectionState,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from experiment_protocol import ComparisonKey, ExperimentProtocol
from project_config import (
    STUDY_IDENTITY,
    RunIdentity,
    assert_output_dir_clean,
    environment_record,
    write_json,
)
from metrics import (
    SemanticConfusion,
    disparity_metrics,
    metrics_from_counts,
    confusion_counts,
    depth_metrics,
    ranking_metrics,
    select_threshold,
)
from objectives import ObjectiveSet
from reproducibility import set_epoch_recursively
from unified_stereo_dataloaders import denormalize_imagenet

logger = logging.getLogger(__name__)

_SEEDED = {"value": False}


def mark_seeded() -> None:
    """Record that global seeding has happened. Called by `seed_run`."""
    _SEEDED["value"] = True


def attach_run_log(experiment: str, variant: str, seed: int) -> Path:
    """Send every log record to a file inside the run directory.
    """
    from project_config import run_identity

    identity = run_identity(experiment, variant, seed)
    identity.directory.mkdir(parents=True, exist_ok=True)
    path = identity.directory / "run.log"

    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger(__name__).info("Logging to %s", path)
    return path


def seed_run(seed: int, deterministic: bool = False) -> int:
    """Seed everything and record it, before any stochastic object exists."""
    from reproducibility import seed_everything

    value = seed_everything(seed, deterministic_algorithms=deterministic)
    mark_seeded()
    return value


@dataclass
class TrainerConfig:
    """Per-run settings that are not part of the fixed protocol."""

    identity: RunIdentity
    protocol: ExperimentProtocol
    device: str = "cuda"
    amp: bool = True
    log_interval: int = 50
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    resume: bool = False


class Trainer:
    """Trains one run under the shared protocol."""

    def __init__(
        self,
        model: nn.Module,
        objectives: ObjectiveSet,
        config: TrainerConfig,
        comparison_key: ComparisonKey,
        split_manifest_hashes: Dict[str, str],
        selection_metric: str,
        selection_rule: str = "final_epoch",
        initialization_mode: str = "random",
        initialization_checkpoint_sha256: Optional[str] = None,
        run_summary: Optional[Dict[str, object]] = None,
    ):
        if not _SEEDED["value"]:
            raise RuntimeError(
                "seed_run() must be called before any dataset, transform, "
                "sampler, loader, model, or optimizer is constructed. "
                "Constructing the trainer without it would leave the run "
                "unreproducible."
            )
        self.model = model
        self.objectives = objectives
        self.config = config
        self.protocol = config.protocol
        self.comparison_key = comparison_key
        self.split_manifest_hashes = dict(split_manifest_hashes)
        self.initialization_mode = initialization_mode
        self.initialization_checkpoint_sha256 = initialization_checkpoint_sha256
        self.run_summary = run_summary or {}

        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model.to(self.device)

        self.directory = config.identity.directory
        assert_output_dir_clean(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.selection = SelectionState(metric=selection_metric, rule=selection_rule)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.epoch = 0
        self.global_step = 0
        self.history: List[Dict[str, object]] = []

    # -- setup -----------------------------------------------------------

    def build_optimizer(self, num_batches: int) -> None:
        protocol = self.protocol
        groups = self.model.parameter_groups(
            protocol.learning_rate_backbone, protocol.learning_rate
        )
        if protocol.optimizer != "adamw":
            raise ValueError(
                f"The protocol fixes the optimizer at adamw; got "
                f"{protocol.optimizer!r}."
            )
        self.optimizer = torch.optim.AdamW(groups, weight_decay=protocol.weight_decay)

        steps_per_epoch = self.optimizer_steps_per_epoch(num_batches)
        total_steps = max(1, steps_per_epoch * protocol.epoch_budget)
        if protocol.scheduler == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[group["lr"] for group in self.optimizer.param_groups],
                total_steps=total_steps,
                pct_start=0.1,
            )
        elif protocol.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        else:
            raise ValueError(f"Unknown scheduler {protocol.scheduler!r}.")
        logger.info(
            "Schedule: %d batches, %d optimizer steps per epoch, %d total",
            num_batches,
            steps_per_epoch,
            total_steps,
        )

    def optimizer_steps_per_epoch(self, num_batches: int) -> int:
        """Ceiling, because the final partial window takes a step too."""
        return max(1, math.ceil(num_batches / self.protocol.gradient_accumulation))

    # -- batch preparation ----------------------------------------------

    def prepare_batch(self, batch: Dict[str, object]) -> Dict[str, object]:
        """Move tensors to the device and add the model's declared range.
        """
        prepared: Dict[str, object] = {}
        for key, value in batch.items():
            prepared[key] = value.to(self.device) if torch.is_tensor(value) else value
        prepared["left_intensity"] = denormalize_imagenet(prepared["left"])
        prepared["right_intensity"] = denormalize_imagenet(prepared["right"])
        if torch.is_tensor(prepared.get("prev_left")):
            prepared["prev_intensity"] = denormalize_imagenet(prepared["prev_left"])
        return prepared

    def forward(self, batch: Dict[str, object], test_mode: bool = False) -> Dict[str, object]:
        previous = None
        if "prev_intensity" in batch and bool(batch["has_temporal"].any()):
            previous = batch["prev_intensity"]
        return self.model(
            batch["left_intensity"],
            batch["right_intensity"],
            prev_image=previous,
            radar_points=batch.get("radar_points"),
            radar_num_points=batch.get("radar_num_points"),
            intrinsics=batch.get("intrinsics"),
            extrinsics=batch.get("extrinsics"),
            test_mode=test_mode,
        )

    # -- training --------------------------------------------------------

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        self.model.freeze_batchnorm()
        accumulation = self.protocol.gradient_accumulation

        num_batches = len(loader)
        if self.config.max_train_batches is not None:
            num_batches = min(num_batches, self.config.max_train_batches)
        if num_batches == 0:
            raise RuntimeError(
                "The training loader yielded no batches. An empty epoch is a "
                "data failure."
            )

        # Window sizes, including the final partial one, so each window is
        # normalized by the number of batches it actually holds.
        window_sizes = [accumulation] * (num_batches // accumulation)
        remainder = num_batches % accumulation
        if remainder:
            window_sizes.append(remainder)

        totals: Dict[str, float] = {}
        count = 0
        skipped_steps = 0
        window_index = 0
        position_in_window = 0
        self.optimizer.zero_grad(set_to_none=True)
        started = time.time()

        for index, raw in enumerate(loader):
            if index >= num_batches:
                break
            batch = self.prepare_batch(raw)
            window = window_sizes[window_index]

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                output = self.forward(batch)
                terms = self.objectives(output, batch)
                loss = terms["total"] / window

            if loss.requires_grad:
                self.scaler.scale(loss).backward()
            else:
                # No objective had a valid sample in this batch and no
                # differentiable output was available to anchor the total.
                # Counted rather than silently ignored.
                self.detached_batches = getattr(self, "detached_batches", 0) + 1
            position_in_window += 1

            if position_in_window == window:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # The scaler skips the update when a gradient overflowed,
                # and signals it by lowering the scale. Advancing the
                # schedule on a step the optimizer did not take would put
                # the learning rate ahead of the parameters, so the
                # schedule advances only when the step was real.
                stepped = self.scaler.get_scale() >= scale_before
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and stepped:
                    self.scheduler.step()
                if not stepped:
                    skipped_steps += 1
                self.global_step += 1
                window_index += 1
                position_in_window = 0

            for name, value in terms.items():
                if torch.is_tensor(value):
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
            count += 1

            if self.config.log_interval and index % self.config.log_interval == 0:
                logger.info(
                    "epoch %d  batch %d/%d  loss %.4f",
                    self.epoch,
                    index,
                    num_batches,
                    float(terms["total"].detach()),
                )

        if position_in_window != 0:
            raise RuntimeError(
                f"{position_in_window} batches were accumulated without an "
                f"optimizer step. The final partial window must be stepped."
            )
        if window_index != len(window_sizes):
            raise RuntimeError(
                f"Took {window_index} optimizer steps for "
                f"{len(window_sizes)} planned windows."
            )

        if skipped_steps:
            logger.warning(
                "%d of %d optimizer steps were skipped on gradient overflow. "
                "A few in the first epoch are ordinary; a sustained rate "
                "means a loss term is unstable.",
                skipped_steps,
                window_index,
            )
        return {
            **{name: value / max(count, 1) for name, value in totals.items()},
            "batches": count,
            "optimizer_steps": window_index,
            "skipped_steps": skipped_steps,
            "seconds": time.time() - started,
        }

    # -- validation ------------------------------------------------------

    @torch.no_grad()
    def validate(self, loader) -> Dict[str, object]:
        """One pass over validation, reporting every task the run touches."""
        self.model.eval()
        crash_scores: List[float] = []
        crash_labels: List[int] = []
        crash_clusters: List[str] = []
        disparity_errors: List[float] = []
        depth_errors_camera: List[float] = []
        depth_errors_fused: List[float] = []
        depth_scenes: List[str] = []
        semantic = SemanticConfusion()
        semantic_seen = False
        batches = 0

        limit = self.config.max_val_batches
        for index, raw in enumerate(loader):
            if limit is not None and index >= limit:
                break
            batch = self.prepare_batch(raw)
            output = self.forward(batch, test_mode=True)
            batches += 1

            supervised = batch["crash_supervised"].bool()
            if bool(supervised.any()):
                probability = output["crash_probability"][:, 1]
                crash_scores.extend(probability[supervised].float().cpu().tolist())
                crash_labels.extend(batch["crash_label"][supervised].cpu().tolist())
                crash_clusters.extend(
                    [
                        source
                        for source, keep in zip(batch["source_id"], supervised.cpu().tolist())
                        if keep
                    ]
                )

            has_disparity = batch["has_disparity"].bool()
            if bool(has_disparity.any()):
                prediction = output["disparity"].squeeze(1)[has_disparity].cpu().numpy()
                target = batch["disparity"][has_disparity].cpu().numpy()
                valid = batch["disparity_valid"][has_disparity].cpu().numpy()
                if valid.sum() > 0:
                    disparity_errors.append(
                        disparity_metrics(prediction, target, valid)["epe"]
                    )

            has_semantic = batch["has_semantic"].bool()
            if bool(has_semantic.any()) and output.get("semantic_logits") is not None:
                prediction = (
                    output["semantic_logits"][has_semantic].argmax(1).cpu().numpy()
                )
                target = batch["semantic"][has_semantic].cpu().numpy()
                semantic.update(prediction, target)
                semantic_seen = True

            if torch.is_tensor(batch.get("depth_target")) and bool(
                batch["has_depth"].any()
            ):
                valid = batch["depth_valid"].cpu().numpy()
                target = batch["depth_target"].cpu().numpy()
                if output.get("depth_camera") is not None and valid.sum() > 0:
                    camera = output["depth_camera"].squeeze(1).float().cpu().numpy()
                    depth_errors_camera.append(
                        depth_metrics(camera, target, valid)["mae"]
                    )
                    if output.get("depth_fused") is not None:
                        fused = output["depth_fused"].squeeze(1).float().cpu().numpy()
                        depth_errors_fused.append(
                            depth_metrics(fused, target, valid)["mae"]
                        )
                    else:
                        depth_errors_fused.append(float("nan"))
                    depth_scenes.extend(batch.get("scene", ["unknown"])[:1])

        if batches == 0:
            raise RuntimeError(
                "The validation loader yielded no batches. Reporting metrics "
                "over an empty split is refused."
            )

        result: Dict[str, object] = {"batches": batches}

        if crash_labels:
            ranking = ranking_metrics(crash_labels, crash_scores)
            threshold = select_threshold(crash_labels, crash_scores)
            predictions = [
                1 if s >= threshold["threshold"] else 0 for s in crash_scores
            ]
            operating = metrics_from_counts(confusion_counts(crash_labels, predictions))
            result["crash"] = {
                **ranking,
                "selected_threshold": threshold["threshold"],
                "operating_point": operating,
                "n_sources": len(set(crash_clusters)),
            }
        if disparity_errors:
            result["disparity"] = {"epe": float(np.mean(disparity_errors))}
        if semantic_seen:
            result["semantic"] = semantic.compute()
        if depth_errors_camera:
            result["depth"] = {
                "mae_camera": float(np.nanmean(depth_errors_camera)),
                "mae_fused": float(np.nanmean(depth_errors_fused)),
                "n_scenes": len(set(depth_scenes)),
            }
        return result

    def selection_value(self, validation: Dict[str, object]) -> Optional[float]:
        metric = self.selection.metric
        if metric == "average_precision":
            crash = validation.get("crash")
            return None if crash is None else crash.get("average_precision")
        if metric == "epe":
            disparity = validation.get("disparity")
            return None if disparity is None else disparity.get("epe")
        if metric == "mae":
            depth = validation.get("depth")
            return None if depth is None else depth.get("mae_fused", depth.get("mae_camera"))
        if metric == "miou":
            semantic = validation.get("semantic")
            return None if semantic is None else semantic.get("miou")
        raise ValueError(f"Unknown selection metric {metric!r}.")

    # -- persistence -----------------------------------------------------

    def write_run_configuration(self) -> None:
        write_json(
            self.directory / "config.json",
            {
                "study_identity": STUDY_IDENTITY,
                "run": self.config.identity.to_dict(),
                "protocol": self.protocol.to_dict(),
                "objectives": self.objectives.to_dict(),
                "build_config": self.model.build_config(),
                "initialization_mode": self.initialization_mode,
                "initialization_checkpoint_sha256": self.initialization_checkpoint_sha256,
                "data": self.run_summary,
            },
        )
        write_json(
            self.directory / "comparison_key.json", self.comparison_key.to_dict()
        )
        write_json(self.directory / "environment.json", environment_record())
        write_json(
            self.directory / "split_hashes.json",
            {"manifests": self.split_manifest_hashes},
        )

    def write_history(self) -> None:
        write_json(
            self.directory / "training_history.json",
            {"run": self.config.identity.name, "epochs": self.history},
        )

    def checkpoint_payload(self, sampler_epoch: Optional[int]) -> Dict[str, object]:
        return build_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            comparison_key=self.comparison_key.to_dict(),
            split_manifest_hashes=self.split_manifest_hashes,
            epoch=self.epoch,
            global_step=self.global_step,
            selection=self.selection,
            initialization_checkpoint_sha256=self.initialization_checkpoint_sha256,
            sampler_epoch=sampler_epoch,
            history={"epochs": self.history},
        )

    def resume_if_requested(self) -> None:
        path = self.directory / "checkpoint_latest.pt"
        if not self.config.resume:
            return
        if not path.is_file():
            raise FileNotFoundError(
                f"Resume was requested but {path} is absent. A resume targets "
                f"exactly one known run rather than whichever checkpoint is "
                f"found by search."
            )
        payload = load_checkpoint(path)
        state = restore_training_state(
            payload, self.model, self.optimizer, self.scheduler, self.scaler
        )
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.selection.best_value = payload["selection_value"]
        self.selection.best_threshold = payload["selected_threshold"]
        self.selection.peak_value = payload.get("selection_peak_value")
        self.selection.peak_epoch = payload.get("selection_peak_epoch")
        self.history = list(payload.get("history", {}).get("epochs", []))
        logger.info("Resumed %s at epoch %d", self.config.identity.name, self.epoch)

    # -- the run ---------------------------------------------------------

    def run(self, train_loader, val_loader) -> Dict[str, object]:
        self.build_optimizer(
            len(train_loader)
            if self.config.max_train_batches is None
            else min(len(train_loader), self.config.max_train_batches)
        )
        self.resume_if_requested()
        self.write_run_configuration()

        budget = self.protocol.epoch_budget
        interval = self.protocol.validation_interval

        for epoch in range(self.epoch + 1, budget + 1):
            self.epoch = epoch
            reached = set_epoch_recursively(train_loader, epoch)
            if reached == 0:
                logger.warning(
                    "No dataset or sampler consumed the epoch; the temporal "
                    "sampling grid will not be redrawn."
                )
            train_stats = self.train_epoch(train_loader)

            record: Dict[str, object] = {"epoch": epoch, "train": train_stats}
            if epoch % interval == 0:
                validation = self.validate(val_loader)
                record["validation"] = validation
                value = self.selection_value(validation)
                threshold = (
                    validation.get("crash", {}).get("selected_threshold", 0.5)
                    if isinstance(validation.get("crash"), dict)
                    else 0.5
                )
                # The running maximum is tracked as a diagnostic. Under the
                # final-epoch rule it selects nothing; the gap between it
                # and the selected value measures how noisy validation is
                # at this sample size.
                self.selection.observe(value, epoch)
                if self.selection.is_improvement(value):
                    self.selection.update(value, epoch, threshold)
                    if self.selection.rule != "final_epoch":
                        save_checkpoint(
                            self.checkpoint_payload(epoch),
                            self.directory / "checkpoint_selected.pt",
                        )
                    if self.selection.rule == "final_epoch":
                        logger.info(
                            "epoch %d %s = %.6f (running peak %.6f at epoch %s)",
                            epoch,
                            self.selection.metric,
                            value,
                            self.selection.peak_value,
                            self.selection.peak_epoch,
                        )
                    else:
                        logger.info(
                            "New best %s = %.6f at epoch %d",
                            self.selection.metric,
                            value,
                            epoch,
                        )
            self.history.append(record)
            save_checkpoint(
                self.checkpoint_payload(epoch), self.directory / "checkpoint_latest.pt"
            )
            self.write_history()

        summary = {
            "run": self.config.identity.to_dict(),
            "selection": self.selection.to_dict(),
            "selected_checkpoint": (
                "checkpoint_latest.pt"
                if self.selection.rule == "final_epoch"
                else "checkpoint_selected.pt"
            ),
            "epochs_completed": self.epoch,
            "optimizer_steps": self.global_step,
        }
        write_json(self.directory / "run_summary.json", summary)
        logger.info(
            "Finished %s under the %s rule: %s = %s at epoch %s "
            "(running peak %s at epoch %s)",
            self.config.identity.name,
            self.selection.rule,
            self.selection.metric,
            self.selection.best_value,
            self.selection.best_epoch,
            self.selection.peak_value,
            self.selection.peak_epoch,
        )
        return summary


def make_comparison_key(
    model: nn.Module,
    protocol: ExperimentProtocol,
    data_config,
    active_objectives: Sequence[str],
    identity: RunIdentity,
    split_manifest_hashes: Dict[str, str],
    initialization_mode: str = "random",
    initialization_checkpoint_sha256: Optional[str] = None,
    radar_filter_mode: Optional[str] = None,
    radar_modality_dropout: Optional[float] = None,
    nuscenes_version: Optional[str] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    nuscenes_subset: Optional[Dict[str, object]] = None,
    train_sizes: Optional[Dict[str, int]] = None,
) -> ComparisonKey:
    """Assemble the key from the objects that actually govern the run.

    Reading the values off the constructed model, protocol, and data
    configuration is what makes the key trustworthy: a field cannot record
    an intention that the run did not follow.
    """
    from project_config import dict_sha256

    build = model.build_config()
    return ComparisonKey(
        model_variant=identity.variant,
        build_config_hash=dict_sha256(build),
        split_manifest_hashes=dict(split_manifest_hashes),
        data_stream=list(protocol.data_stream),
        active_objectives=list(active_objectives),
        seed=identity.seed,
        random_seed=data_config.random_seed,
        train_sizes=dict(train_sizes or {}),
        initialization_mode=initialization_mode,
        initialization_checkpoint_sha256=initialization_checkpoint_sha256,
        max_disparity=build["max_disp"],
        hidden_dims=list(build["hidden_dims"]),
        correlation_levels=build["corr_levels"],
        correlation_radius=build["corr_radius"],
        train_iterations=build["train_iters"],
        evaluation_iterations=build["eval_iters"],
        backbone_normalization=build["backbone_norm"],
        head_normalization=build["head_norm"],
        semantic_enabled=build["enable_semantic"],
        temporal_crash_mode=build["temporal_crash_mode"],
        previous_frame_spacing=data_config.ccd_previous_frame_spacing,
        previous_frame_source=data_config.ccd_previous_frame_source,
        crash_supervision_scope=protocol.crash_supervision_scope,
        crash_class_weights=list(protocol.crash_class_weights),
        ccd_balance_mode=data_config.ccd_balance_mode,
        ccd_samples_per_clip=data_config.ccd_samples_per_clip,
        ccd_frame_jitter_policy=(
            "train_only" if data_config.ccd_frame_jitter else "none"
        ),
        camera_transform=data_config.camera_transform(),
        photometric_masks={
            "photometric_applied_to": "rectified_pairs_only",
            "smoothness_applied_to": "rectified_pairs_only",
        },
        rig_disparity_scaling={
            "reference_rig": "kitti",
            "applied_in": "photometric_stereo_loss",
        },
        sensor_fusion_enabled=bool(build["enable_radar_depth"]),
        fusion_mode=build["fusion_mode"],
        positional_encoding=build["use_positional_encoding"],
        radar_filter_mode=radar_filter_mode,
        radar_modality_dropout=radar_modality_dropout,
        nuscenes_version=nuscenes_version or protocol.nuscenes_version,
        batch_size=protocol.batch_size,
        gradient_accumulation=protocol.gradient_accumulation,
        optimizer=protocol.optimizer,
        learning_rates=protocol.learning_rates(),
        weight_decay=protocol.weight_decay,
        scheduler=protocol.scheduler,
        epoch_budget=protocol.epoch_budget,
        validation_interval=protocol.validation_interval,
        checkpoint_selection_metric=protocol.checkpoint_selection_metric,
        threshold_selection_method=protocol.threshold_selection_method,
        loss_weights=dict(loss_weights or {}),
        samples_per_epoch=data_config.samples_per_epoch,
        nuscenes_subset=dict(nuscenes_subset or {"applied": False}),
    )


def assert_class_weights_match_balance(
    balance_mode: str, crash_class_weights: Sequence[float]
) -> None:
    """Refuse to correct the same class imbalance in two places.

    Oversampling to a one-to-one stream and then applying a positive class
    weight on top of it corrects twice, which leaves the decision threshold
    far from the operating point the reported metrics assume.
    """
    weights = list(crash_class_weights)
    if balance_mode == "oversample" and not all(
        abs(w - weights[0]) < 1e-9 for w in weights
    ):
        raise ValueError(
            f"ccd_balance_mode='oversample' already balances the stream, so "
            f"crash class weights must be equal; got {weights}."
        )
    if balance_mode == "weight" and all(abs(w - weights[0]) < 1e-9 for w in weights):
        raise ValueError(
            f"ccd_balance_mode='weight' leaves the stream at natural "
            f"prevalence, so the class weights must do the correcting; got "
            f"{weights}."
        )


__all__ = [
    "Trainer",
    "TrainerConfig",
    "attach_run_log",
    "seed_run",
    "mark_seeded",
    "make_comparison_key",
    "assert_class_weights_match_balance",
]
