#!/usr/bin/env python3
"""Train the unified model: stereo, semantics, and temporal crash.

 python train_unified_base.py --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from experiment_protocol import ExperimentProtocol
from project_config import SPLITS_ROOT, run_identity
from objectives import LossWeights, ObjectiveSet
from training_core import (
    Trainer,
    attach_run_log,
    TrainerConfig,
    assert_class_weights_match_balance,
    make_comparison_key,
    seed_run,
)
from unified_stereo_dataloaders import (
    DataConfig,
    create_visual_dataloaders,
    manifest_hashes,
)
from model import UnifiedStereoSafetyModel

logger = logging.getLogger(__name__)

VARIANT = "unified_base"
ACTIVE_OBJECTIVES = (
    "supervised_disparity",
    "photometric_stereo",
    "smoothness",
    "semantic",
    "temporal_crash",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the unified stereo, semantic, and temporal-crash model."
    )
    parser.add_argument("--experiment", default="objective_isolation")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--temporal_crash_mode",
        default="difference",
        choices=("none", "difference", "concat"),
        help="How the earlier frame reaches the crash head.",
    )
    parser.add_argument(
        "--previous_frame_spacing",
        type=int,
        default=1,
        help="Gap in video frames between the current and earlier frame.",
    )
    parser.add_argument(
        "--previous_frame_source",
        default="clip",
        choices=("clip", "other_clip"),
        help="Own recording, or the mismatched-frame control.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--splits_dir", type=Path, default=SPLITS_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument(
        "--loss_weight_semantic",
        type=float,
        default=None,
        help="Multiplier on the registered semantic weight, for the "
        "sensitivity analysis.",
    )
    parser.add_argument(
        "--loss_weight_crash",
        type=float,
        default=None,
        help="Multiplier on the registered temporal crash weight.",
    )
    parser.add_argument(
        "--loss_weight_depth",
        type=float,
        default=None,
        help="Multiplier on the registered metric depth weight.",
    )
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=None,
        help="Override the registered epoch budget.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args(argv)


def scaled_loss_weights(args: argparse.Namespace) -> LossWeights:
    """Apply the sensitivity multipliers to the registered weights.

    The sensitivity analysis varies the auxiliary weights around the
    registered setting rather than sweeping a full grid, which would cost
    more than the campaign can afford and would answer a question nobody
    asked. Its purpose is to show that the chosen configuration is not a
    fragile isolated point. Every multiplier lands in the comparison key,
    so a sensitivity run and the registered run differ in one declared
    field.
    """
    weights = LossWeights()
    if args.loss_weight_semantic is not None:
        weights.semantic *= float(args.loss_weight_semantic)
    if args.loss_weight_crash is not None:
        weights.temporal_crash *= float(args.loss_weight_crash)
    if args.loss_weight_depth is not None:
        weights.metric_depth *= float(args.loss_weight_depth)
    return weights


def build_and_run(args: argparse.Namespace) -> dict:
    protocol = ExperimentProtocol(
        data_stream=("kitti", "ccd", "cityscapes"),
        checkpoint_selection_metric="average_precision",
    )
    if args.epochs is not None:
        protocol.epoch_budget = int(args.epochs)

    # Seeding happens here, after argument validation and before any
    # dataset, transform, sampler, loader, model, or optimizer exists.
    seed_run(args.seed)

    data_config = DataConfig(
        image_height=protocol.image_height,
        image_width=protocol.image_width,
        crop_height=protocol.crop_height,
        crop_width=protocol.crop_width,
        max_disparity=protocol.max_disparity,
        batch_size=protocol.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed,
        splits_dir=args.splits_dir,
        ccd_samples_per_clip=protocol.ccd_samples_per_clip,
        ccd_balance_mode=protocol.ccd_balance_mode,
        ccd_previous_frame_spacing=args.previous_frame_spacing,
        ccd_previous_frame_source=args.previous_frame_source,
        samples_per_epoch=args.samples_per_epoch or protocol.samples_per_epoch,
    )
    assert_class_weights_match_balance(
        data_config.ccd_balance_mode, protocol.crash_class_weights
    )

    train_loader, val_loader, summary = create_visual_dataloaders(
        data_config, protocol.data_stream
    )

    model = UnifiedStereoSafetyModel(
        max_disp=protocol.max_disparity,
        train_iters=protocol.train_iterations,
        eval_iters=protocol.evaluation_iterations,
        corr_levels=protocol.correlation_levels,
        corr_radius=protocol.correlation_radius,
        hidden_dims=list(protocol.hidden_dims),
        enable_semantic=True,
        temporal_crash_mode=args.temporal_crash_mode,
        head_norm=protocol.head_normalization,
        backbone_norm=protocol.backbone_normalization,
    )

    weights = scaled_loss_weights(args)
    objectives = ObjectiveSet(
        active=ACTIVE_OBJECTIVES,
        weights=weights,
        max_disparity=protocol.max_disparity,
        crash_class_weights=protocol.crash_class_weights,
        crash_scope=protocol.crash_supervision_scope,
    )

    identity = run_identity(args.experiment, VARIANT, args.seed)
    hashes = manifest_hashes(args.splits_dir)
    key = make_comparison_key(
        model,
        protocol,
        data_config,
        ACTIVE_OBJECTIVES,
        identity,
        hashes,
        train_sizes=summary["train_sizes"],
        initialization_mode="random",
        loss_weights=weights.to_dict(),
    )

    trainer = Trainer(
        model=model,
        objectives=objectives,
        config=TrainerConfig(
            identity=identity,
            protocol=protocol,
            device=args.device,
            amp=not args.no_amp,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        ),
        comparison_key=key,
        split_manifest_hashes=hashes,
        selection_metric="average_precision",
        run_summary=summary,
    )
    return trainer.run(train_loader, val_loader)


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args(argv)
    attach_run_log(args.experiment, VARIANT, args.seed)
    build_and_run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
