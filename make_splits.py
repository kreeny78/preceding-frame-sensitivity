#!/usr/bin/env python3
"""Generate the split manifests this study partitions its data with.

"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from project_config import (
    EXPERIMENT_SCHEMA,
    SPLITS_ROOT,
    read_json,
    text_sha256,
    write_json,
)
from unified_stereo_dataloaders import resolve_dataset_root

logger = logging.getLogger(__name__)


def _fingerprint(items: Sequence[str]) -> str:
    """A digest of the sorted unit list, identifying the source contents."""
    return text_sha256("\n".join(sorted(items)))


def _manifest(
    dataset: str,
    split_unit: str,
    splits: Dict[str, List[str]],
    seed: int,
    configuration: Dict[str, object],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    everything = [unit for units in splits.values() for unit in units]
    counts = {name: len(units) for name, units in splits.items()}
    duplicated = [unit for unit, n in Counter(everything).items() if n > 1]
    if duplicated:
        raise RuntimeError(
            f"{dataset}: {len(duplicated)} units appear in more than one "
            f"partition, starting with {duplicated[:5]}."
        )
    meta = {
        "dataset": dataset,
        "split_unit": split_unit,
        "seed": int(seed),
        "counts": counts,
        "total_units": len(everything),
        "fingerprint": _fingerprint(everything),
        "configuration": configuration,
    }
    meta.update(extra or {})
    return {"splits": {k: sorted(v) for k, v in splits.items()}, "meta": meta}


# ---------------------------------------------------------------------------
# CCD
# ---------------------------------------------------------------------------


def probe_readable(videos: Sequence[Path]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Open every clip once and report which cannot be decoded.

    A corrupt or truncated file raises at training time, in a worker
    process, after the run has already started. Probing here converts that
    into a property of the partition: the unreadable recordings are named
    in the manifest and excluded from every split, so the exclusion is a
    recorded fact rather than a silent skip inside the loader.
    """
    import cv2

    readable: List[str] = []
    unreadable: List[Dict[str, str]] = []
    for video in videos:
        identifier = f"{video.parent.name}/{video.stem}"
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            unreadable.append({"recording": identifier, "reason": "cannot be opened"})
            continue
        ok, _ = capture.read()
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not ok:
            unreadable.append({"recording": identifier, "reason": "no decodable frame"})
        elif frames < 50:
            unreadable.append(
                {"recording": identifier, "reason": f"only {frames} frames, fifty required"}
            )
        else:
            readable.append(identifier)
    return readable, unreadable


def build_ccd(
    root: Path, seed: int, ratios: Tuple[float, float, float]
) -> Dict[str, object]:
    """Partition dashcam recordings, stratified by class.

    The unit is the recording. Frames drawn from one recording share a
    camera, a scene, and a compression history, so a partition that split
    frames would leak the recording across the boundary and inflate every
    number computed on the other side.
    """
    crash_dir = None
    for name in ("Crash-1500", "Crash"):
        if (root / name).is_dir():
            crash_dir = root / name
            break
    if crash_dir is None:
        raise FileNotFoundError(f"Neither Crash-1500 nor Crash exists under {root}.")
    normal_dir = root / "Normal"
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"{normal_dir} is absent.")

    logger.info("Probing CCD recordings for decodability")
    crash, crash_bad = probe_readable(sorted(crash_dir.glob("*.mp4")))
    normal, normal_bad = probe_readable(sorted(normal_dir.glob("*.mp4")))
    unreadable = crash_bad + normal_bad
    if unreadable:
        logger.warning(
            "%d CCD recordings cannot be decoded and are excluded from every "
            "partition; they are named in the manifest.",
            len(unreadable),
        )
        for entry in unreadable[:10]:
            logger.warning("  %s: %s", entry["recording"], entry["reason"])
    if not crash or not normal:
        raise RuntimeError(
            f"CCD holds {len(crash)} crash and {len(normal)} normal "
            f"recordings under {root}; both classes are required."
        )

    train_ratio, val_ratio, test_ratio = ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"CCD ratios must sum to one, got {total}.")

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for group in (crash, normal):
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    class_counts = {
        name: {
            "crash": sum(1 for u in units if u.startswith(crash_dir.name + "/")),
            "normal": sum(1 for u in units if u.startswith(normal_dir.name + "/")),
        }
        for name, units in splits.items()
    }

    return _manifest(
        dataset="ccd",
        split_unit="source_recording",
        splits=splits,
        seed=seed,
        configuration={
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "stratified_by": "clip_class",
            "crash_directory": crash_dir.name,
            "normal_directory": normal_dir.name,
            "root": str(root),
        },
        extra={
            "class_counts": class_counts,
            "unreadable_recordings": unreadable,
            "n_unreadable": len(unreadable),
            "probe": "every recording was opened and one frame decoded",
        },
    )


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------


def build_kitti(root: Path, seed: int, train_ratio: float) -> Dict[str, object]:
    """Split the 200 annotated training pairs into train and local val."""
    base = root / "data_scene_flow" / "training"
    if not base.is_dir():
        base = root / "training"
    left_dir = base / "image_2"
    if not left_dir.is_dir():
        raise FileNotFoundError(f"{left_dir} is absent.")

    names = sorted(p.stem for p in left_dir.glob("*_10.png"))
    if not names:
        names = sorted(p.stem for p in left_dir.glob("*.png"))
    if not names:
        raise RuntimeError(f"No KITTI images found under {left_dir}.")

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio))
    splits = {"train": shuffled[:n_train], "val": shuffled[n_train:]}

    return _manifest(
        dataset="kitti",
        split_unit="annotated_training_pair",
        splits=splits,
        seed=seed,
        configuration={
            "train_ratio": train_ratio,
            "source": "the annotated training pairs of the stereo benchmark",
            "root": str(root),
        },
        extra={
            "validation_is_local": True,
            "note": (
                "The validation partition is a chosen split of the annotated "
                "training pairs. It is not the withheld benchmark test set "
                "and no number computed on it is a benchmark result."
            ),
        },
    )


# ---------------------------------------------------------------------------
# nuScenes
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------


def build_cityscapes(root: Path) -> Dict[str, object]:
    """Record the official partition by city, with the counts found."""
    left = root / "leftImg8bit"
    if not left.is_dir():
        raise FileNotFoundError(f"{left} is absent.")

    splits: Dict[str, List[str]] = {}
    image_counts: Dict[str, Dict[str, int]] = {}
    directories: Dict[str, str] = {}
    for split in ("train", "val"):
        split_dir = left / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{split_dir} is absent.")
        cities = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if not cities:
            raise RuntimeError(f"{split_dir} holds no city directories.")
        splits[split] = cities
        directories[split] = str(split_dir)
        image_counts[split] = {
            city: len(list((split_dir / city).glob("*_leftImg8bit.png")))
            for city in cities
        }

    return _manifest(
        dataset="cityscapes",
        split_unit="city",
        splits=splits,
        seed=0,
        configuration={
            "source": "the official train and validation partition",
            "root": str(root),
        },
        extra={
            "directories": directories,
            "image_counts": image_counts,
            "images_total": {
                split: sum(counts.values()) for split, counts in image_counts.items()
            },
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the schema-versioned split manifests."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=SPLITS_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ccd", "kitti", "cityscapes"],
        choices=["ccd", "kitti", "cityscapes"],
    )
    parser.add_argument("--ccd_train", type=float, default=0.7)
    parser.add_argument("--ccd_val", type=float, default=0.15)
    parser.add_argument("--ccd_test", type=float, default=0.15)
    parser.add_argument("--kitti_train", type=float, default=0.8)
    
    parser.add_argument(
        "--max_train_scenes",
        type=int,
        default=None,
        help="Declared scene budget for the training partition.",
    )
    parser.add_argument(
        "--max_val_scenes",
        type=int,
        default=None,
        help="Declared scene budget for the validation partition.",
    )
    parser.add_argument("--subset_seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    if "ccd" in args.datasets:
        manifest = build_ccd(
            resolve_dataset_root("ccd"),
            args.seed,
            (args.ccd_train, args.ccd_val, args.ccd_test),
        )
        written.append(write_json(out_dir / "ccd_splits.json", manifest))
    if "kitti" in args.datasets:
        manifest = build_kitti(
            resolve_dataset_root("kitti"), args.seed, args.kitti_train
        )
        written.append(write_json(out_dir / "kitti_splits.json", manifest))
    if "cityscapes" in args.datasets:
        manifest = build_cityscapes(resolve_dataset_root("cityscapes"))
        written.append(write_json(out_dir / "cityscapes_splits.json", manifest))

    for path in written:
        logger.info("wrote %s", path)
    logger.info("schema %s", EXPERIMENT_SCHEMA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
