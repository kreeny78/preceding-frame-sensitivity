#!/usr/bin/env python3
"""Datasets, transforms, heterogeneous collation, and sampling.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from project_config import (
    CANONICAL_SPLIT_MANIFESTS,
    SPLITS_ROOT,
    SchemaError,
    file_sha256,
    read_json,
)
from reproducibility import loader_generator, worker_init_fn

logger = logging.getLogger(__name__)

try:  # decoding CCD clips
    import cv2
except ImportError:  # pragma: no cover - reported at first use
    cv2 = None

DATASET_NAMES = ("kitti", "ccd", "cityscapes", "nuscenes", "nexar")
DATASET_PATHS_FILENAME = "dataset_paths.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREVIOUS_FRAME_SOURCES = ("clip", "other_clip")
TEMPORAL_JITTER_POLICIES = ("shared", "asymmetric")

# Directory markers that identify a dataset root, so a path that exists
# but points at something else is rejected rather than accepted.
_DATASET_MARKERS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "kitti": (("data_scene_flow/training/image_2",), ("training/image_2",)),
    "ccd": (("Crash-1500", "Normal"), ("Crash", "Normal")),
    "cityscapes": (("leftImg8bit", "gtFine"),),
    "nuscenes": (("maps",), ("v1.0-trainval",), ("v1.0-mini",)),
    "nexar": (("videos",), ("train",)),
}


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------


def _dataset_paths_file() -> Path:
    from project_config import PROJECT_ROOT

    return PROJECT_ROOT / DATASET_PATHS_FILENAME


def _configured_paths() -> Dict[str, str]:
    path = _dataset_paths_file()
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required. Write a JSON object mapping each of "
            f"{list(DATASET_NAMES)} to its root directory. Dataset locations "
            f"are declared once, in one file, rather than guessed from a "
            f"search order."
        )
    # utf-8-sig rather than utf-8: several Windows editors and shells
    # prefix a byte-order mark, which is invisible in the file and fatal to
    # a strict utf-8 decode. The two encodings are identical when no mark
    # is present.
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a JSON object of dataset roots.")
    return {k: str(v) for k, v in payload.items()}


def dataset_layout(name: str) -> str:
    """The directory layout a root must present, for error messages."""
    options = _DATASET_MARKERS[name]
    return " or ".join("/".join(option) for option in options)


def validate_dataset_dir(name: str, path: Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    for option in _DATASET_MARKERS[name]:
        if all((path / marker).exists() for marker in option):
            return True
    return False


def resolve_dataset_root(name: str) -> Path:
    """The declared root for a dataset, validated. Raises when unusable."""
    if name not in DATASET_NAMES:
        raise KeyError(f"Unknown dataset {name!r}; expected one of {DATASET_NAMES}.")
    configured = _configured_paths()
    if name not in configured:
        raise KeyError(
            f"{_dataset_paths_file()} declares no root for {name!r}. Add it; "
            f"the expected layout is {dataset_layout(name)}."
        )
    root = Path(configured[name])
    if not root.exists():
        raise FileNotFoundError(
            f"{name} root {root} does not exist. Correct "
            f"{_dataset_paths_file()} or mount the volume."
        )
    if not validate_dataset_dir(name, root):
        raise NotADirectoryError(
            f"{name} root {root} exists but does not present the expected "
            f"layout {dataset_layout(name)}. This is a wrong path rather "
            f"than a missing dataset."
        )
    return root


def assert_datasets_available(names: Sequence[str]) -> Dict[str, Path]:
    return {name: resolve_dataset_root(name) for name in names}


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def load_split_manifest(splits_dir: Optional[Path], name: str) -> Dict:
    """Read one schema-versioned manifest and validate its structure."""
    splits_dir = Path(splits_dir or SPLITS_ROOT)
    path = splits_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest {path} is absent. Generate the partitions with "
            f"make_splits.py before training; a partition invented at load "
            f"time is not reproducible."
        )
    manifest = read_json(path)
    for key in ("splits", "meta"):
        if key not in manifest:
            raise SchemaError(f"{path} has no {key!r} section.")
    meta = manifest["meta"]
    for key in ("split_unit", "dataset", "seed", "counts", "fingerprint"):
        if key not in meta:
            raise SchemaError(
                f"{path} meta is missing {key!r}. A manifest that does not "
                f"record how it was built cannot be audited."
            )
    assert_split_disjoint(manifest, str(path))
    return manifest


def assert_split_disjoint(manifest: Dict, source: str) -> None:
    """No unit may appear in more than one partition."""
    seen: Dict[str, str] = {}
    for split_name, units in manifest["splits"].items():
        for unit in units:
            key = str(unit)
            if key in seen:
                raise SchemaError(
                    f"{source}: unit {key!r} appears in both "
                    f"{seen[key]!r} and {split_name!r}. Partitions must be "
                    f"disjoint at the level of "
                    f"{manifest['meta']['split_unit']}."
                )
            seen[key] = split_name


def manifest_hashes(
    splits_dir: Optional[Path], names: Sequence[str] = CANONICAL_SPLIT_MANIFESTS
) -> Dict[str, str]:
    splits_dir = Path(splits_dir or SPLITS_ROOT)
    out = {}
    for name in names:
        path = splits_dir / name
        out[name] = file_sha256(path) if path.is_file() else "absent"
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Every setting that changes what a sample contains.
    """

    image_height: int = 376
    image_width: int = 1240
    crop_height: int = 320
    crop_width: int = 720
    max_disparity: int = 192

    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    drop_last: bool = False
    random_seed: int = 42

    splits_dir: Optional[Path] = None

    ccd_samples_per_clip: int = 5
    ccd_frame_stride: int = 10
    ccd_max_train_samples: Optional[int] = 3000
    ccd_max_eval_samples: Optional[int] = 2000
    ccd_anticipation_horizon: int = 15
    ccd_fps: float = 10.0
    ccd_frame_jitter: bool = True
    ccd_balance_mode: str = "weight"
    ccd_previous_frame_spacing: int = 1
    ccd_previous_frame_source: str = "clip"

    temporal_jitter_policy: str = "shared"
    use_geometric_augmentation: bool = True

    dataset_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "kitti": 1.0,
            "ccd": 1.0,
            "cityscapes": 1.0,
            "nuscenes": 1.0,
        }
    )
    samples_per_epoch: Optional[int] = None
    smoke_batches_per_dataset: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ccd_previous_frame_source not in PREVIOUS_FRAME_SOURCES:
            raise ValueError(
                f"ccd_previous_frame_source must be one of "
                f"{PREVIOUS_FRAME_SOURCES}, got "
                f"{self.ccd_previous_frame_source!r}."
            )
        if self.temporal_jitter_policy not in TEMPORAL_JITTER_POLICIES:
            raise ValueError(
                f"temporal_jitter_policy must be one of "
                f"{TEMPORAL_JITTER_POLICIES}, got "
                f"{self.temporal_jitter_policy!r}."
            )
        if self.ccd_balance_mode not in ("weight", "oversample", "none"):
            raise ValueError(
                f"ccd_balance_mode must be 'weight', 'oversample', or "
                f"'none', got {self.ccd_balance_mode!r}."
            )
        if isinstance(self.ccd_previous_frame_spacing, bool) or not isinstance(
            self.ccd_previous_frame_spacing, int
        ):
            raise TypeError(
                f"ccd_previous_frame_spacing must be an int gap in video "
                f"frames, got {self.ccd_previous_frame_spacing!r}. A value "
                f"of another type produces a plausible null result, so it "
                f"is a hard failure."
            )
        if self.ccd_previous_frame_spacing < 1:
            raise ValueError(
                "ccd_previous_frame_spacing must be at least one frame; the "
                "temporal baseline is a property of this setting alone."
            )

    def camera_transform(self) -> Dict[str, object]:
        return {
            "resize": [self.image_height, self.image_width],
            "resize_mode": "aspect_preserving_then_pad_or_crop",
            "crop": [self.crop_height, self.crop_width],
            "crop_shared_across_temporal_pair": True,
            "photometric_jitter": self.temporal_jitter_policy,
            "normalization": "imagenet_statistics",
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


# A removable volume that suspends can stay unavailable for tens of
# seconds while the bus re-enumerates and the device spins back up, so the
# schedule has to outlast that rather than merely survive a brief stall.
# The delay doubles and then holds, giving roughly seventy-five seconds
# across the full sequence.
IO_RETRIES = int(os.environ.get("UNIFIED_SAFETY_IO_RETRIES", "8"))
IO_RETRY_MAX_DELAY_S = 15.0


def io_retry_delay(attempt: int) -> float:
    return min(2.0 ** (attempt - 1), IO_RETRY_MAX_DELAY_S)


def open_image_with_retry(path: Path, retries: int = IO_RETRIES) -> Image.Image:
    """Open an image, retrying a transient failure of the storage device.
    """
    last: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            image = Image.open(path)
            image.load()
            return image
        except OSError as error:
            last = error
            if attempt == retries:
                break
            delay = io_retry_delay(attempt)
            logger.warning(
                "Read of %s failed on attempt %d of %d (%s); retrying in "
                "%.1f s.",
                path,
                attempt,
                retries,
                error,
                delay,
            )
            time.sleep(delay)
    raise OSError(
        f"{path} could not be read after {retries} attempts. The volume may "
        f"be unavailable, or the file may be damaged."
    ) from last


def normalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor - mean) / std


def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """Recover [0, 1] intensities, the range the model's contract declares."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device)
    shape = (1, -1, 1, 1) if tensor.dim() == 4 else (-1, 1, 1)
    return (tensor * std.view(*shape) + mean.view(*shape)).clamp(0.0, 1.0)


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, dtype=np.uint8, copy=True)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


@dataclass
class GeometryPlan:
    """The exact resize, pad, and crop one sample receives.

    Constructed once per sample and applied to every image in it, so the
    current frame, the right frame, and the earlier frame share pixel
    coordinates. It is also the record of how the intrinsics must be
    adjusted, which the nuScenes loader consumes.
    """

    native_h: int
    native_w: int
    scale: float
    resized_h: int
    resized_w: int
    pad_top: int
    pad_left: int
    target_h: int
    target_w: int
    crop_top: int
    crop_left: int
    crop_h: int
    crop_w: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "native": [self.native_h, self.native_w],
            "scale": self.scale,
            "resized": [self.resized_h, self.resized_w],
            "pad": [self.pad_top, self.pad_left],
            "target": [self.target_h, self.target_w],
            "crop_origin": [self.crop_top, self.crop_left],
            "crop_size": [self.crop_h, self.crop_w],
        }


def plan_geometry(
    native_h: int,
    native_w: int,
    target_h: int,
    target_w: int,
    crop_h: Optional[int],
    crop_w: Optional[int],
    rng: Optional[random.Random],
) -> GeometryPlan:
    """Aspect-preserving resize, symmetric pad, then a crop.

       """
    scale = min(target_h / native_h, target_w / native_w)
    resized_h = max(1, int(round(native_h * scale)))
    resized_w = max(1, int(round(native_w * scale)))
    pad_top = max(0, (target_h - resized_h) // 2)
    pad_left = max(0, (target_w - resized_w) // 2)

    if crop_h is None or crop_w is None:
        crop_h, crop_w = target_h, target_w
    crop_h = min(crop_h, target_h)
    crop_w = min(crop_w, target_w)
    if rng is None:
        crop_top = (target_h - crop_h) // 2
        crop_left = (target_w - crop_w) // 2
    else:
        crop_top = rng.randint(0, target_h - crop_h)
        crop_left = rng.randint(0, target_w - crop_w)

    return GeometryPlan(
        native_h=native_h,
        native_w=native_w,
        scale=scale,
        resized_h=resized_h,
        resized_w=resized_w,
        pad_top=pad_top,
        pad_left=pad_left,
        target_h=target_h,
        target_w=target_w,
        crop_top=crop_top,
        crop_left=crop_left,
        crop_h=crop_h,
        crop_w=crop_w,
    )


def apply_geometry(
    image: Image.Image, plan: GeometryPlan, interpolation: int = Image.BILINEAR
) -> torch.Tensor:
    resized = image.resize((plan.resized_w, plan.resized_h), interpolation)
    tensor = to_tensor(resized)
    canvas = torch.zeros(tensor.shape[0], plan.target_h, plan.target_w)
    h = min(plan.resized_h, plan.target_h - plan.pad_top)
    w = min(plan.resized_w, plan.target_w - plan.pad_left)
    canvas[:, plan.pad_top : plan.pad_top + h, plan.pad_left : plan.pad_left + w] = tensor[
        :, :h, :w
    ]
    return canvas[
        :,
        plan.crop_top : plan.crop_top + plan.crop_h,
        plan.crop_left : plan.crop_left + plan.crop_w,
    ]


def apply_geometry_label(
    label: np.ndarray, plan: GeometryPlan, ignore_index: int = 255
) -> torch.Tensor:
    image = Image.fromarray(label.astype(np.uint8))
    resized = np.asarray(image.resize((plan.resized_w, plan.resized_h), Image.NEAREST))
    canvas = np.full((plan.target_h, plan.target_w), ignore_index, dtype=np.uint8)
    h = min(plan.resized_h, plan.target_h - plan.pad_top)
    w = min(plan.resized_w, plan.target_w - plan.pad_left)
    canvas[plan.pad_top : plan.pad_top + h, plan.pad_left : plan.pad_left + w] = resized[
        :h, :w
    ]
    cropped = canvas[
        plan.crop_top : plan.crop_top + plan.crop_h,
        plan.crop_left : plan.crop_left + plan.crop_w,
    ]
    return torch.from_numpy(cropped.astype(np.int64))


def apply_geometry_disparity(disparity: np.ndarray, plan: GeometryPlan) -> torch.Tensor:
    """Resize disparity and scale its values by the same factor.
    """
    array = disparity.astype(np.float32)
    tensor = torch.from_numpy(array)[None, None]
    resized = F.interpolate(
        tensor, size=(plan.resized_h, plan.resized_w), mode="nearest"
    )[0, 0]
    resized = resized * plan.scale
    canvas = torch.zeros(plan.target_h, plan.target_w)
    h = min(plan.resized_h, plan.target_h - plan.pad_top)
    w = min(plan.resized_w, plan.target_w - plan.pad_left)
    canvas[plan.pad_top : plan.pad_top + h, plan.pad_left : plan.pad_left + w] = resized[
        :h, :w
    ]
    return canvas[
        plan.crop_top : plan.crop_top + plan.crop_h,
        plan.crop_left : plan.crop_left + plan.crop_w,
    ]


class PhotometricJitter:
    """Brightness, contrast, and saturation jitter with a declared key.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, enabled: bool = True):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.enabled = enabled

    def __call__(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        if not self.enabled:
            return tensor
        rng = random.Random(key)
        out = tensor
        factor = 1.0 + rng.uniform(-self.brightness, self.brightness)
        out = (out * factor).clamp(0.0, 1.0)
        factor = 1.0 + rng.uniform(-self.contrast, self.contrast)
        mean = out.mean(dim=(1, 2), keepdim=True)
        out = ((out - mean) * factor + mean).clamp(0.0, 1.0)
        factor = 1.0 + rng.uniform(-self.saturation, self.saturation)
        gray = out.mean(dim=0, keepdim=True)
        out = ((out - gray) * factor + gray).clamp(0.0, 1.0)
        return out


# ---------------------------------------------------------------------------
# Rig geometry
# ---------------------------------------------------------------------------

_RIG_FOCAL_TIMES_BASELINE = {"kitti": 0.54 * 721.5, "cityscapes": 0.2093 * 2262.5}
_RIG_NATIVE_WIDTH = {"kitti": 1242, "cityscapes": 2048}
DISPARITY_REFERENCE_RIG = "kitti"


def rig_disparity_scale(dataset: str, out_w: int) -> float:
    """Factor putting one rig's disparity on the reference rig's scale.
    """
    if dataset not in _RIG_FOCAL_TIMES_BASELINE:
        return 1.0
    reference = _RIG_FOCAL_TIMES_BASELINE[DISPARITY_REFERENCE_RIG] * (
        out_w / _RIG_NATIVE_WIDTH[DISPARITY_REFERENCE_RIG]
    )
    this = _RIG_FOCAL_TIMES_BASELINE[dataset] * (out_w / _RIG_NATIVE_WIDTH[dataset])
    return float(reference / this)


# ---------------------------------------------------------------------------
# Cityscapes label mapping
# ---------------------------------------------------------------------------

CITYSCAPES_LABEL_MAP = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1,
    9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5,
    18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: 255,
}

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def convert_cityscapes_label(label: np.ndarray) -> np.ndarray:
    """Map raw identifiers to the 19 training classes, rest to ignore."""
    out = np.full(label.shape, 255, dtype=np.uint8)
    for raw, train in CITYSCAPES_LABEL_MAP.items():
        if train != 255:
            out[label == raw] = train
    return out


# ---------------------------------------------------------------------------
# Sample scaffolding
# ---------------------------------------------------------------------------


def _blank_sample(dataset: str, crop_h: int, crop_w: int) -> Dict[str, object]:
    """The fields every sample carries, so collation never guesses."""
    return {
        "dataset": dataset,
        "left": torch.zeros(3, crop_h, crop_w),
        "right": torch.zeros(3, crop_h, crop_w),
        "prev_left": None,
        "disparity": torch.zeros(crop_h, crop_w),
        "disparity_valid": torch.zeros(crop_h, crop_w, dtype=torch.bool),
        "semantic": torch.full((crop_h, crop_w), 255, dtype=torch.long),
        "has_disparity": torch.tensor(False),
        "has_semantic": torch.tensor(False),
        "has_temporal": torch.tensor(False),
        "is_rectified": torch.tensor(False),
        "is_stereo": torch.tensor(False),
        "rig_disparity_scale": torch.tensor(1.0),
        "crash_label": torch.tensor(-1, dtype=torch.long),
        "crash_supervised": torch.tensor(False),
        "source_id": "",
        "recording_id": "",
        "clip_id": "",
        "frame_index": torch.tensor(-1, dtype=torch.long),
        "onset_frame": torch.tensor(-1, dtype=torch.long),
        "frames_to_accident": torch.tensor(-1, dtype=torch.long),
        "previous_frame_source": "",
        "photometric_policy": "",
    }


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------


class KITTIStereoDataset(Dataset):
    """Rectified stereo pairs with sparse disparity from the 200 annotated
    training scenes.
    """

    def __init__(self, root: Path, config: DataConfig, split: str = "train"):
        if split not in ("train", "val"):
            raise ValueError(f"KITTI split must be train or val, got {split!r}.")
        self.root = Path(root)
        self.config = config
        self.split = split
        self.manifest = load_split_manifest(config.splits_dir, "kitti_splits.json")
        self.jitter = PhotometricJitter(enabled=split == "train")
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(
                f"KITTI {split} resolved to zero samples under {self.root}. "
                f"An empty dataset is a data failure."
            )
        logger.info("KITTI %s: %d pairs", split, len(self.samples))

    def _base(self) -> Path:
        candidate = self.root / "data_scene_flow" / "training"
        return candidate if candidate.is_dir() else self.root / "training"

    def _load_samples(self) -> List[Dict]:
        base = self._base()
        left_dir, right_dir = base / "image_2", base / "image_3"
        disp_dir = base / "disp_occ_0"
        if not disp_dir.is_dir():
            disp_dir = base / "disp_noc_0"
        allowed = set(self.manifest["splits"][self.split])
        samples = []
        for name in sorted(allowed):
            left = left_dir / f"{name}.png"
            right = right_dir / f"{name}.png"
            disparity = disp_dir / f"{name}.png"
            if not (left.is_file() and right.is_file()):
                raise FileNotFoundError(
                    f"KITTI manifest lists {name} but {left} or {right} is "
                    f"absent. Regenerate the manifest against this copy of "
                    f"the dataset."
                )
            samples.append(
                {
                    "id": name,
                    "left": left,
                    "right": right,
                    "disparity": disparity if disparity.is_file() else None,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.samples[index]
        cfg = self.config
        left_image = open_image_with_retry(record["left"]).convert("RGB")
        right_image = open_image_with_retry(record["right"]).convert("RGB")
        native_w, native_h = left_image.size

        rng = (
            random.Random(f"{record['id']}|{cfg.random_seed}|{index}")
            if self.split == "train" and cfg.use_geometric_augmentation
            else None
        )
        plan = plan_geometry(
            native_h,
            native_w,
            cfg.image_height,
            cfg.image_width,
            cfg.crop_height,
            cfg.crop_width,
            rng,
        )

        sample = _blank_sample("kitti", plan.crop_h, plan.crop_w)
        key = f"kitti|{record['id']}|{cfg.random_seed}"
        sample["left"] = normalize_imagenet(
            self.jitter(apply_geometry(left_image, plan), key)
        )
        sample["right"] = normalize_imagenet(
            self.jitter(apply_geometry(right_image, plan), key)
        )
        sample["is_rectified"] = torch.tensor(True)
        sample["is_stereo"] = torch.tensor(True)
        sample["rig_disparity_scale"] = torch.tensor(
            rig_disparity_scale("kitti", cfg.image_width)
        )
        sample["source_id"] = record["id"]
        sample["recording_id"] = record["id"]
        sample["clip_id"] = record["id"]
        sample["photometric_policy"] = cfg.temporal_jitter_policy

        if record["disparity"] is not None:
            raw = np.asarray(open_image_with_retry(record["disparity"]), dtype=np.float32) / 256.0
            disparity = apply_geometry_disparity(raw, plan)
            valid = (disparity > 0) & (disparity < cfg.max_disparity)
            if valid.sum() == 0:
                raise RuntimeError(
                    f"KITTI sample {record['id']} has no valid disparity "
                    f"pixels after the geometric transform."
                )
            sample["disparity"] = disparity
            sample["disparity_valid"] = valid
            sample["has_disparity"] = torch.tensor(True)
        return sample


# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------


class CityscapesStereoDataset(Dataset):
    """Rectified stereo pairs with 19-class semantic labels.
    """

    def __init__(self, root: Path, config: DataConfig, split: str = "train"):
        if split not in ("train", "val"):
            raise ValueError(f"Cityscapes split must be train or val, got {split!r}.")
        self.root = Path(root)
        self.config = config
        self.split = split
        self.manifest = load_split_manifest(config.splits_dir, "cityscapes_splits.json")
        self.jitter = PhotometricJitter(enabled=split == "train")
        self.samples = self._load_samples()
        if not self.samples:
            raise RuntimeError(
                f"Cityscapes {split} resolved to zero samples under "
                f"{self.root}. An empty dataset is a data failure."
            )
        logger.info("Cityscapes %s: %d pairs", split, len(self.samples))

    def _load_samples(self) -> List[Dict]:
        left_root = self.root / "leftImg8bit" / self.split
        right_root = self.root / "rightImg8bit" / self.split
        label_root = self.root / "gtFine" / self.split
        if not left_root.is_dir():
            raise FileNotFoundError(
                f"{left_root} is absent; Cityscapes must present "
                f"leftImg8bit/<split>."
            )
        cities = set(self.manifest["splits"][self.split])
        samples = []
        for city in sorted(cities):
            for left in sorted((left_root / city).glob("*_leftImg8bit.png")):
                stem = left.name.replace("_leftImg8bit.png", "")
                right = right_root / city / f"{stem}_rightImg8bit.png"
                label = label_root / city / f"{stem}_gtFine_labelIds.png"
                samples.append(
                    {
                        "id": stem,
                        "city": city,
                        "left": left,
                        "right": right if right.is_file() else None,
                        "label": label if label.is_file() else None,
                    }
                )

        # The manifest records what each city contributed when the
        # partition was built. A city that enumerates a different count
        # now is a changed data exposure, and it leaves the manifest hash
        # untouched, so nothing downstream would show it.
        expected = (self.manifest["meta"].get("image_counts") or {}).get(self.split)
        if expected:
            found: Dict[str, int] = {}
            for sample in samples:
                found[sample["city"]] = found.get(sample["city"], 0) + 1
            wrong = {
                city: (count, found.get(city, 0))
                for city, count in expected.items()
                if count != found.get(city, 0)
            }
            if wrong:
                shown = dict(sorted(wrong.items())[:5])
                raise FileNotFoundError(
                    f"Cityscapes {self.split}: {len(wrong)} of "
                    f"{len(expected)} cities enumerate a different image "
                    f"count than the manifest recorded. Showing city: "
                    f"(manifest, found) for {shown}. A run on a changed "
                    f"exposure records the same manifest hash as a run on "
                    f"the original."
                )

        unlabelled = [s["id"] for s in samples if s["label"] is None]
        if unlabelled:
            raise FileNotFoundError(
                f"Cityscapes {self.split}: {len(unlabelled)} images have no "
                f"semantic label on disk, first {unlabelled[:5]}. The "
                f"failure belongs at construction, not part way through an "
                f"epoch."
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.samples[index]
        cfg = self.config
        left_image = open_image_with_retry(record["left"]).convert("RGB")
        native_w, native_h = left_image.size
        rng = (
            random.Random(f"{record['id']}|{cfg.random_seed}|{index}")
            if self.split == "train" and cfg.use_geometric_augmentation
            else None
        )
        plan = plan_geometry(
            native_h,
            native_w,
            cfg.image_height,
            cfg.image_width,
            cfg.crop_height,
            cfg.crop_width,
            rng,
        )
        sample = _blank_sample("cityscapes", plan.crop_h, plan.crop_w)
        key = f"cityscapes|{record['id']}|{cfg.random_seed}"
        left_tensor = normalize_imagenet(self.jitter(apply_geometry(left_image, plan), key))
        sample["left"] = left_tensor
        if record["right"] is not None:
            right_image = open_image_with_retry(record["right"]).convert("RGB")
            sample["right"] = normalize_imagenet(
                self.jitter(apply_geometry(right_image, plan), key)
            )
            sample["is_rectified"] = torch.tensor(True)
            sample["is_stereo"] = torch.tensor(True)
        else:
            sample["right"] = left_tensor.clone()
        sample["rig_disparity_scale"] = torch.tensor(
            rig_disparity_scale("cityscapes", cfg.image_width)
        )
        sample["source_id"] = record["city"]
        sample["recording_id"] = record["id"]
        sample["clip_id"] = record["id"]
        sample["photometric_policy"] = cfg.temporal_jitter_policy

        if record["label"] is None:
            raise FileNotFoundError(
                f"Cityscapes sample {record['id']} has no gtFine label. "
                f"Semantic supervision is required before segmentation is "
                f"reported."
            )
        raw = np.asarray(open_image_with_retry(record["label"]))
        converted = convert_cityscapes_label(raw)
        semantic = apply_geometry_label(converted, plan)
        if int((semantic != 255).sum()) == 0:
            raise RuntimeError(
                f"Cityscapes sample {record['id']} has no labelled pixels "
                f"after the geometric transform."
            )
        sample["semantic"] = semantic
        sample["has_semantic"] = torch.tensor(True)
        return sample


# ---------------------------------------------------------------------------
# CCD
# ---------------------------------------------------------------------------


class CCDDataset(Dataset):
    """Dashcam clips with per-frame accident onset, sampled temporally.
       """

    def __init__(self, root: Path, config: DataConfig, split: str = "train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"CCD split must be train/val/test, got {split!r}.")
        if cv2 is None:
            raise ImportError(
                "OpenCV is required to decode CCD clips. Install opencv-python."
            )
        self.root = Path(root)
        self.config = config
        self.split = split
        self.crash_dir = self._find_crash_dir()
        self.normal_dir = self.root / "Normal"
        self.manifest = load_split_manifest(config.splits_dir, "ccd_splits.json")
        if self.manifest["meta"]["split_unit"] != "source_recording":
            raise SchemaError(
                f"CCD manifest splits by "
                f"{self.manifest['meta']['split_unit']!r}; this study "
                f"requires source_recording so no recording spans two "
                f"partitions."
            )
        self.annotations = self._load_annotations()
        self.jitter = PhotometricJitter(enabled=split == "train")
        self._epoch = 0
        self.samples: List[Dict] = []
        self.build_samples()
        if not self.samples:
            raise RuntimeError(
                f"CCD {split} resolved to zero samples under {self.root}. "
                f"An empty dataset is a data failure."
            )
        logger.info("CCD %s: %d frame samples", split, len(self.samples))

    # -- construction ---------------------------------------------------

    def _find_crash_dir(self) -> Path:
        for name in ("Crash-1500", "Crash"):
            candidate = self.root / name
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Neither Crash-1500 nor Crash exists under {self.root}."
        )

    def _load_annotations(self) -> Dict[str, Dict]:
        """Parse per-frame accident labels.
        """
        candidates = [f"{self.crash_dir.name}.txt", "Crash-1500.txt", "Crash.txt"]
        path = None
        for name in candidates:
            for base in (self.root, self.root.parent, self.crash_dir):
                if (base / name).is_file():
                    path = base / name
                    break
            if path:
                break
        if path is None:
            raise FileNotFoundError(
                f"CCD annotation file not found (tried {candidates} under "
                f"{self.root}). Per-frame labels define accident onset and "
                f"are required."
            )

        annotations: Dict[str, Dict] = {}
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                clip = line.split(",", 1)[0].strip()
                start, end = line.find("["), line.rfind("]")
                if start == -1 or end == -1 or end < start:
                    raise ValueError(f"{path}:{lineno} has no bracketed label list.")
                tokens = [t.strip() for t in line[start + 1 : end].split(",") if t.strip()]
                if len(tokens) != 50:
                    raise ValueError(
                        f"{path}:{lineno} has {len(tokens)} per-frame labels; "
                        f"fifty are required."
                    )
                tail = [t.strip() for t in line[end + 1 :].split(",") if t.strip()]
                if len(tail) < 2:
                    raise ValueError(
                        f"{path}:{lineno} carries no source-video field. The "
                        f"fourth field indexes ytb_list.txt and defines which "
                        f"recording a clip was cut from; partitions depend on it."
                    )
                annotations[clip] = {
                    "binlabels": [int(t) for t in tokens],
                    "ytb_index": tail[1],
                }
        return annotations

    def set_epoch(self, epoch: int) -> None:
        if epoch == self._epoch:
            return
        self._epoch = epoch
        if self.split == "train" and self.config.ccd_frame_jitter:
            self.build_samples(quiet=True)

    def build_samples(self, quiet: bool = False) -> None:
        level = logger.level
        if quiet:
            logger.setLevel(logging.WARNING)
        try:
            self.samples = self._create_samples()
            if self.split == "train" and self.config.ccd_balance_mode == "oversample":
                self._balance_classes()
        finally:
            logger.setLevel(level)

    def _clip_paths(self) -> List[Tuple[Path, bool]]:
        allowed = set(self.manifest["splits"][self.split])
        found = []
        for directory, is_crash in ((self.crash_dir, True), (self.normal_dir, False)):
            if not directory.is_dir():
                continue
            for video in sorted(directory.glob("*.mp4")):
                identifier = f"{directory.name}/{video.stem}"
                if identifier in allowed:
                    found.append((video, is_crash))
        present = {f"{video.parent.name}/{video.stem}" for video, _ in found}
        absent = sorted(allowed - present)
        if absent:
            raise FileNotFoundError(
                f"CCD {self.split}: {len(absent)} manifest recordings are "
                f"absent from disk, first {absent[:5]}. A run that trains "
                f"on part of its manifest records the same comparison key "
                f"as one that trains on all of it, so the difference would "
                f"not appear in any downstream check."
            )
        return found

    def _create_samples(self) -> List[Dict]:
        cfg = self.config
        samples: List[Dict] = []
        for video, is_crash in self._clip_paths():
            samples.extend(self._clip_samples(video, is_crash))

        positives = sum(1 for s in samples if s["crash_label"] == 1)
        logger.info(
            "CCD %s: %d frame samples, %d positive (%.1f%%) at horizon %d "
            "frames (%.1f s); frames at or after onset excluded",
            self.split,
            len(samples),
            positives,
            100.0 * positives / max(len(samples), 1),
            cfg.ccd_anticipation_horizon,
            cfg.ccd_anticipation_horizon / cfg.ccd_fps,
        )

        cap = (
            cfg.ccd_max_train_samples
            if self.split == "train"
            else cfg.ccd_max_eval_samples
        )
        if cap is not None and len(samples) > cap:
            rng = random.Random(cfg.random_seed)
            samples = rng.sample(samples, cap)
            logger.info(
                "CCD %s: uniform subsample to %d, natural balance preserved",
                self.split,
                len(samples),
            )
        return samples

    def _clip_samples(self, video: Path, is_crash: bool) -> List[Dict]:
        cfg = self.config
        stem = video.stem
        recording_id = f"{video.parent.name}/{stem}"
        if is_crash:
            entry = self.annotations.get(stem)
            if entry is None or not entry.get("ytb_index"):
                raise KeyError(
                    f"Crash clip {stem} has no source-video index. Clips cut "
                    f"from one recording share a camera and a scene, so the "
                    f"recording is the partition and bootstrap unit."
                )
            source_id = f"ytb/{entry['ytb_index']}"
        else:
            source_id = recording_id
        base = list(range(0, 50, cfg.ccd_frame_stride))[: cfg.ccd_samples_per_clip]
        if self.split == "train" and cfg.ccd_frame_jitter:
            rng = random.Random(f"{stem}|{self._epoch}|{cfg.random_seed}")
            shift = rng.randrange(0, max(cfg.ccd_frame_stride, 1))
            indices = [min(i + shift, 49) for i in base]
        else:
            indices = base

        if is_crash:
            annotation = self.annotations.get(stem)
            if annotation is None:
                raise KeyError(
                    f"Crash clip {stem} has no annotation entry; accident "
                    f"onset is undefined and the label cannot be formed."
                )
            onset = next(
                (k for k, v in enumerate(annotation["binlabels"]) if v == 1), None
            )
            if onset is None:
                return []
        else:
            onset = None

        out = []
        for frame_index in indices:
            if onset is not None:
                if frame_index >= onset:
                    continue
                to_accident = onset - frame_index
                label = 1 if to_accident <= cfg.ccd_anticipation_horizon else 0
            else:
                to_accident = -1
                label = 0
            out.append(
                {
                    "video": video,
                    "frame_index": int(frame_index),
                    "crash_label": int(label),
                    "frames_to_accident": int(to_accident),
                    "onset_frame": -1 if onset is None else int(onset),
                    "source_id": source_id,
                    "recording_id": recording_id,
                    "clip_id": f"{recording_id}#{frame_index}",
                    "clip_is_crash": bool(is_crash),
                }
            )
        return out

    def _balance_classes(self) -> None:
        """Duplicate the minority class to a one-to-one stream.
               """
        positives = [s for s in self.samples if s["crash_label"] == 1]
        negatives = [s for s in self.samples if s["crash_label"] == 0]
        if not positives or not negatives:
            return
        if len(positives) < len(negatives):
            repeat = len(negatives) // len(positives) + 1
            positives = (positives * repeat)[: len(negatives)]
        else:
            repeat = len(positives) // len(negatives) + 1
            negatives = (negatives * repeat)[: len(positives)]
        self.samples = positives + negatives
        random.Random(self.config.random_seed).shuffle(self.samples)

    # -- temporal pairing -----------------------------------------------

    def _donor_index(self, index: int) -> Optional[int]:
        """A sample from a different source recording, chosen by index.
        """
        n = len(self.samples)
        source = self.samples[index]["source_id"]
        for step in range(1, n):
            candidate = (index + n // 2 + step) % n
            if self.samples[candidate]["source_id"] != source:
                return candidate
        return None

    def _previous_plan(self, index: int, sample: Dict) -> Optional[Tuple[Path, int, str]]:
        """The clip and frame the earlier image is decoded from.
        """
        gap = self.config.ccd_previous_frame_spacing
        own = sample["frame_index"] - gap
        if own < 0:
            return None
        if self.config.ccd_previous_frame_source == "clip":
            return sample["video"], own, sample["source_id"]
        donor_index = self._donor_index(index)
        if donor_index is None:
            return None
        donor = self.samples[donor_index]
        donor_frame = donor["frame_index"] - gap
        if donor_frame < 0:
            return None
        return donor["video"], donor_frame, donor["source_id"]

    def _read_frame(self, video: Path, frame_index: int) -> np.ndarray:
        """Decode one frame, retrying a transient failure of the volume."""
        for attempt in range(1, IO_RETRIES + 1):
            capture = cv2.VideoCapture(str(video))
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                capture.release()
                if ok:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                capture.release()
            if attempt < IO_RETRIES:
                delay = io_retry_delay(attempt)
                logger.warning(
                    "Decode of frame %d from %s failed on attempt %d of %d; "
                    "retrying in %.1f s.",
                    frame_index,
                    video,
                    attempt,
                    IO_RETRIES,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Frame {frame_index} of {video} could not be decoded after "
            f"{IO_RETRIES} attempts."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.samples[index]
        cfg = self.config

        frame = Image.fromarray(self._read_frame(record["video"], record["frame_index"]))
        native_w, native_h = frame.size
        rng = (
            random.Random(f"{record['clip_id']}|{self._epoch}|{cfg.random_seed}")
            if self.split == "train" and cfg.use_geometric_augmentation
            else None
        )
        plan = plan_geometry(
            native_h,
            native_w,
            cfg.image_height,
            cfg.image_width,
            cfg.crop_height,
            cfg.crop_width,
            rng,
        )

        sample = _blank_sample("ccd", plan.crop_h, plan.crop_w)
        current_key = f"ccd|{record['clip_id']}|{self._epoch}|{cfg.random_seed}"
        current = self.jitter(apply_geometry(frame, plan), current_key)
        sample["left"] = normalize_imagenet(current)
        sample["right"] = sample["left"].clone()
        sample["crash_label"] = torch.tensor(record["crash_label"], dtype=torch.long)
        sample["crash_supervised"] = torch.tensor(True)
        sample["source_id"] = record["source_id"]
        sample["recording_id"] = record["recording_id"]
        sample["clip_id"] = record["clip_id"]
        sample["frame_index"] = torch.tensor(record["frame_index"], dtype=torch.long)
        sample["onset_frame"] = torch.tensor(record["onset_frame"], dtype=torch.long)
        sample["frames_to_accident"] = torch.tensor(
            record["frames_to_accident"], dtype=torch.long
        )
        sample["previous_frame_source"] = cfg.ccd_previous_frame_source
        sample["photometric_policy"] = cfg.temporal_jitter_policy

        plan_previous = self._previous_plan(index, record)
        if plan_previous is not None:
            video, frame_index, _ = plan_previous
            previous = Image.fromarray(self._read_frame(video, frame_index))
            # Identical plan, so both frames cover the same pixel region.
            previous_tensor = apply_geometry(previous, plan)
            key = (
                current_key
                if cfg.temporal_jitter_policy == "shared"
                else f"{current_key}|previous"
            )
            sample["prev_left"] = normalize_imagenet(self.jitter(previous_tensor, key))
            sample["has_temporal"] = torch.tensor(True)
        return sample


# ---------------------------------------------------------------------------
# Sampling and collation
# ---------------------------------------------------------------------------


class BalancedMultiDatasetSampler(torch.utils.data.Sampler):
    """Draws from several datasets under fixed weights.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        names: Sequence[str],
        weights: Dict[str, float],
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        if len(datasets) != len(names):
            raise ValueError("Each dataset must be named for weighting.")
        self.datasets = list(datasets)
        self.names = list(names)
        self.weights = dict(weights)
        self.seed = int(seed)
        self.epoch = 0
        lengths = [len(d) for d in self.datasets]
        if any(length == 0 for length in lengths):
            empty = [n for n, length in zip(self.names, lengths) if length == 0]
            raise RuntimeError(
                f"Datasets {empty} are empty. Sampling from an empty dataset "
                f"is refused."
            )
        self.offsets: List[int] = []
        running = 0
        for length in lengths:
            self.offsets.append(running)
            running += length
        self.total = running
        self.num_samples = int(num_samples or running)
        self.sample_weights = self._build_weights(lengths)

    def _build_weights(self, lengths: Sequence[int]) -> np.ndarray:
        weights = np.zeros(self.total, dtype=np.float64)
        for name, offset, length in zip(self.names, self.offsets, lengths):
            share = float(self.weights.get(name, 1.0))
            weights[offset : offset + length] = share / length
        total = weights.sum()
        if total <= 0:
            raise ValueError("All dataset weights are zero.")
        return weights / total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        picks = rng.choice(self.total, size=self.num_samples, p=self.sample_weights)
        return iter(int(i) for i in picks)

    def __len__(self) -> int:
        return self.num_samples


_TENSOR_KEYS = (
    "left",
    "right",
    "disparity",
    "disparity_valid",
    "semantic",
    "has_disparity",
    "has_semantic",
    "has_temporal",
    "is_rectified",
    "is_stereo",
    "rig_disparity_scale",
    "crash_label",
    "crash_supervised",
    "frame_index",
    "onset_frame",
    "frames_to_accident",
)

_STRING_KEYS = (
    "dataset",
    "source_id",
    "recording_id",
    "clip_id",
    "previous_frame_source",
    "photometric_policy",
)


def collate_heterogeneous(batch: List[Dict]) -> Dict[str, object]:
    """Batch samples from different datasets without inventing labels.
    """
    if not batch:
        raise ValueError("Empty batch; an empty loader is a data failure.")

    heights = {tuple(item["left"].shape[-2:]) for item in batch}
    if len(heights) != 1:
        raise ValueError(
            f"Batch mixes image sizes {sorted(heights)}. Every sample must "
            f"pass through the same crop for a batch to be well formed."
        )

    out: Dict[str, object] = {}
    for key in _TENSOR_KEYS:
        out[key] = torch.stack([torch.as_tensor(item[key]) for item in batch])
    for key in _STRING_KEYS:
        out[key] = [item[key] for item in batch]

    reference = batch[0]["left"]
    previous = []
    for item in batch:
        value = item.get("prev_left")
        previous.append(value if value is not None else torch.zeros_like(reference))
    out["prev_left"] = torch.stack(previous)

    # Sensor and depth fields belong to nuScenes samples only. Where a
    # sample does not carry them they are zero-filled and the accompanying
    # mask is false, so a loss or a projection can tell an absent sensor
    # from a sensor that read zero.
    crop = reference.shape[-2:]
    radar_shapes = {
        tuple(item["radar_points"].shape)
        for item in batch
        if torch.is_tensor(item.get("radar_points"))
    }
    if len(radar_shapes) > 1:
        raise ValueError(
            f"Batch mixes radar tensor shapes {sorted(radar_shapes)}; the "
            f"point capacity is a fixed configuration value."
        )
    radar_shape = radar_shapes.pop() if radar_shapes else None

    defaults = {
        "intrinsics": lambda: torch.zeros(3, 3),
        "extrinsics": lambda: torch.zeros(4, 4),
        "depth_target": lambda: torch.zeros(*crop),
        "depth_valid": lambda: torch.zeros(*crop, dtype=torch.bool),
        "has_depth": lambda: torch.tensor(False),
        "radar_points": (
            (lambda: torch.zeros(*radar_shape)) if radar_shape else None
        ),
        "radar_num_points": lambda: torch.tensor(0, dtype=torch.long),
    }
    present = {
        key
        for key in defaults
        if any(torch.is_tensor(item.get(key)) for item in batch)
    }
    for key in present:
        factory = defaults[key]
        if factory is None:
            continue
        out[key] = torch.stack(
            [
                torch.as_tensor(item[key])
                if torch.is_tensor(item.get(key))
                else factory()
                for item in batch
            ]
        )
    if "has_depth" in present or "depth_target" in present:
        out.setdefault("has_depth", torch.stack([torch.tensor(False)] * len(batch)))
    out["scene"] = [item.get("scene", item.get("source_id", "")) for item in batch]

    # Crash labels the loss may score. An unlabelled sample keeps its
    # sentinel; substituting zero would enter the loss as a negative.
    supervised = out["crash_supervised"].bool()
    labels = out["crash_label"].clone()
    labels[~supervised] = -1
    out["crash_label"] = labels
    out["batch_size"] = len(batch)
    return out


def make_loader(
    dataset: Dataset,
    config: DataConfig,
    shuffle: bool,
    sampler: Optional[torch.utils.data.Sampler] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        drop_last=config.drop_last,
        collate_fn=collate_heterogeneous,
        worker_init_fn=worker_init_fn,
        generator=loader_generator(config.random_seed),
    )


def create_visual_dataloaders(
    config: DataConfig,
    data_stream: Sequence[str] = ("kitti", "ccd", "cityscapes"),
) -> Tuple[DataLoader, DataLoader, Dict[str, object]]:
    """Build the shared training and validation loaders.
    """
    roots = assert_datasets_available([n for n in data_stream if n != "nuscenes"])
    train_sets, val_sets, names = [], [], []

    if "kitti" in data_stream:
        train_sets.append(KITTIStereoDataset(roots["kitti"], config, "train"))
        val_sets.append(KITTIStereoDataset(roots["kitti"], config, "val"))
        names.append("kitti")
    if "ccd" in data_stream:
        train_sets.append(CCDDataset(roots["ccd"], config, "train"))
        val_sets.append(CCDDataset(roots["ccd"], config, "val"))
        names.append("ccd")
    if "cityscapes" in data_stream:
        train_sets.append(CityscapesStereoDataset(roots["cityscapes"], config, "train"))
        val_sets.append(CityscapesStereoDataset(roots["cityscapes"], config, "val"))
        names.append("cityscapes")

    if not train_sets:
        raise ValueError(f"data_stream {list(data_stream)} selected no dataset.")

    sampler = BalancedMultiDatasetSampler(
        train_sets,
        names,
        config.dataset_weights,
        num_samples=config.samples_per_epoch,
        seed=config.random_seed,
    )
    train_loader = make_loader(
        ConcatDataset(train_sets), config, shuffle=False, sampler=sampler
    )
    val_loader = make_loader(ConcatDataset(val_sets), config, shuffle=False)

    summary = {
        "data_stream": list(names),
        "train_sizes": {n: len(d) for n, d in zip(names, train_sets)},
        "val_sizes": {n: len(d) for n, d in zip(names, val_sets)},
        "samples_per_epoch": len(sampler),
        "dataset_weights": dict(config.dataset_weights),
        "camera_transform": config.camera_transform(),
    }
    logger.info("Data stream: %s", summary["train_sizes"])
    return train_loader, val_loader, summary


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def _contract_tests() -> int:
    """Batch fields, masks, epoch propagation, and transform alignment."""
    from reproducibility import set_epoch_recursively

    failures = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        if condition:
            print(f"  pass  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name} {detail}")

    print("unified_stereo_dataloaders contract tests")

    plan = plan_geometry(900, 1600, 376, 1240, 320, 720, random.Random(0))
    check(
        "resize preserves aspect ratio to within a pixel of rounding",
        abs(plan.resized_w / plan.resized_h - 1600 / 900) < 1e-2,
        f"{plan.resized_w}x{plan.resized_h}",
    )
    check("one scale for both axes", plan.scale == min(376 / 900, 1240 / 1600))

    image = Image.fromarray((np.random.rand(900, 1600, 3) * 255).astype(np.uint8))
    a = apply_geometry(image, plan)
    b = apply_geometry(image, plan)
    check("the same plan gives the same pixels", torch.equal(a, b))
    check("crop size is honoured", tuple(a.shape[-2:]) == (320, 720))

    jitter = PhotometricJitter(enabled=True)
    shared_a = jitter(a, "key")
    shared_b = jitter(b, "key")
    check("shared jitter is identical across a pair", torch.equal(shared_a, shared_b))
    check(
        "asymmetric jitter differs",
        not torch.equal(jitter(a, "key"), jitter(a, "key|previous")),
    )

    blank = _blank_sample("ccd", 8, 8)
    labelled = dict(blank)
    labelled["crash_label"] = torch.tensor(1, dtype=torch.long)
    labelled["crash_supervised"] = torch.tensor(True)
    batch = collate_heterogeneous([dict(blank), labelled])
    check("unlabelled crash stays at the sentinel", int(batch["crash_label"][0]) == -1)
    check("labelled crash survives collation", int(batch["crash_label"][1]) == 1)
    check(
        "supervision mask marks only the labelled sample",
        batch["crash_supervised"].tolist() == [False, True],
    )
    check(
        "absent previous frame becomes a masked zero tensor",
        bool((batch["prev_left"][0] == 0).all()) and not bool(batch["has_temporal"][0]),
    )
    for key in ("source_id", "recording_id", "clip_id", "frame_index",
                "onset_frame", "frames_to_accident"):
        check(f"batch carries {key}", key in batch)

    class _Recorder:
        def __init__(self):
            self.epoch = -1

        def set_epoch(self, epoch):
            self.epoch = epoch

    class _Wrapper:
        def __init__(self, child):
            self.dataset = child

    recorder = _Recorder()
    reached = set_epoch_recursively(_Wrapper(recorder), 7)
    check("epoch propagates through a wrapper", reached == 1 and recorder.epoch == 7)

    scale = rig_disparity_scale("cityscapes", 1240)
    check("rig scaling is defined and finite", 0.0 < scale < 100.0, f"{scale:.4f}")
    check("reference rig scales to one", rig_disparity_scale("kitti", 1240) == 1.0)

    try:
        collate_heterogeneous([])
        check("empty batch raises", False, "no error")
    except ValueError:
        check("empty batch raises", True)

    try:
        DataConfig(ccd_previous_frame_spacing="1")
        check("a non-integer frame gap raises", False, "no error")
    except TypeError:
        check("a non-integer frame gap raises", True)

    print(f"\n{'all contract tests passed' if failures == 0 else f'{failures} failed'}")
    return failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(_contract_tests())
