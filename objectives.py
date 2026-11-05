#!/usr/bin/env python3
"""The loss terms, defined once for every trainer.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

OBJECTIVES = (
    "supervised_disparity",
    "photometric_stereo",
    "smoothness",
    "semantic",
    "temporal_crash",
    "metric_depth",
)


@dataclass
class LossWeights:
    """Relative weight of each term. Recorded in the run configuration."""

    supervised_disparity: float = 1.0
    photometric_stereo: float = 0.1
    smoothness: float = 0.05
    semantic: float = 0.5
    temporal_crash: float = 1.0
    metric_depth: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in OBJECTIVES}


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Structural similarity over 3x3 windows, returned as a dissimilarity."""
    pool = nn.AvgPool2d(3, 1, padding=1, count_include_pad=False)
    mu_x, mu_y = pool(x), pool(y)
    sigma_x = pool(x * x) - mu_x**2
    sigma_y = pool(y * y) - mu_y**2
    sigma_xy = pool(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return ((1 - numerator / denominator) / 2).clamp(0, 1)


def warp_right_to_left(right: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
    """Sample the right image at the disparity-shifted coordinates."""
    b, _, h, w = right.shape
    ys = torch.linspace(-1, 1, h, device=right.device, dtype=right.dtype)
    xs = torch.linspace(-1, 1, w, device=right.device, dtype=right.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.unsqueeze(0).expand(b, -1, -1)
    gy = gy.unsqueeze(0).expand(b, -1, -1)
    shift = 2.0 * disparity.squeeze(1) / max(w - 1, 1)
    grid = torch.stack([gx - shift, gy], dim=-1)
    return F.grid_sample(
        right, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def supervised_disparity_loss(
    predictions: Sequence[torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    has_disparity: torch.Tensor,
    max_disparity: int = 192,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Geometric-weighted absolute error over the refinement sequence.

    """
    device = target.device
    sample_mask = has_disparity.to(device).bool()
    if not bool(sample_mask.any()):
        return torch.zeros((), device=device)

    pixel_mask = valid.to(device).bool() & (target < max_disparity)
    pixel_mask = pixel_mask & sample_mask.view(-1, 1, 1)
    if not bool(pixel_mask.any()):
        return torch.zeros((), device=device)

    n = len(predictions)
    total = torch.zeros((), device=device)
    for i, prediction in enumerate(predictions):
        weight = gamma ** (n - i - 1)
        error = (prediction.squeeze(1) - target).abs()
        total = total + weight * error[pixel_mask].mean()
    return total


def photometric_stereo_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    predictions: Sequence[torch.Tensor],
    is_rectified: torch.Tensor,
    disparity_scale: torch.Tensor,
    gamma: float = 0.9,
) -> torch.Tensor:
    """Appearance consistency, applied only to physically rectified pairs.
    """
    device = left.device
    mask = is_rectified.to(device).bool()
    if not bool(mask.any()):
        return torch.zeros((), device=device)

    left_sel = left[mask]
    right_sel = right[mask]
    scale = disparity_scale.to(device)[mask].view(-1, 1, 1, 1)

    n = len(predictions)
    total = torch.zeros((), device=device)
    for i, prediction in enumerate(predictions):
        weight = gamma ** (n - i - 1)
        disparity = prediction[mask] * scale
        warped = warp_right_to_left(right_sel, disparity)
        photometric = 0.85 * ssim(warped, left_sel).mean(1, keepdim=True) + 0.15 * (
            warped - left_sel
        ).abs().mean(1, keepdim=True)
        total = total + weight * photometric.mean()
    return total


def smoothness_loss(
    disparity: torch.Tensor, image: torch.Tensor, is_rectified: torch.Tensor
) -> torch.Tensor:
    """Edge-aware first-order smoothness on rectified samples."""
    device = disparity.device
    mask = is_rectified.to(device).bool()
    if not bool(mask.any()):
        return torch.zeros((), device=device)
    d = disparity[mask]
    im = image[mask]
    # Normalise by the mean absolute disparity, with an additive floor.
    #
    # A signed mean can approach zero through cancellation, because the
    # refinement starts at zero disparity and its early deltas take both
    # signs. Dividing by that mean then produces a loss several orders of
    # magnitude larger than every other term, which overflows fp16, causes
    # the gradient scaler to skip the step, and makes the reported loss
    # meaningless for those batches. The absolute value removes the
    # cancellation and the floor bounds the ratio.
    mean = d.abs().mean(dim=(2, 3), keepdim=True) + 1e-2
    d = d / mean
    grad_dx = (d[:, :, :, :-1] - d[:, :, :, 1:]).abs()
    grad_dy = (d[:, :, :-1, :] - d[:, :, 1:, :]).abs()
    grad_ix = (im[:, :, :, :-1] - im[:, :, :, 1:]).abs().mean(1, keepdim=True)
    grad_iy = (im[:, :, :-1, :] - im[:, :, 1:, :]).abs().mean(1, keepdim=True)
    return (grad_dx * torch.exp(-grad_ix)).mean() + (
        grad_dy * torch.exp(-grad_iy)
    ).mean()


def semantic_loss(
    logits: Optional[torch.Tensor],
    target: torch.Tensor,
    has_semantic: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Cross-entropy over labelled pixels of labelled samples."""
    if logits is None:
        return torch.zeros((), device=target.device)
    device = logits.device
    mask = has_semantic.to(device).bool()
    if not bool(mask.any()):
        return torch.zeros((), device=device)
    selected_logits = logits[mask]
    selected_target = target.to(device)[mask]
    if int((selected_target != ignore_index).sum()) == 0:
        return torch.zeros((), device=device)
    return F.cross_entropy(
        selected_logits,
        selected_target,
        weight=class_weights.to(device) if class_weights is not None else None,
        ignore_index=ignore_index,
    )


def temporal_crash_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    supervised: torch.Tensor,
    class_weights: Sequence[float] = (1.0, 2.0),
    scope: str = "labelled",
) -> torch.Tensor:
    """Class-weighted cross-entropy over genuinely labelled samples.
    """
    device = logits.device
    mask = supervised.to(device).bool()
    if scope == "all":
        mask = mask | (labels.to(device) >= 0)
    elif scope != "labelled":
        raise ValueError(f"scope must be 'labelled' or 'all', got {scope!r}.")
    if not bool(mask.any()):
        return torch.zeros((), device=device)

    selected_labels = labels.to(device)[mask]
    if int((selected_labels < 0).sum()) > 0:
        raise ValueError(
            "An unlabelled sample reached the crash loss. Unavailable labels "
            "must be masked, never zero-filled and read as negatives."
        )
    weights = torch.tensor(list(class_weights), device=device, dtype=logits.dtype)
    return F.cross_entropy(logits[mask], selected_labels, weight=weights)


def metric_depth_loss(
    prediction: Optional[torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    has_depth: torch.Tensor,
) -> torch.Tensor:
    """Masked absolute error in metres against projected LiDAR."""
    if prediction is None:
        return torch.zeros((), device=target.device)
    device = prediction.device
    sample_mask = has_depth.to(device).bool()
    if not bool(sample_mask.any()):
        return torch.zeros((), device=device)
    pixel_mask = valid.to(device).bool() & sample_mask.view(-1, 1, 1)
    if not bool(pixel_mask.any()):
        return torch.zeros((), device=device)
    error = (prediction.squeeze(1) - target.to(device)).abs()
    return error[pixel_mask].mean()


class ObjectiveSet(nn.Module):
    """Computes exactly the declared objectives and nothing else.
    """

    def __init__(
        self,
        active: Sequence[str],
        weights: Optional[LossWeights] = None,
        max_disparity: int = 192,
        crash_class_weights: Sequence[float] = (1.0, 2.0),
        crash_scope: str = "labelled",
        semantic_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        unknown = set(active) - set(OBJECTIVES)
        if unknown:
            raise ValueError(f"Unknown objectives {sorted(unknown)}.")
        if not active:
            raise ValueError("At least one objective must be active.")
        self.active = tuple(sorted(active))
        self.weights = weights or LossWeights()
        self.max_disparity = max_disparity
        self.crash_class_weights = tuple(crash_class_weights)
        self.crash_scope = crash_scope
        self.semantic_class_weights = semantic_class_weights

    def forward(self, output: Dict[str, object], batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        device = output["disparity"].device
        terms: Dict[str, torch.Tensor] = {}

        if "supervised_disparity" in self.active:
            terms["supervised_disparity"] = supervised_disparity_loss(
                output["disparity_predictions"],
                batch["disparity"].to(device),
                batch["disparity_valid"].to(device),
                batch["has_disparity"],
                max_disparity=self.max_disparity,
            )
        if "photometric_stereo" in self.active:
            terms["photometric_stereo"] = photometric_stereo_loss(
                batch["left_intensity"].to(device),
                batch["right_intensity"].to(device),
                output["disparity_predictions"],
                batch["is_rectified"],
                batch["rig_disparity_scale"],
            )
        if "smoothness" in self.active:
            terms["smoothness"] = smoothness_loss(
                output["disparity"],
                batch["left_intensity"].to(device),
                batch["is_rectified"],
            )
        if "semantic" in self.active:
            terms["semantic"] = semantic_loss(
                output.get("semantic_logits"),
                batch["semantic"],
                batch["has_semantic"],
                class_weights=self.semantic_class_weights,
            )
        if "temporal_crash" in self.active:
            terms["temporal_crash"] = temporal_crash_loss(
                output["crash_logits"],
                batch["crash_label"],
                batch["crash_supervised"],
                class_weights=self.crash_class_weights,
                scope=self.crash_scope,
            )
        if "metric_depth" in self.active:
            fused = output.get("depth_fused")
            camera = output.get("depth_camera")
            target = batch["depth_target"]
            valid = batch["depth_valid"]
            has_depth = batch["has_depth"]
            fused_term = metric_depth_loss(fused, target, valid, has_depth)
            camera_term = metric_depth_loss(camera, target, valid, has_depth)
            # Both paths are supervised, so the camera-only prediction is
            # in distribution at evaluation rather than a co-adapted head
            # run on inputs it never saw.
            terms["metric_depth"] = fused_term + camera_term
            terms["metric_depth_fused"] = fused_term.detach()
            terms["metric_depth_camera"] = camera_term.detach()

        total = torch.zeros((), device=device)
        for name in self.active:
            total = total + float(getattr(self.weights, name)) * terms[name]

        # Every objective returns a fresh zero when its batch carries no
        # valid samples, and a fresh zero has no grad_fn. With one active
        # objective such a batch severs the graph and backward raises.
        # Anchoring the total to a differentiable output leaves the value
        # unchanged and lets the batch contribute a gradient of zero.
        if not total.requires_grad:
            anchor = None
            for key in (
                "crash_logits",
                "semantic_logits",
                "disparity",
                "depth_camera",
                "depth_fused",
                "disparity_predictions",
            ):
                value = output.get(key)
                candidates = value if isinstance(value, (list, tuple)) else [value]
                for item in candidates:
                    if torch.is_tensor(item) and item.requires_grad:
                        anchor = item
                        break
                if anchor is not None:
                    break
            if anchor is not None:
                total = total + anchor.sum() * 0.0

        terms["total"] = total
        return terms

    def to_dict(self) -> Dict[str, object]:
        return {
            "active_objectives": list(self.active),
            "loss_weights": self.weights.to_dict(),
            "crash_class_weights": list(self.crash_class_weights),
            "crash_scope": self.crash_scope,
            "max_disparity": self.max_disparity,
        }


__all__ = [
    "OBJECTIVES",
    "LossWeights",
    "ObjectiveSet",
    "ssim",
    "warp_right_to_left",
    "supervised_disparity_loss",
    "photometric_stereo_loss",
    "smoothness_loss",
    "semantic_loss",
    "temporal_crash_loss",
    "metric_depth_loss",
]
