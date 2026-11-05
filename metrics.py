#!/usr/bin/env python3
"""Metric definitions and clustered uncertainty for the whole study.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, by the rank formulation, ties averaged.
    """
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    sorted_scores = s[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision, summed over recall increments at distinct scores.
    """
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp_cum = np.cumsum(ys == 1)
    fp_cum = np.cumsum(ys == 0)
    ends = np.append(np.flatnonzero(np.diff(ss)), ss.size - 1)
    tp, fp = tp_cum[ends], fp_cum[ends]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    previous = np.concatenate([[0.0], recall[:-1]])
    return float((precision * (recall - previous)).sum())


def ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    """AP and AUROC together with the prevalence AP must be read against."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    n = int(y.size)
    prevalence = float(y.mean()) if n else float("nan")
    out: Dict[str, float] = {"prevalence": prevalence, "n": n}
    if n == 0 or y.min() == y.max():
        out.update(
            {
                "auroc": float("nan"),
                "average_precision": float("nan"),
                "ap_over_chance": float("nan"),
                "available": False,
            }
        )
        return out
    ap = average_precision(y, s)
    out.update(
        {
            "auroc": float(auroc(y, s)),
            "average_precision": float(ap),
            "ap_over_chance": float(ap) - prevalence,
            "available": True,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------


def confusion_counts(labels: Sequence[int], predictions: Sequence[int]) -> Dict[str, int]:
    y = np.asarray(labels).astype(int)
    p = np.asarray(predictions).astype(int)
    if y.size != p.size:
        raise ValueError(f"{y.size} labels against {p.size} predictions.")
    return {
        "tp": int(((y == 1) & (p == 1)).sum()),
        "tn": int(((y == 0) & (p == 0)).sum()),
        "fp": int(((y == 0) & (p == 1)).sum()),
        "fn": int(((y == 1) & (p == 0)).sum()),
    }


def metrics_from_counts(counts: Dict[str, int]) -> Dict[str, float]:
    tp, tn = counts["tp"], counts["tn"]
    fp, fn = counts["fp"], counts["fn"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n": int(total),
        **{k: int(v) for k, v in counts.items()},
    }


def select_threshold(
    labels: Sequence[int], scores: Sequence[float], n_grid: int = 200
) -> Dict[str, float]:
    """Choose a decision threshold on validation scores.

    """
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or y.min() == y.max():
        return {"threshold": 0.5, "f1": float("nan"), "available": False}
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return {"threshold": float(lo), "f1": float("nan"), "available": False}
    grid = np.linspace(lo, hi, int(n_grid))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in grid:
        stats = metrics_from_counts(confusion_counts(y, (s >= threshold).astype(int)))
        f1 = stats["f1"]
        if not np.isnan(f1) and f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
    return {"threshold": best_threshold, "f1": float(best_f1), "available": True}


# ---------------------------------------------------------------------------
# Dense-task metrics
# ---------------------------------------------------------------------------


def disparity_metrics(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> Dict[str, float]:
    """End-point error and threshold error rates over valid pixels."""
    mask = np.asarray(valid).astype(bool)
    if mask.sum() == 0:
        raise ValueError(
            "No valid disparity pixels. An empty metric mask is a data "
            "failure and is raised rather than reported as zero."
        )
    error = np.abs(np.asarray(prediction)[mask] - np.asarray(target)[mask])
    return {
        "epe": float(error.mean()),
        "d1": float((error > 3.0).mean() * 100.0),
        "bad_1px": float((error > 1.0).mean() * 100.0),
        "n_pixels": int(mask.sum()),
    }


def depth_metrics(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> Dict[str, float]:
    """Metric-depth errors in metres over pixels holding a LiDAR return."""
    mask = np.asarray(valid).astype(bool)
    if mask.sum() == 0:
        raise ValueError(
            "No valid depth pixels. An empty metric mask is a data failure "
            "and is raised rather than reported as zero."
        )
    p = np.asarray(prediction, dtype=float)[mask]
    t = np.asarray(target, dtype=float)[mask]
    difference = p - t
    ratio = np.maximum(p / np.maximum(t, 1e-6), t / np.maximum(p, 1e-6))
    return {
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt((difference**2).mean())),
        "abs_rel": float((np.abs(difference) / np.maximum(t, 1e-6)).mean()),
        "delta_1": float((ratio < 1.25).mean()),
        "n_pixels": int(mask.sum()),
    }


class SemanticConfusion:
    """Accumulates a confusion matrix for mIoU over a whole split."""

    def __init__(self, num_classes: int = 19, ignore_index: int = 255):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        p = np.asarray(prediction).reshape(-1)
        t = np.asarray(target).reshape(-1)
        keep = t != self.ignore_index
        if keep.sum() == 0:
            return
        index = t[keep].astype(np.int64) * self.num_classes + p[keep].astype(np.int64)
        counts = np.bincount(index, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        if self.matrix.sum() == 0:
            raise ValueError(
                "No labelled semantic pixels were accumulated. Reporting "
                "segmentation without supervision is refused."
            )
        intersection = np.diag(self.matrix).astype(float)
        union = (
            self.matrix.sum(axis=1) + self.matrix.sum(axis=0) - np.diag(self.matrix)
        ).astype(float)
        present = union > 0
        iou = np.full(self.num_classes, np.nan, dtype=float)
        iou[present] = intersection[present] / union[present]
        return {
            "miou": float(np.nanmean(iou)),
            "pixel_accuracy": float(intersection.sum() / self.matrix.sum()),
            "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
            "classes_present": int(present.sum()),
        }


# ---------------------------------------------------------------------------
# Clustered bootstrap
# ---------------------------------------------------------------------------


def _cluster_index(clusters: Sequence[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    names = sorted(set(clusters))
    array = np.asarray(clusters)
    return names, {name: np.flatnonzero(array == name) for name in names}


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile interval for a mean, resampling whole clusters.
    """
    v = np.asarray(values, dtype=float)
    names, index = _cluster_index(clusters)
    if len(names) == 0:
        raise ValueError("No clusters supplied; an interval cannot be formed.")
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=float)
    name_array = np.asarray(names)
    for b in range(int(n_boot)):
        picked = rng.choice(name_array, size=len(names), replace=True)
        rows = np.concatenate([index[name] for name in picked])
        draws[b] = float(v[rows].mean())
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(v.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_clusters": len(names),
        "n_rows": int(v.size),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def paired_cluster_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Interval for the mean of (a - b), resampling clusters once per draw.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"Paired comparison needs aligned arrays; got {a.shape} and "
            f"{b.shape}."
        )
    difference = a - b
    result = cluster_bootstrap(difference, clusters, n_boot=n_boot, seed=seed, alpha=alpha)
    result["mean_a"] = float(a.mean())
    result["mean_b"] = float(b.mean())
    return result


def cluster_bootstrap_ranking(
    labels: Sequence[int],
    scores: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Intervals for AP and AUROC, resampling whole clusters."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    names, index = _cluster_index(clusters)
    if len(names) == 0:
        raise ValueError("No clusters supplied; an interval cannot be formed.")
    rng = np.random.default_rng(seed)
    name_array = np.asarray(names)
    ap_draws: List[float] = []
    auc_draws: List[float] = []
    for _ in range(int(n_boot)):
        picked = rng.choice(name_array, size=len(names), replace=True)
        rows = np.concatenate([index[name] for name in picked])
        yb, sb = y[rows], s[rows]
        if yb.min() == yb.max():
            continue
        ap_draws.append(average_precision(yb, sb))
        auc_draws.append(auroc(yb, sb))
    point = ranking_metrics(y, s)

    def interval(draws: List[float], value: float) -> Dict[str, float]:
        if len(draws) < 2:
            return {
                "point": float(value),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_draws": len(draws),
            }
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {
            "point": float(value),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_draws": len(draws),
        }

    return {
        "average_precision": interval(ap_draws, point["average_precision"]),
        "auroc": interval(auc_draws, point["auroc"]),
        "prevalence": point["prevalence"],
        "n_clusters": len(names),
        "n_rows": int(y.size),
    }


def paired_cluster_bootstrap_ranking(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Interval for AP(a) - AP(b) on identical labels and clusters."""
    y = np.asarray(labels).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    names, index = _cluster_index(clusters)
    rng = np.random.default_rng(seed)
    name_array = np.asarray(names)
    draws: List[float] = []
    for _ in range(int(n_boot)):
        picked = rng.choice(name_array, size=len(names), replace=True)
        rows = np.concatenate([index[name] for name in picked])
        yb = y[rows]
        if yb.min() == yb.max():
            continue
        draws.append(average_precision(yb, a[rows]) - average_precision(yb, b[rows]))
    point = average_precision(y, a) - average_precision(y, b)
    if len(draws) < 2:
        return {
            "point": float(point),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "excludes_zero": False,
            "n_clusters": len(names),
        }
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_clusters": len(names),
        "n_draws": len(draws),
    }


def seed_spread(values: Sequence[float]) -> Dict[str, float]:
    """Spread across repeated seeds, reported beside every seeded result."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"n_seeds": 0, "mean": float("nan"), "sd": float("nan"), "range": float("nan")}
    return {
        "n_seeds": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "range": float(v.max() - v.min()),
    }


__all__ = [
    "auroc",
    "average_precision",
    "ranking_metrics",
    "confusion_counts",
    "metrics_from_counts",
    "select_threshold",
    "disparity_metrics",
    "depth_metrics",
    "SemanticConfusion",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
    "cluster_bootstrap_ranking",
    "paired_cluster_bootstrap_ranking",
    "seed_spread",
]
