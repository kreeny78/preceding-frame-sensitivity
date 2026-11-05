# Visual Perception Research Code

This repository contains Python implementations for CCD video-based crash classification,
KITTI stereo estimation, and Cityscapes semantic segmentation.

## Installation

Install dependencies in an isolated Python environment:

```bash
pip install -r requirements.txt
```

Run commands from the `src` directory so local modules are importable.

## Data

Data must be downloaded from their respective providers and configured using the
project's dataset root settings. Dataset files and trained checkpoints are not bundled.

- CCD: https://github.com/Cogito2012/CarCrashDataset
- KITTI: https://www.cvlibs.net/datasets/kitti/
- Cityscapes: https://www.cityscapes-dataset.com/

## Main scripts

- `make_splits.py`: generate training, validation, and test manifests
- `train_crash_only.py`: train using crash labels
- `train_unified_base.py`: train with stereo, semantic, and crash objectives
- `train_stereo_only.py`: stereo-only training
- `train_semantic_only.py`: semantic-only training
- `evaluate_research_system.py`: evaluate supported image-based tasks

Additional modules in `src` implement the visual model, data loading, training engine,
losses, performance metrics, and checkpoint metadata.

## Validation scope

This is a curated source release. The code has passed syntax and import checks; exact
reproduction requires the original dataset partitions and checkpoints. The split-generation
routine must be checked against the source-disjoint partitioning protocol before any
results are described as equivalent to archived experiments.
