# Artifact paths, run identities, and checkpoint metadata.
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
EXPERIMENT_SCHEMA = 'unified_safety_2026_v1'
STUDY_IDENTITY = {'title': 'Image-based stereo, semantic, and crash classification', 'supported_claims': [], 'claims_not_supported': [], 'sensor_roles': {'camera': 'visual input'}}
SCHEMA_FIELD = 'experiment_schema'
PROJECT_ROOT = Path(os.environ.get('UNIFIED_SAFETY_ROOT', Path(__file__).resolve().parent.parent))
RUNS_ROOT = PROJECT_ROOT / 'runs' / EXPERIMENT_SCHEMA
SPLITS_ROOT = PROJECT_ROOT / 'splits' / EXPERIMENT_SCHEMA
EVAL_ROOT = PROJECT_ROOT / 'eval' / EXPERIMENT_SCHEMA
CANONICAL_SPLIT_MANIFESTS = ('ccd_splits.json', 'kitti_splits.json', 'nuscenes_splits.json', 'cityscapes_splits.json')

class SchemaError(RuntimeError):
    pass

def stamp(payload: Dict) -> Dict:
    out = dict(payload)
    out[SCHEMA_FIELD] = EXPERIMENT_SCHEMA
    return out

def require_schema(payload: Dict, source: str) -> Dict:
    if not isinstance(payload, dict):
        raise SchemaError(f'{source} is a {type(payload).__name__}, not a schema-stamped mapping.')
    found = payload.get(SCHEMA_FIELD)
    if found is None:
        raise SchemaError(f'{source} carries no {SCHEMA_FIELD} field. Artifacts without a schema stamp are not part of {EXPERIMENT_SCHEMA} and cannot enter it.')
    if found != EXPERIMENT_SCHEMA:
        raise SchemaError(f'{source} is stamped {found!r}; this study is {EXPERIMENT_SCHEMA!r}. Regenerate the artifact under the current schema.')
    return payload

def write_json(path: Path, payload: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp(payload), indent=2, sort_keys=True, default=_json_default), encoding='utf-8')
    return path

def read_json(path: Path) -> Dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Artifact not found: {path}')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    return require_schema(payload, str(path))

def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    raise TypeError(f'{type(obj).__name__} is not JSON serialisable')

def assert_output_dir_clean(directory: Path) -> None:
    directory = Path(directory)
    if not directory.exists():
        return
    for entry in sorted(directory.glob('*.json')):
        try:
            payload = json.loads(entry.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        found = payload.get(SCHEMA_FIELD)
        if found != EXPERIMENT_SCHEMA:
            raise SchemaError(f'{directory} already holds {entry.name}, stamped {found!r}. Measurements for {EXPERIMENT_SCHEMA} must be written to a directory that contains no other schema. Choose a new run identity or remove the directory.')

@dataclass(frozen=True)
class RunIdentity:
    experiment: str
    variant: str
    seed: int

    @property
    def name(self) -> str:
        return f'{self.experiment}__{self.variant}__seed{self.seed}'

    @property
    def directory(self) -> Path:
        return RUNS_ROOT / self.name

    def eval_directory(self, evaluator: str) -> Path:
        return EVAL_ROOT / evaluator / self.name

    def to_dict(self) -> Dict[str, object]:
        return {'experiment': self.experiment, 'variant': self.variant, 'seed': int(self.seed), 'run_name': self.name, 'run_directory': str(self.directory)}

def run_identity(experiment: str, variant: str, seed: int) -> RunIdentity:
    for label, value in (('experiment', experiment), ('variant', variant)):
        if not value or any((c in value for c in ' /\\\'"')):
            raise ValueError(f'{label} must be a non-empty token without whitespace, path separators, or quotes; got {value!r}.')
    return RunIdentity(experiment=experiment, variant=variant, seed=int(seed))

def resolve_run(name: str) -> Path:
    directory = RUNS_ROOT / name
    if not directory.is_dir():
        raise FileNotFoundError(f'No run named {name!r} under {RUNS_ROOT}. Runs are addressed by their deterministic identity; check the experiment, variant, and seed.')
    return directory

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def dict_sha256(payload: Dict) -> str:
    return text_sha256(json.dumps(payload, sort_keys=True, default=_json_default))

def source_files() -> Sequence[Path]:
    here = Path(__file__).resolve().parent
    return tuple(sorted((p for p in here.rglob('*.py') if '__pycache__' not in p.parts)))

def code_hash() -> str:
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(path.name.encode('utf-8'))
        digest.update(file_sha256(path).encode('utf-8'))
    return digest.hexdigest()

def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None

def environment_record() -> Dict[str, object]:
    record: Dict[str, object] = {'python': sys.version.split()[0], 'platform': platform.platform(), 'code_hash': code_hash(), 'git_commit': git_commit()}
    for module in ('torch', 'torchvision', 'numpy', 'cv2', 'scipy'):
        try:
            imported = __import__(module)
            record[module] = getattr(imported, '__version__', 'unknown')
        except ImportError:
            record[module] = None
    try:
        import torch
        record['cuda_available'] = bool(torch.cuda.is_available())
        record['cuda_version'] = torch.version.cuda
        if torch.cuda.is_available():
            record['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        record['cuda_available'] = False
    return record

def manifest_hashes(splits_dir: Path, names: Iterable[str]=CANONICAL_SPLIT_MANIFESTS) -> Dict[str, str]:
    splits_dir = Path(splits_dir)
    hashes: Dict[str, str] = {}
    for name in names:
        path = splits_dir / name
        hashes[name] = file_sha256(path) if path.is_file() else 'absent'
    return hashes
