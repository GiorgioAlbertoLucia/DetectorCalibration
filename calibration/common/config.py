"""
Config loading and validation.

Usage
-----
    from calibration.common.config import load_config
    cfg = load_config('configs/LHC23_pass4_ITS.yaml')
    # cfg is a plain dict; access as cfg['its']['Pr']['x_min_fit'] etc.
"""

from __future__ import annotations
import yaml
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """
    Load a YAML config file and return it as a plain dict.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    KeyError
        If mandatory top-level sections are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')

    with path.open() as f:
        cfg = yaml.safe_load(f)

    _validate(cfg, path)
    return cfg


# ── Validation helpers ────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = {'dataset', 'output'}

def _validate(cfg: dict, path: Path) -> None:
    missing = _REQUIRED_SECTIONS - cfg.keys()
    if missing:
        raise KeyError(
            f'Config {path} is missing required sections: {missing}'
        )
    _validate_dataset(cfg['dataset'], path)


def _validate_dataset(ds: dict, path: Path) -> None:
    for key in ('label',):
        if key not in ds:
            raise KeyError(f'Config {path}: dataset.{key} is required')