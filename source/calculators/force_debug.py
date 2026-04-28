"""Force debugging — toggle `enabled = True` before running."""

import logging
import numpy as np

enabled = False
_logger = logging.getLogger("ForceDebug")


def log_forces(label: str, forces: np.ndarray, atomic_numbers: np.ndarray):
    if not enabled:
        return
    norms = np.linalg.norm(forces, axis=1)
    h_mask = atomic_numbers == 1

    _logger.info(f"========== {label} Force Debug ==========")
    _logger.info(f"  total norm: {norms.sum():.4f}")
    _logger.info(f"  mean: {norms.mean():.4f}  max: {norms.max():.4f}  min: {norms.min():.4f}")
    _logger.info(f"  |x| mean: {np.abs(forces[:,0]).mean():.4f}  "
                 f"|y|: {np.abs(forces[:,1]).mean():.4f}  "
                 f"|z|: {np.abs(forces[:,2]).mean():.4f}")
    if h_mask.any():
        _logger.info(f"  H  — n={h_mask.sum()} mean={norms[h_mask].mean():.4f} max={norms[h_mask].max():.4f}")
    if (~h_mask).any():
        _logger.info(f"  non-H — n={(~h_mask).sum()} mean={norms[~h_mask].mean():.4f} max={norms[~h_mask].max():.4f}")
    _logger.info(f"==========================================")
