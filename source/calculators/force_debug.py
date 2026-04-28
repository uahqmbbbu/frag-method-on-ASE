"""Force debugging — toggle `enabled = True` before running."""

import logging
import numpy as np

enabled = False
_step = 0
_logger = logging.getLogger("ForceDebug")


def step_header():
    global _step
    if not enabled:
        return
    _logger.info(f"--- Step {_step} ---")
    _step += 1


def log_forces(label: str, forces: np.ndarray):
    if not enabled:
        return
    norms = np.linalg.norm(forces, axis=1)
    _logger.info(f"  {label:12s}  max={norms.max():12.4f}  min={norms.min():12.4f}")
