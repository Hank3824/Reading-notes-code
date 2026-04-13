#!/usr/bin/env python3
"""Parse VASP OUTCAR files: total energy, forces, convergence status.

Used after every VASP single-point or optimization job to extract results
before updating the screening database.

Example:
    from scripts.vasp.parse_outcar import parse_outcar

    result = parse_outcar("work/Phase02/NiZn_NC/TM1/COOH/preopt/OUTCAR")
    if result["converged"]:
        db.update_adsorbate(system_id, adsorbate, E_ads=result["energy"], ...)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class OutcarResult(TypedDict):
    """Parsed results from a VASP OUTCAR file."""

    energy: float          # Total energy (eV), last ionic step
    forces: list[list[float]]  # Forces (eV/Å), shape (N_atoms, 3)
    converged: bool        # True if electronic SCF converged
    n_ionic_steps: int     # Number of ionic steps completed
    cpu_time: float        # Total CPU time (seconds)


def parse_outcar(outcar_path: str | Path) -> OutcarResult:
    """Parse a VASP OUTCAR file and return key results.

    Args:
        outcar_path: Path to OUTCAR file.

    Returns:
        OutcarResult with energy, forces, convergence status.

    Raises:
        FileNotFoundError: If the OUTCAR file does not exist.
        ValueError: If the OUTCAR file is incomplete or unreadable.
    """
    # TODO: implement
    raise NotImplementedError
