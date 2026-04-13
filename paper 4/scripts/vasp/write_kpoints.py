#!/usr/bin/env python3
"""Write Gamma-centered KPOINTS files.

Generates a Gamma-centered Monkhorst-Pack mesh. For 4×4 graphene slabs
the standard mesh is 3×3×1 (kspacing ≈ 0.04 Å⁻¹).

Example:
    from scripts.vasp.write_kpoints import write_kpoints
    write_kpoints(mesh=(3, 3, 1), output_dir="work/Phase01_Slab/NiZn_NC/opt/")
"""
from __future__ import annotations

import logging
from pathlib import Path

from pymatgen.io.vasp.inputs import Kpoints

logger = logging.getLogger(__name__)


def write_kpoints(
    mesh: tuple[int, int, int],
    output_dir: str | Path,
) -> None:
    """Write a Gamma-centered KPOINTS file.

    Args:
        mesh:       (kx, ky, kz) grid divisions.
        output_dir: Target directory. Created if it does not exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "KPOINTS"

    kpoints = Kpoints.gamma_automatic(kpts=mesh, shift=(0, 0, 0))
    kpoints.write_file(str(dest))
    logger.info("Wrote KPOINTS %s → %s", mesh, dest)
