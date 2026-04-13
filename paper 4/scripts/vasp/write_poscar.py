#!/usr/bin/env python3
"""Write POSCAR files from pymatgen Structure objects.

Always use this function instead of ``structure.to(fmt="poscar")`` directly,
to enforce consistent formatting and selective-dynamics ordering.

Example::

    from pymatgen.core import Structure
    from scripts.vasp.write_poscar import write_poscar

    # Plain POSCAR (no selective dynamics)
    structure = Structure.from_file("CONTCAR")
    write_poscar(structure, output_dir="work/Phase01_Slab/NiZn_NC/opt/")

    # POSCAR with Selective Dynamics (e.g. for frequency calculations)
    sd_flags = [[False, False, False]] * 70 + [[True, True, True]] * 2
    write_poscar(
        structure,
        output_dir="work/Phase02_Adsorption/NiZn_NC/COOH_TM1/freq/",
        selective_dynamics=sd_flags,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.inputs import Poscar

logger = logging.getLogger(__name__)


def _sort_fixed_first(
    structure: Structure,
    selective_dynamics: list[list[bool]],
) -> tuple[Structure, list[list[bool]]]:
    """Within each species group, reorder so fixed (F F F) sites precede free (T T T).

    Pymatgen's Poscar already groups atoms by species; this helper additionally
    ensures that within each species block the constrained slab atoms appear
    before the mobile adsorbate atoms.  This preserves POTCAR compatibility
    (element line order is unchanged) while producing a human-readable POSCAR
    where slab and adsorbate atoms are visually separated.

    Args:
        structure:           Pymatgen Structure (read from CONTCAR).
        selective_dynamics:  Per-atom [bool, bool, bool] flags — True means free.

    Returns:
        A (new_structure, new_sd) tuple with sites reordered within each species.
    """
    species_order: list[str] = list(
        dict.fromkeys(site.specie.symbol for site in structure)
    )

    fixed: dict[str, list[tuple]] = {sp: [] for sp in species_order}
    free: dict[str, list[tuple]] = {sp: [] for sp in species_order}

    for site, sd in zip(structure, selective_dynamics):
        sp = site.specie.symbol
        (free if any(sd) else fixed)[sp].append((site, sd))

    sorted_sites = []
    sorted_sd: list[list[bool]] = []
    for sp in species_order:
        for site, sd in fixed[sp] + free[sp]:
            sorted_sites.append(site)
            sorted_sd.append(sd)

    new_structure = Structure.from_sites(sorted_sites)
    return new_structure, sorted_sd


def write_poscar(
    structure: Structure,
    output_dir: str | Path,
    comment: str = "",
    selective_dynamics: list[list[bool]] | None = None,
    sort_fixed_first: bool = True,
    significant_figures: int = 8,
) -> None:
    """Write a POSCAR file from a pymatgen Structure.

    When ``selective_dynamics`` is supplied, the POSCAR includes a
    ``Selective dynamics`` block and, by default, fixed slab atoms are written
    before free adsorbate atoms within each species group
    (``sort_fixed_first=True``).

    Args:
        structure:            Pymatgen Structure object.
        output_dir:           Target directory. Created if it does not exist.
        comment:              First-line comment (defaults to reduced formula).
        selective_dynamics:   Per-atom ``[bool, bool, bool]`` flags.  ``True``
                              means the atom is free to move (T T T); ``False``
                              means fixed (F F F).  Must have the same length
                              as ``structure`` when provided.
        sort_fixed_first:     If True and ``selective_dynamics`` is given,
                              reorder sites within each species so fixed atoms
                              come before free atoms.  Does not affect the
                              element-line order (POTCAR compatibility is
                              preserved).
        significant_figures:  Coordinate precision (default 8).

    Raises:
        ValueError: If ``selective_dynamics`` length does not match the
                    number of sites.
    """
    if selective_dynamics is not None and len(selective_dynamics) != len(structure):
        raise ValueError(
            f"selective_dynamics has {len(selective_dynamics)} entries but "
            f"structure has {len(structure)} sites."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "POSCAR"

    write_struct = structure
    write_sd = selective_dynamics

    if selective_dynamics is not None and sort_fixed_first:
        write_struct, write_sd = _sort_fixed_first(structure, selective_dynamics)

    poscar = Poscar(
        write_struct,
        comment=comment or structure.formula,
        selective_dynamics=write_sd,
    )
    dest.write_text(
        poscar.get_string(significant_figures=significant_figures),
        encoding="utf-8",
    )
    logger.info("Wrote POSCAR (%s) → %s", write_struct.formula, dest)
