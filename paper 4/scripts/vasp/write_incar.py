#!/usr/bin/env python3
"""Write INCAR files from project templates with per-system substitution.

Templates live in ``templates/`` and use ``${SYSTEM}`` / ``${MAGMOM}`` as
placeholders (Python ``string.Template`` syntax).  This function fills them in
from a system identifier and a pymatgen Structure carrying ``"magmom"`` site
properties (written by ``build_slab.py``).

Always use this function instead of pymatgen's ``Incar.write_file()`` to ensure
ENCUT=500 consistency required by the LFT pipeline.

Templates:
    templates/INCAR_opt    Full ionic relaxation (NSW=300, IBRION=2)
    templates/INCAR_sp     Single-point (NSW=0, IBRION=-1)  ← LFT labeling
    templates/INCAR_freq   Frequency (IBRION=5) ← VASP validation only

Example:
    from pymatgen.core import Structure
    from scripts.vasp.write_incar import write_incar

    struct = Structure.from_file("data/structures/slabs/NiZn_NC/structure.json")
    write_incar(mode="opt", output_dir="work/Phase01_Slab/NiZn_NC/opt/",
                system_id="NiZn_NC", structure=struct)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from string import Template

from pymatgen.core import Structure

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"
_MODE_MAP: dict[str, str] = {
    "opt":      "INCAR_opt",       # Adsorption structure optimization (ISYM=-1)
    "opt_slab": "INCAR_opt_slab",  # Bare slab optimization (ISYM=0)
    "sp":       "INCAR_sp",
    "freq":     "INCAR_freq",
    "freq_ads": "INCAR_freq_ads",  # Frequency calc for adsorption structures
}


def _compress_magmom(magmoms: list[float]) -> str:
    """Run-length encode a magmom list into the VASP INCAR format.

    Args:
        magmoms: Per-atom magnetic moments in POSCAR species order.

    Returns:
        Compressed string, e.g. ``"62*0 6*0 1*4 1*3"``.
    """
    result: list[str] = []
    i = 0
    while i < len(magmoms):
        val = magmoms[i]
        count = 1
        while i + count < len(magmoms) and magmoms[i + count] == val:
            count += 1
        val_str = str(int(val)) if val == int(val) else f"{val:.1f}"
        result.append(f"{count}*{val_str}")
        i += count
    return " ".join(result)


def magmom_from_structure(structure: Structure) -> str:
    """Build the INCAR MAGMOM string from a Structure's ``"magmom"`` site property.

    Atoms are grouped by species in first-appearance order, mirroring the
    species block order that ``write_poscar`` produces.  Sites without a
    ``"magmom"`` property are assigned 0.

    Args:
        structure: Pymatgen Structure with optional ``"magmom"`` site property.

    Returns:
        Compressed MAGMOM string for INCAR (e.g. ``"62*0 6*0 1*4 1*3"``).
    """
    # Preserve first-appearance species order (same as Poscar writer)
    species_order: list[str] = list(dict.fromkeys(
        site.specie.symbol for site in structure
    ))
    grouped: dict[str, list[float]] = defaultdict(list)
    for site in structure:
        grouped[site.specie.symbol].append(
            float(site.properties.get("magmom", 0.0))
        )

    all_magmoms: list[float] = []
    for sym in species_order:
        all_magmoms.extend(grouped[sym])

    return _compress_magmom(all_magmoms)


def write_incar(
    mode: str,
    output_dir: str | Path,
    system_id: str | None = None,
    structure: Structure | None = None,
) -> None:
    """Render an INCAR template and write it to ``output_dir/INCAR``.

    Args:
        mode:       Calculation mode: ``"opt"``, ``"sp"``, or ``"freq"``.
        output_dir: Target directory. Created if it does not exist.
        system_id:  Replaces the ``${SYSTEM}`` placeholder (e.g. ``"NiZn_NC"``).
                    Defaults to the output directory name.
        structure:  Pymatgen Structure with ``"magmom"`` site property.
                    Replaces the ``${MAGMOM}`` placeholder.  When omitted,
                    the placeholder is left as-is in the output file and a
                    warning is logged.

    Raises:
        ValueError:       If ``mode`` is not one of the known modes.
        FileNotFoundError: If the template file does not exist.
    """
    if mode not in _MODE_MAP:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_MODE_MAP)}")

    template_path = _TEMPLATE_DIR / _MODE_MAP[mode]
    if not template_path.exists():
        raise FileNotFoundError(f"INCAR template not found: {template_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resolved_system = system_id or out.name
    subs: dict[str, str] = {"SYSTEM": resolved_system}

    if structure is not None:
        subs["MAGMOM"] = magmom_from_structure(structure)
    else:
        logger.warning(
            "No structure provided for %s; MAGMOM placeholder left unresolved.", out
        )

    content = Template(template_path.read_text(encoding="utf-8")).safe_substitute(subs)
    dest = out / "INCAR"
    dest.write_text(content, encoding="utf-8")
    logger.info("Wrote INCAR (mode=%s, system=%s) → %s", mode, resolved_system, dest)
