#!/usr/bin/env python3
"""Stage 01 — Build TM₁-TM₂-N-C slab models from a DAC template structure.

Reads a pymatgen Structure template (JSON or CIF format) that contains two TM
placeholder sites, replaces them with the requested TM₁ and TM₂ elements,
assigns initial magnetic moments as a site property, and writes the result as
both a POSCAR (via ``write_poscar``) and a pymatgen Structure JSON.

The template is expected to be a pre-built TM₁-TM₂-N-C double-vacancy
graphene supercell (4×4 or 5×5) with four N atoms coordinating the two TM
sites.  This script performs element substitution only — it does NOT construct
the double-vacancy geometry from scratch.

Spec: DEV_SPEC/phase01_slab.md §1.2–1.3

Usage:
    # Single pair
    python scripts/build/build_slab.py \\
        --tm1 Ni --tm2 Zn \\
        --template data/structures/graphene_template.json \\
        --output data/structures/slabs/NiZn_NC/

    # Batch from CSV (output of sample_initial_combinations.py)
    python scripts/build/build_slab.py \\
        --csv data/sampled_combinations.csv \\
        --template data/structures/graphene_template.json \\
        --outroot data/structures/slabs/

    python scripts/build/build_slab.py --csv data/sampled_combinations.csv --template data/structures/graphene_template.json --outroot data/structures/slabs/
    python scripts/build/build_slab.py --csv data/sampled_combinations.csv --template data/structures/graphene_template.json --outroot data/structures/slabs/ --overwrite --write-inputs



Outputs (per system):
    {output_dir}/POSCAR          VASP input structure (written by write_poscar)
    {output_dir}/structure.json  Pymatgen Structure JSON with MAGMOM site property
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is invoked directly
# (e.g. `python scripts/build/build_slab.py`).  Has no effect when the package
# is installed or when the root is already on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymatgen.core import Structure

from scripts.vasp.write_incar import write_incar
from scripts.vasp.write_kpoints import write_kpoints
from scripts.vasp.write_poscar import write_poscar

logger = logging.getLogger(__name__)

# ── TM catalogue ─────────────────────────────────────────────────────────────
# 27 transition metals in scope (Tc excluded); see DEV_SPEC/DEV_SPEC.md §1.
# C(27, 2) = 351 unordered pairs cover the full chemical space.
TM_LIST: list[str] = [
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",   # 3d
    "Y",  "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",          # 4d
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",                 # 5d
]

# (covalent_radius_Å, d_electron_count, initial_magmom_μB)
# Covalent radii are not used during slab construction.
# d-electron counts are used by the sampling script, not here.
# Initial magmoms are stored as a site property for downstream INCAR generation.
_TM_DATA: dict[str, tuple[float, int, float]] = {
    "Sc": (1.70, 1, 1.0),  "Ti": (1.60, 2, 2.0),  "V":  (1.53, 3, 3.0),
    "Cr": (1.39, 5, 6.0),  "Mn": (1.61, 5, 5.0),  "Fe": (1.32, 6, 4.0),
    "Co": (1.26, 7, 3.0),  "Ni": (1.24, 8, 2.0),  "Cu": (1.32, 10, 1.0),
    "Zn": (1.22, 10, 0.0), "Y":  (1.90, 1, 1.0),  "Zr": (1.75, 2, 2.0),
    "Nb": (1.64, 4, 3.0),  "Mo": (1.54, 5, 4.0),  "Ru": (1.44, 7, 3.0),
    "Rh": (1.42, 8, 2.0),  "Pd": (1.39, 10, 0.0), "Ag": (1.45, 10, 1.0),
    "Cd": (1.44, 10, 0.0), "Hf": (1.75, 2, 2.0),  "Ta": (1.70, 3, 3.0),
    "W":  (1.62, 4, 4.0),  "Re": (1.51, 5, 3.0),  "Os": (1.44, 6, 2.0),
    "Ir": (1.41, 7, 3.0),  "Pt": (1.36, 9, 2.0),  "Au": (1.36, 10, 1.0),
}

_TM_SET: frozenset[str] = frozenset(TM_LIST)


def _initial_magmom(symbol: str) -> float:
    """Return the initial magnetic moment (μB) for a TM element.

    Args:
        symbol: Element symbol; must be present in the TM catalogue.

    Returns:
        Initial magnetic moment as a float.

    Raises:
        KeyError: If ``symbol`` is not in the TM catalogue.
    """
    return _TM_DATA[symbol][2]


def build_slab(
    template: Structure,
    tm1: str,
    tm2: str,
    template_tm1: str | None = None,
    template_tm2: str | None = None,
) -> Structure:
    """Replace TM placeholder sites in a template with tm1 and tm2.

    The template must contain exactly two TM sites (any element from
    TM_LIST).  The first TM site encountered by index becomes TM₁; the second
    becomes TM₂.  All other sites (C, N, …) are left unchanged.

    Magnetic moments are stored as the ``"magmom"`` site property so that
    downstream INCAR writers can generate the MAGMOM tag without re-reading
    this file.

    Args:
        template:     Pymatgen Structure of the TM₁-TM₂-N-C slab template.
        tm1:          Target TM₁ element symbol (e.g. ``"Fe"``).
        tm2:          Target TM₂ element symbol (e.g. ``"Co"``).
        template_tm1: Expected TM₁ element already in the template.  When
                      provided, raises if the found element differs, guarding
                      against accidental template swaps.
        template_tm2: Expected TM₂ element already in the template (same
                      guard as ``template_tm1``).

    Returns:
        New Structure with TM sites replaced and ``"magmom"`` site property set.

    Raises:
        KeyError:   If ``tm1`` or ``tm2`` are not in the TM catalogue.
        ValueError: If the template does not contain exactly two TM sites, or
                    if the actual template TM elements differ from the expected
                    values passed in ``template_tm1`` / ``template_tm2``.
    """
    for sym in (tm1, tm2):
        if sym not in _TM_SET:
            raise KeyError(
                f"'{sym}' is not in the TM catalogue.  "
                f"Supported elements: {TM_LIST}"
            )

    # Identify TM site indices in order of appearance
    tm_indices = [
        i for i, site in enumerate(template)
        if site.specie.symbol in _TM_SET
    ]
    if len(tm_indices) != 2:
        raise ValueError(
            f"Template must contain exactly 2 TM sites; found {len(tm_indices)}: "
            f"{[template[i].specie.symbol for i in tm_indices]}"
        )

    idx1, idx2 = tm_indices
    actual_tm1 = template[idx1].specie.symbol
    actual_tm2 = template[idx2].specie.symbol

    if template_tm1 is not None and actual_tm1 != template_tm1:
        raise ValueError(
            f"Template TM₁ is '{actual_tm1}', expected '{template_tm1}'."
        )
    if template_tm2 is not None and actual_tm2 != template_tm2:
        raise ValueError(
            f"Template TM₂ is '{actual_tm2}', expected '{template_tm2}'."
        )

    # Assign per-site magmoms: 0.0 for non-TM sites, catalogue value for TM
    site_magmoms: list[float] = []
    for i, site in enumerate(template):
        if i == idx1:
            site_magmoms.append(_initial_magmom(tm1))
        elif i == idx2:
            site_magmoms.append(_initial_magmom(tm2))
        else:
            site_magmoms.append(0.0)

    new_struct = template.copy()
    new_struct.replace(idx1, tm1)
    new_struct.replace(idx2, tm2)
    new_struct.add_site_property("magmom", site_magmoms)

    logger.debug(
        "Built %s%s-N-C: TM₁@idx%d (%s→%s, magmom=%.1f), TM₂@idx%d (%s→%s, magmom=%.1f)",
        tm1, tm2,
        idx1, actual_tm1, tm1, _initial_magmom(tm1),
        idx2, actual_tm2, tm2, _initial_magmom(tm2),
    )
    return new_struct


def build_and_save(
    tm1: str,
    tm2: str,
    template: Structure,
    output_dir: Path,
    overwrite: bool = False,
    write_inputs: bool = False,
) -> bool:
    """Build one TM₁-TM₂-N-C slab and write POSCAR + structure.json.

    Args:
        tm1:          TM₁ element symbol.
        tm2:          TM₂ element symbol.
        template:     Pre-loaded template Structure.
        output_dir:   Destination directory; created if absent.
        overwrite:    When False (default), skip if POSCAR already exists.
        write_inputs: When True, also write INCAR (opt) and KPOINTS alongside POSCAR.

    Returns:
        True on success; False if skipped or on error.
    """
    name = f"{tm1}{tm2}_NC"
    poscar_path = output_dir / "POSCAR"

    if poscar_path.exists() and not overwrite:
        logger.info("[skip] %s — POSCAR already exists at %s", name, poscar_path)
        return False

    try:
        struct = build_slab(template, tm1, tm2)
    except (KeyError, ValueError) as exc:
        logger.error("[FAIL] %s — %s", name, exc)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    comment = f"{tm1}{tm2}-N-C  DAC slab  MAGMOM from _TM_DATA"
    write_poscar(struct, output_dir, comment=comment)

    json_path = output_dir / "structure.json"
    struct.to(fmt="json", filename=str(json_path))

    if write_inputs:
        write_incar(mode="opt_slab", output_dir=output_dir,
                    system_id=name, structure=struct)
        write_kpoints(mesh=(3, 3, 1), output_dir=output_dir)

    logger.info("[OK]   %s → %s", name, output_dir)
    return True


def build_from_csv(
    csv_path: Path,
    template: Structure,
    outroot: Path,
    overwrite: bool = False,
    write_inputs: bool = False,
) -> tuple[int, int]:
    """Batch-build slabs for all (TM₁, TM₂) rows in a sampled-combinations CSV.

    The CSV must contain at least the columns ``TM1`` and ``TM2``, matching the
    format written by ``sample_initial_combinations.py``.  Each system is
    written to ``{outroot}/{TM1}{TM2}_NC/``.

    Args:
        csv_path:     Path to the sampled combinations CSV.
        template:     Pre-loaded template Structure.
        outroot:      Root output directory.
        overwrite:    Passed through to :func:`build_and_save`.
        write_inputs: Passed through to :func:`build_and_save`.

    Returns:
        Tuple of ``(n_ok, n_fail_or_skipped)``.
    """
    combos: list[tuple[str, str]] = []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            combos.append((row["TM1"].strip(), row["TM2"].strip()))

    logger.info("Loaded %d combinations from %s", len(combos), csv_path)

    n_ok = n_fail = 0
    for tm1, tm2 in combos:
        success = build_and_save(
            tm1, tm2, template, outroot / f"{tm1}{tm2}_NC",
            overwrite=overwrite, write_inputs=write_inputs,
        )
        if success:
            n_ok += 1
        else:
            n_fail += 1

    logger.info("Batch complete — ok: %d  fail/skip: %d", n_ok, n_fail)
    return n_ok, n_fail


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build TM₁-TM₂-N-C slab POSCARs from a DAC template structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--template",
        default="data/structures/graphene_template.json",
        help="Path to the template structure (pymatgen JSON or CIF).",
    )
    p.add_argument(
        "--outroot",
        default="data/structures/slabs",
        metavar="DIR",
        help="Root output directory.  Each system is written to {outroot}/{TM1}{TM2}_NC/.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing POSCAR files.",
    )
    p.add_argument(
        "--write-inputs",
        action="store_true",
        help="Also write INCAR (opt) and KPOINTS alongside each POSCAR.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--csv",
        metavar="CSV_PATH",
        help="Batch mode: path to sampled_combinations.csv.",
    )
    mode.add_argument(
        "--tm1",
        metavar="ELEMENT",
        help="Single-pair mode: TM₁ element symbol.",
    )

    p.add_argument(
        "--tm2",
        metavar="ELEMENT",
        help="Single-pair mode: TM₂ element symbol (required with --tm1).",
    )
    p.add_argument(
        "--output",
        metavar="DIR",
        help=(
            "Output directory for single-pair mode.  "
            "Defaults to {outroot}/{TM1}{TM2}_NC/."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the slab builder.

    Returns:
        0 on success, 1 on error.
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    template_path = Path(args.template)
    if not template_path.exists():
        logger.error("Template not found: %s", template_path)
        return 1

    logger.info("Loading template from %s", template_path)
    template = Structure.from_file(str(template_path))

    # Single-pair mode
    if args.tm1:
        if not args.tm2:
            logger.error("--tm2 is required when --tm1 is given.")
            return 1
        for sym, label in ((args.tm1, "--tm1"), (args.tm2, "--tm2")):
            if sym not in _TM_SET:
                logger.error("%s '%s' is not in the TM catalogue.", label, sym)
                return 1
        output_dir = (
            Path(args.output) if args.output
            else Path(args.outroot) / f"{args.tm1}{args.tm2}_NC"
        )
        success = build_and_save(
            args.tm1, args.tm2, template, output_dir,
            overwrite=args.overwrite, write_inputs=args.write_inputs,
        )
        return 0 if success else 1

    # Batch CSV mode
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return 1
    n_ok, n_fail = build_from_csv(
        csv_path, template, Path(args.outroot),
        overwrite=args.overwrite, write_inputs=args.write_inputs,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
