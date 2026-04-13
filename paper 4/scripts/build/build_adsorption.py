#!/usr/bin/env python3
"""Stage 02a — Build initial adsorption structures from an optimised slab CONTCAR.

Reads the VASP-optimised bare slab, places each requested adsorbate above the
TM₁ or TM₂ site, and writes a complete set of VASP inputs for adsorption
geometry optimisation.

Slab atom convention (must match build_slab.py):
    1×TM₁  +  1×TM₂  +  62×C  +  6×N  =  70 base atoms
    Element order in POSCAR: TM₁, TM₂, C, N  (then O, H appended for adsorbates)

Selective Dynamics:
    All slab atoms → F F F (fixed during adsorption optimisation)
    All adsorbate atoms → T T T (free)

Output layout::

    {output}/{ads}_{site}/
        POSCAR    ← initial adsorption geometry, Selective Dynamics
        INCAR     ← ionic relaxation (IBRION=2)
        KPOINTS   ← Gamma 3×3×1

Usage::

    # Single system, one adsorbate
    python scripts/build/build_adsorption.py \\
        --slab data/structures/slabs/NiZn_NC/CONTCAR \\
        --tm1 Ni --tm2 Zn --ads COOH --site TM1 \\
        --output work/Phase02_Adsorption/NiZn_NC/

    # Single system, all adsorbates and both sites
    python scripts/build/build_adsorption.py \\
        --slab data/structures/slabs/NiZn_NC/CONTCAR \\
        --tm1 Ni --tm2 Zn \\
        --output work/Phase02_Adsorption/NiZn_NC/

    # Batch from CSV (columns: TM1, TM2); slab paths resolved automatically
    python scripts/build/build_adsorption.py \\
        --csv data/sampled_combinations.csv \\
        --slab-root data/structures/slabs \\
        --output-root work/Phase02_Adsorption/
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.constants import MAGMOM_CONFIG
from scripts.vasp.write_incar import write_incar
from scripts.vasp.write_kpoints import write_kpoints
from scripts.vasp.write_poscar import write_poscar

logger = logging.getLogger(__name__)

ALL_ADS: list[str] = ["CO2", "COOH", "CO", "H", "OH", "O"]
ALL_SITES: list[str] = ["TM1", "TM2"]
KPOINTS_MESH: tuple[int, int, int] = (3, 3, 1)

# ── Initial adsorbate geometries ──────────────────────────────────────────────
# Format: list of (element, dx, dy, dz) in Ångström, relative to TM site.
# dz is always positive here; vacuum_sign flips it toward the vacuum layer.
ADSORBATE_CONFIGS: dict[str, list[tuple[str, float, float, float]]] = {
    "CO2": [
        ("C",  0.00,  0.00, 1.90),  # TM-C ≈ 1.9 Å
        ("O",  1.04,  0.00, 2.50),  # bent CO₂, O-C-O ≈ 120°
        ("O", -1.04,  0.00, 2.50),
    ],
    "COOH": [
        ("C",  0.00,  0.00, 1.90),  # TM-C ≈ 1.9 Å
        ("O",  0.00,  0.00, 3.10),  # C=O (carbonyl), C-O = 1.20 Å
        ("O",  1.20,  0.00, 2.20),  # C-OH (hydroxyl), C-O = 1.35 Å
        ("H",  1.95,  0.00, 2.85),  # O-H = 0.98 Å
    ],
    "CO": [
        ("C",  0.00,  0.00, 1.90),  # TM-C ≈ 1.9 Å
        ("O",  0.00,  0.00, 3.10),  # C-O = 1.20 Å, upright
    ],
    "H": [
        ("H",  0.00,  0.00, 1.60),  # TM-H ≈ 1.6 Å
    ],
    "OH": [
        ("O",  0.00,  0.00, 1.90),  # TM-O ≈ 1.9 Å
        ("H",  0.68,  0.00, 2.57),  # O-H ≈ 0.97 Å, tilted
    ],
    "O": [
        ("O",  0.00,  0.00, 1.70),  # TM-O ≈ 1.7 Å (short, strong bond)
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_tm_index(structure: Structure, element: str) -> int:
    """Return the index of the first site matching *element*.

    Args:
        structure: Pymatgen Structure of the bare slab.
        element:   Element symbol to search for (e.g. ``"Ni"``).

    Returns:
        Zero-based site index.

    Raises:
        ValueError: If no site with that element is found.
    """
    for i, site in enumerate(structure):
        if site.specie.symbol == element:
            return i
    raise ValueError(
        f"Element '{element}' not found in structure. "
        f"Present elements: {list(dict.fromkeys(s.specie.symbol for s in structure))}"
    )


def _vacuum_sign(structure: Structure, tm_index: int) -> float:
    """Return +1 or -1 indicating which z-direction points toward the vacuum.

    Compares the z-coordinate of the TM site to the mean z of all N atoms.
    The adsorbate is placed on the side facing away from the bulk graphene.

    Args:
        structure: Bare slab Structure.
        tm_index:  Index of the TM site used for adsorption.

    Returns:
        +1.0 if vacuum is above the TM site (TM z ≥ mean N z), else -1.0.
    """
    n_z = [s.coords[2] for s in structure if s.specie.symbol == "N"]
    avg_n_z = float(np.mean(n_z)) if n_z else 0.0
    return 1.0 if structure[tm_index].coords[2] >= avg_n_z else -1.0


def _place_adsorbate(
    slab: Structure,
    tm_index: int,
    ads_type: str,
) -> tuple[Structure, list[list[bool]]]:
    """Place adsorbate atoms on *slab* at the given TM site.

    Args:
        slab:      Optimised bare slab Structure (not mutated).
        tm_index:  Index of the TM site to adsorb on.
        ads_type:  Adsorbate label from :data:`ADSORBATE_CONFIGS`.

    Returns:
        ``(new_structure, selective_dynamics)`` where ``new_structure``
        contains slab + adsorbate atoms and ``selective_dynamics`` has
        ``[F,F,F]`` for every slab atom and ``[T,T,T]`` for every
        adsorbate atom.

    Raises:
        KeyError: If ``ads_type`` is not in :data:`ADSORBATE_CONFIGS`.
    """
    config = ADSORBATE_CONFIGS[ads_type]
    tm_cart = slab[tm_index].coords.copy()
    vsign = _vacuum_sign(slab, tm_index)

    # Start with a copy so the original slab is not mutated
    new_struct = slab.copy()

    for elem, dx, dy, dz in config:
        cart = tm_cart + np.array([dx, dy, vsign * dz])
        frac = slab.lattice.get_fractional_coords(cart)
        new_struct.append(elem, frac, coords_are_cartesian=False)

    n_slab = len(slab)
    n_total = len(new_struct)
    sd_flags: list[list[bool]] = (
        [[False, False, False]] * n_slab
        + [[True, True, True]] * (n_total - n_slab)
    )

    logger.debug(
        "Placed %s on site %d (%s): added %d atom(s)",
        ads_type,
        tm_index,
        slab[tm_index].specie.symbol,
        n_total - n_slab,
    )
    return new_struct, sd_flags


def _attach_magmom(structure: Structure) -> None:
    """Attach initial magnetic moments as a ``"magmom"`` site property.

    Moments are looked up from :data:`~scripts.utils.constants.MAGMOM_CONFIG`
    by element symbol and stored in-place on *structure*.

    Args:
        structure: Pymatgen Structure (mutated in-place).
    """
    magmoms = [MAGMOM_CONFIG.get(s.specie.symbol, 0.0) for s in structure]
    structure.add_site_property("magmom", magmoms)


# ── Per-system builder ────────────────────────────────────────────────────────

def build_one(
    slab_path: Path,
    tm1: str,
    tm2: str,
    ads: str,
    site: str,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """Build one adsorption optimisation input set.

    Reads *slab_path*, places *ads* on the site specified by *site* (TM₁ or
    TM₂), and writes ``POSCAR``, ``INCAR``, and ``KPOINTS`` into
    ``{output_dir}/{ads}_{site}/``.

    Args:
        slab_path:  Path to the optimised bare slab CONTCAR (or POSCAR).
        tm1:        TM₁ element symbol (e.g. ``"Ni"``).
        tm2:        TM₂ element symbol (e.g. ``"Zn"``).
        ads:        Adsorbate label (e.g. ``"COOH"``).
        site:       ``"TM1"`` or ``"TM2"``.
        output_dir: Root output directory for this system
                    (e.g. ``work/Phase02_Adsorption/NiZn_NC/``).
        force:      Overwrite existing outputs when True.

    Returns:
        ``True`` on success, ``False`` on failure or skip.
    """
    system_id = f"{tm1}{tm2}_NC"
    label = f"{system_id}/{ads}_{site}"
    target_elem = tm1 if site == "TM1" else tm2
    dest = output_dir / f"{ads}_{site}"

    if not slab_path.exists():
        logger.warning("[%s] slab file not found: %s", label, slab_path)
        return False

    if not force and (dest / "POSCAR").exists() and (dest / "INCAR").exists():
        logger.info("[%s] already exists — skipping (use --force to overwrite)", label)
        return True

    try:
        slab = Structure.from_file(str(slab_path))

        tm_index = _find_tm_index(slab, target_elem)
        new_struct, sd_flags = _place_adsorbate(slab, tm_index, ads)
        _attach_magmom(new_struct)

        write_poscar(
            new_struct,
            output_dir=dest,
            comment=f"{system_id}-{ads}-{site}",
            selective_dynamics=sd_flags,
        )
        write_incar(
            mode="opt",
            output_dir=dest,
            system_id=f"{system_id}-{ads}-{site}",
            structure=new_struct,
        )
        write_kpoints(mesh=KPOINTS_MESH, output_dir=dest)

        logger.info("[%s] ✓ → %s", label, dest)
        return True

    except (KeyError, ValueError) as exc:
        logger.error("[%s] %s", label, exc)
        return False
    except Exception:
        logger.exception("[%s] unexpected error", label)
        return False


# ── Batch runner ──────────────────────────────────────────────────────────────

def build_batch(
    combos: list[tuple[str, str]],
    ads_list: list[str],
    site_list: list[str],
    slab_root: Path,
    output_root: Path,
    force: bool = False,
) -> tuple[int, int]:
    """Run :func:`build_one` over all (combo, adsorbate, site) combinations.

    Args:
        combos:      List of ``(TM₁, TM₂)`` pairs.
        ads_list:    Adsorbate labels to process.
        site_list:   Site labels to process.
        slab_root:   Directory containing per-system slab subdirectories
                     (e.g. ``data/structures/slabs/``).
        output_root: Root directory for all adsorption job directories
                     (e.g. ``work/Phase02_Adsorption/``).
        force:       Passed through to :func:`build_one`.

    Returns:
        ``(n_ok, n_fail)`` counts.
    """
    total = len(combos) * len(ads_list) * len(site_list)
    logger.info(
        "Batch: %d system(s) × %d adsorbate(s) × %d site(s) = %d task(s)",
        len(combos), len(ads_list), len(site_list), total,
    )
    ok = fail = 0
    for tm1, tm2 in combos:
        system_id = f"{tm1}{tm2}_NC"
        slab_path = slab_root / system_id / "CONTCAR"
        out_dir = output_root / system_id
        for ads in ads_list:
            for site in site_list:
                if build_one(slab_path, tm1, tm2, ads, site, out_dir, force=force):
                    ok += 1
                else:
                    fail += 1
    logger.info("Finished: %d succeeded, %d failed / %d total", ok, fail, total)
    return ok, fail


def _load_csv(csv_path: Path) -> list[tuple[str, str]]:
    """Read ``(TM1, TM2)`` pairs from a CSV file.

    Args:
        csv_path: Path to a CSV with at least ``TM1`` and ``TM2`` columns.

    Returns:
        List of ``(TM₁, TM₂)`` tuples.
    """
    combos: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            combos.append((row["TM1"].strip(), row["TM2"].strip()))
    logger.info("Loaded %d combinations from %s", len(combos), csv_path)
    return combos


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build initial adsorption structures from an optimised slab CONTCAR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--ads",
        nargs="+",
        default=ALL_ADS,
        metavar="ADS",
        help=f"Adsorbate(s): {' '.join(ALL_ADS)}. Default: all.",
    )
    p.add_argument(
        "--site",
        nargs="+",
        default=ALL_SITES,
        dest="site",
        metavar="SITE",
        help="Site(s): TM1 TM2. Default: both.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directories.",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tm1", metavar="ELEM", help="TM₁ element (requires --tm2 and --slab).")
    mode.add_argument("--csv", type=Path, metavar="FILE", help="CSV with TM1/TM2 columns.")

    p.add_argument("--tm2", metavar="ELEM", help="TM₂ element (used with --tm1).")
    p.add_argument(
        "--slab",
        type=Path,
        metavar="FILE",
        help="Optimised slab CONTCAR (used with --tm1/--tm2).",
    )
    p.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        help="Output directory for a single system (used with --tm1/--tm2).",
    )
    p.add_argument(
        "--slab-root",
        type=Path,
        default=Path("data/structures/slabs"),
        metavar="DIR",
        help="Root containing slab subdirs (used with --csv). Default: data/structures/slabs/",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/Phase02_Adsorption"),
        metavar="DIR",
        help="Root for adsorption job dirs (used with --csv). Default: work/Phase02_Adsorption/",
    )
    return p


def main() -> None:
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()

    unknown_ads = [a for a in args.ads if a not in ALL_ADS]
    if unknown_ads:
        logger.error("Unknown adsorbate(s): %s", unknown_ads)
        sys.exit(1)
    unknown_sites = [s for s in args.site if s not in ALL_SITES]
    if unknown_sites:
        logger.error("Unknown site(s): %s", unknown_sites)
        sys.exit(1)

    if args.csv:
        if not args.csv.exists():
            logger.error("CSV not found: %s", args.csv)
            sys.exit(1)
        combos = _load_csv(args.csv)
        ok, fail = build_batch(
            combos,
            ads_list=args.ads,
            site_list=args.site,
            slab_root=args.slab_root,
            output_root=args.output_root,
            force=args.force,
        )
    else:
        if not args.tm2:
            logger.error("--tm2 is required with --tm1.")
            sys.exit(1)
        if not args.slab:
            logger.error("--slab is required with --tm1/--tm2.")
            sys.exit(1)
        output = args.output or Path("work/Phase02_Adsorption") / f"{args.tm1}{args.tm2}_NC"
        ok = fail = 0
        for ads in args.ads:
            for site in args.site:
                if build_one(args.slab, args.tm1, args.tm2, ads, site, output, args.force):
                    ok += 1
                else:
                    fail += 1
        logger.info("Finished: %d succeeded, %d failed", ok, fail)

    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
