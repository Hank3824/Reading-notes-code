#!/usr/bin/env python3
"""Stage 02b — CHGNet pre-relaxation of initial adsorption structures.

Reads the initial POSCAR produced by build_adsorption.py (Selective Dynamics:
slab atoms F F F, adsorbate atoms T T T), runs a fast CHGNet/FIRE pre-relaxation
with the slab atoms frozen and the adsorbate free to relax in all directions,
then overwrites the POSCAR in-place with the pre-optimised geometry.

The INCAR and KPOINTS written by build_adsorption.py remain unchanged.  The
pre-relaxed directory is then ready for MLIP single-point labelling (Stage 02c)
or direct VASP optimisation.

The original lattice is restored after relaxation to avoid numerical drift from
the CHGNet cell.

Workflow::

    work/Phase02_Adsorption/{system_id}/{ads}_{site}/
        POSCAR    ← initial structure from build_adsorption.py  (overwritten)
        INCAR     ← written by build_adsorption.py              (unchanged)
        KPOINTS   ← written by build_adsorption.py              (unchanged)

        ↓  CHGNet FIRE  (slab atoms fixed, adsorbate free, cell fixed)
        ↓  overwrite POSCAR with pre-relaxed geometry
        ↓  upload directory to HPC, add POTCAR, run VASP

Structural quality logged per system:

* ``fmax_ads``  — max force on adsorbate atoms (eV/Å) at convergence
* ``d_tm_ads``  — TM–adsorbate nearest-atom bond length (Å)

Skip logic:
    A POSCAR whose first comment line contains "prerelax" is treated as already
    pre-relaxed and skipped.  Use ``--force`` to override.

Usage::

    # Single system, one adsorbate/site
    python scripts/build/prerelax_adsorption.py \\
        --tm1 Ni --tm2 Zn --ads COOH --site TM1

    # Single system, all adsorbates and both sites
    python scripts/build/prerelax_adsorption.py --tm1 Ni --tm2 Zn

    # Batch from CSV (columns: TM1, TM2)
    python scripts/build/prerelax_adsorption.py \\
        --csv data/sampled_combinations.csv

    # Batch, specific adsorbate subset, with force overwrite
    python scripts/build/prerelax_adsorption.py \\
        --csv data/sampled_combinations.csv --ads COOH CO --force

    # Custom adsorption root
    python scripts/build/prerelax_adsorption.py \\
        --csv data/sampled_combinations.csv \\
        --ads-root work/Phase02_Adsorption/
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils.constants import MAGMOM_CONFIG, TM_SET
from scripts.vasp.write_poscar import write_poscar

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
FMAX: float = 0.05        # eV/Å — loose pre-relax; VASP/MLIP does the fine opt
MAX_STEPS: int = 500

ADS_ROOT_DEFAULT = Path("work/Phase02_Adsorption")

ALL_ADS: list[str] = ["CO2", "COOH", "CO", "H", "OH", "O"]
ALL_SITES: list[str] = ["TM1", "TM2"]

# Sentinel string embedded in POSCAR comment after pre-relaxation
_PRERELAX_TAG: str = "prerelax"


# ── Structure utilities ───────────────────────────────────────────────────────

def _attach_magmom(structure: Structure) -> None:
    """Attach initial magnetic moments as a ``"magmom"`` site property.

    Args:
        structure: Pymatgen Structure (mutated in-place).
    """
    structure.add_site_property(
        "magmom",
        [MAGMOM_CONFIG.get(s.specie.symbol, 0.0) for s in structure],
    )


def _fix_lattice(relaxed: Structure, reference: Structure) -> Structure:
    """Replace the lattice of *relaxed* with the original from *reference*.

    CHGNet FIRE optimises only atomic positions (no cell DOF), but small
    numerical drift can accumulate.  This function restores the lattice exactly.

    Args:
        relaxed:   Structure after CHGNet relaxation.
        reference: Original structure whose lattice should be preserved.

    Returns:
        New Structure with *reference* lattice and *relaxed* fractional coords.
    """
    return Structure(
        lattice=reference.lattice,
        species=[s.specie for s in relaxed],
        coords=relaxed.frac_coords,
        coords_are_cartesian=False,
        site_properties=relaxed.site_properties,
    )


def _read_poscar_with_sd(
    poscar_path: Path,
) -> tuple[Structure, list[list[bool]] | None]:
    """Read a POSCAR and return (structure, selective_dynamics).

    Args:
        poscar_path: Path to the POSCAR file.

    Returns:
        Tuple of the pymatgen Structure and per-atom SD flags
        (``[bool, bool, bool]`` per atom), or ``None`` if no SD block.
    """
    poscar = Poscar.from_file(
        str(poscar_path),
        check_for_POTCAR=False,
        read_velocities=False,
    )
    return poscar.structure, poscar.selective_dynamics


def _fixed_indices(
    sd_flags: list[list[bool]] | None,
) -> list[int]:
    """Return indices of atoms flagged F F F in Selective Dynamics.

    Args:
        sd_flags: Per-atom ``[bool, bool, bool]`` list, or ``None`` for no SD.

    Returns:
        Zero-based atom indices to freeze in the ASE calculator.
    """
    if sd_flags is None:
        return []
    return [i for i, flags in enumerate(sd_flags) if not any(flags)]


def _check_geometry(
    structure: Structure,
    tm_elem: str,
    ads_indices: list[int],
) -> dict[str, float]:
    """Compute post-relaxation geometry metrics for the adsorbate.

    Args:
        structure:   Relaxed Structure.
        tm_elem:     Element symbol of the adsorption-site TM atom.
        ads_indices: Indices of adsorbate atoms.

    Returns:
        Dict with ``d_tm_ads`` (Å) — TM to nearest adsorbate atom distance.
    """
    metrics: dict[str, float] = {}
    tm_index = next(
        (i for i, s in enumerate(structure) if s.specie.symbol == tm_elem),
        None,
    )
    if tm_index is None or not ads_indices:
        return metrics

    tm_cart = structure[tm_index].coords
    distances = [
        float(np.linalg.norm(structure[i].coords - tm_cart))
        for i in ads_indices
    ]
    metrics["d_tm_ads"] = min(distances)
    return metrics


# ── Core relaxation ───────────────────────────────────────────────────────────

def relax_one(
    tm1: str,
    tm2: str,
    ads: str,
    site: str,
    ads_root: Path,
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
    force: bool = False,
) -> dict:
    """Run CHGNet pre-relaxation for one adsorption structure.

    Reads ``POSCAR`` from ``{ads_root}/{system_id}/{ads}_{site}/``, runs
    CHGNet/FIRE with slab atoms frozen and adsorbate atoms free, then overwrites
    the POSCAR in-place.  INCAR and KPOINTS are not modified.

    Args:
        tm1:       TM₁ element symbol.
        tm2:       TM₂ element symbol.
        ads:       Adsorbate label (``"CO2"``, ``"COOH"``, ``"CO"``, etc.).
        site:      ``"TM1"`` or ``"TM2"``.
        ads_root:  Root containing ``{system_id}/{ads}_{site}/POSCAR``.
        fmax:      Force convergence threshold (eV/Å).
        max_steps: Maximum FIRE steps.
        force:     Re-run even if POSCAR already bears the pre-relax tag.

    Returns:
        Dict with keys: ``system_id``, ``ads``, ``site``, ``status``,
        ``n_steps``, ``fmax_final``, ``d_tm_ads``.
    """
    # Lazy import: avoid slow CHGNet/ASE load on import when not needed
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        from pymatgen.io.ase import AseAtomsAdaptor
        from ase.constraints import FixAtoms
        from ase.optimize import FIRE as ASE_FIRE
    except ImportError as exc:
        raise ImportError(
            "CHGNet and ASE are required for pre-relaxation. "
            "Install with: pip install chgnet ase"
        ) from exc

    system_id = f"{tm1}{tm2}_NC"
    label = f"{system_id}/{ads}_{site}"
    target_elem = tm1 if site == "TM1" else tm2
    job_dir = ads_root / system_id / f"{ads}_{site}"
    poscar_path = job_dir / "POSCAR"

    result: dict = {
        "system_id": system_id, "ads": ads, "site": site,
        "status": "fail", "n_steps": 0, "fmax_final": 999.0, "d_tm_ads": float("nan"),
    }

    if not poscar_path.exists():
        logger.warning("[%s] POSCAR not found: %s", label, poscar_path)
        return result

    # Skip if already pre-relaxed (sentinel in comment line)
    if not force:
        comment = poscar_path.read_text(encoding="utf-8").splitlines()[0]
        if _PRERELAX_TAG in comment.lower():
            logger.info("[%s] already pre-relaxed — skipping (use --force)", label)
            result["status"] = "skip"
            return result

    try:
        structure, sd_flags = _read_poscar_with_sd(poscar_path)
        logger.info(
            "[%s] %d atoms  a=%.4f Å  c=%.4f Å",
            label, structure.num_sites,
            structure.lattice.a, structure.lattice.c,
        )

        fixed_idx = _fixed_indices(sd_flags)
        ads_idx = [i for i in range(structure.num_sites) if i not in set(fixed_idx)]
        logger.info(
            "[%s] slab (frozen): %d atoms  adsorbate (free): %d atoms",
            label, len(fixed_idx), len(ads_idx),
        )
        if not ads_idx:
            logger.warning(
                "[%s] no free atoms found — check SD flags in POSCAR", label
            )

        # ── CHGNet relaxation (cell fixed, slab frozen) ────────────────────
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        magmoms = [MAGMOM_CONFIG.get(sym, 0.0) for sym in atoms.get_chemical_symbols()]
        atoms.set_initial_magnetic_moments(magmoms)

        if fixed_idx:
            atoms.set_constraint(FixAtoms(indices=fixed_idx))

        chgnet = CHGNet.load(check_cuda_mem=False)
        atoms.calc = CHGNetCalculator(model=chgnet, check_cuda_mem=False)

        opt = ASE_FIRE(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)

        n_steps = opt.get_number_of_steps()
        converged = n_steps < max_steps

        # Report force on adsorbate atoms only
        forces_all = atoms.get_forces()
        if ads_idx:
            fmax_ads = float(np.max(np.linalg.norm(forces_all[ads_idx], axis=1)))
        else:
            fmax_ads = float(np.max(np.linalg.norm(forces_all, axis=1)))

        logger.info(
            "[%s] %s  steps=%d  fmax_ads=%.4f eV/Å",
            label,
            "converged" if converged else "NOT converged",
            n_steps,
            fmax_ads,
        )

        # ── Rebuild pymatgen Structure, restore original lattice ───────────
        relaxed_raw = adaptor.get_structure(atoms)
        relaxed = _fix_lattice(relaxed_raw, structure)
        _attach_magmom(relaxed)

        # ── Geometry quality check ─────────────────────────────────────────
        metrics = _check_geometry(relaxed, target_elem, ads_idx)
        for key, val in metrics.items():
            logger.info("[%s] %s = %.4f Å", label, key, val)

        # ── Overwrite POSCAR (SD flags preserved) ─────────────────────────
        write_poscar(
            relaxed,
            output_dir=job_dir,
            comment=f"{label}-{_PRERELAX_TAG}",
            selective_dynamics=sd_flags,
        )
        logger.info("[%s] ✓ POSCAR overwritten → %s", label, poscar_path)

        result.update({
            "status": "ok" if converged else "unconverged",
            "n_steps": n_steps,
            "fmax_final": round(fmax_ads, 5),
            **{k: round(v, 4) for k, v in metrics.items()},
        })
        return result

    except Exception:
        logger.exception("[%s] unexpected error", label)
        return result


# ── Batch runner ──────────────────────────────────────────────────────────────

def build_batch(
    combos: list[tuple[str, str]],
    ads_list: list[str],
    site_list: list[str],
    ads_root: Path,
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
    force: bool = False,
) -> tuple[int, int]:
    """Run :func:`relax_one` for all (system, adsorbate, site) combinations.

    Args:
        combos:    List of ``(TM₁, TM₂)`` pairs.
        ads_list:  Adsorbate labels to process.
        site_list: Site labels to process (``"TM1"``, ``"TM2"``).
        ads_root:  Root containing per-system adsorption job directories.
        fmax:      Force convergence threshold (eV/Å).
        max_steps: Maximum FIRE steps.
        force:     Overwrite existing pre-relaxed outputs.

    Returns:
        ``(n_ok, n_fail)`` counts.  Skipped jobs count as ``n_ok``.
    """
    total = len(combos) * len(ads_list) * len(site_list)
    logger.info(
        "Batch: %d system(s) × %d adsorbate(s) × %d site(s) = %d task(s)",
        len(combos), len(ads_list), len(site_list), total,
    )

    ok = fail = 0
    log_rows: list[dict] = []

    for tm1, tm2 in combos:
        for ads in ads_list:
            for site in site_list:
                res = relax_one(tm1, tm2, ads, site, ads_root, fmax, max_steps, force)
                log_rows.append(res)
                if res["status"] in ("ok", "skip", "unconverged"):
                    ok += 1
                else:
                    fail += 1

    _write_log(log_rows, ads_root)
    logger.info("Finished: %d succeeded, %d failed / %d total", ok, fail, total)
    return ok, fail


def _write_log(rows: list[dict], ads_root: Path) -> None:
    """Write a CSV run summary to ``{ads_root}/prerelax_ads_log.csv``."""
    if not rows:
        return
    log_path = ads_root / "prerelax_ads_log.csv"
    ads_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Log written → %s", log_path)


def _load_csv(csv_path: Path) -> list[tuple[str, str]]:
    """Read ``(TM1, TM2)`` pairs from a CSV file with ``TM1`` / ``TM2`` columns."""
    combos: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            combos.append((row["TM1"].strip(), row["TM2"].strip()))
    logger.info("Loaded %d combinations from %s", len(combos), csv_path)
    return combos


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CHGNet pre-relaxation of initial adsorption structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--ads-root",
        type=Path,
        default=ADS_ROOT_DEFAULT,
        metavar="DIR",
        help=(
            "Root directory containing per-system adsorption job subdirectories. "
            "POSCAR is read from and overwritten in "
            "{ads_root}/{system_id}/{ads}_{site}/. "
            f"(default: {ADS_ROOT_DEFAULT})"
        ),
    )
    p.add_argument(
        "--ads",
        nargs="+",
        default=ALL_ADS,
        metavar="ADS",
        choices=ALL_ADS,
        help=f"Adsorbate(s) to process. Default: all ({' '.join(ALL_ADS)}).",
    )
    p.add_argument(
        "--site",
        nargs="+",
        default=ALL_SITES,
        metavar="SITE",
        choices=ALL_SITES,
        help="Site(s) to process: TM1 TM2. Default: both.",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=FMAX,
        metavar="F",
        help=f"Force convergence threshold in eV/Å (default: {FMAX}).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=MAX_STEPS,
        metavar="N",
        help=f"Maximum FIRE optimisation steps (default: {MAX_STEPS}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite POSCARs that already bear the pre-relax tag.",
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tm1",
        metavar="ELEM",
        help="TM₁ element symbol for single-system mode (requires --tm2).",
    )
    mode.add_argument(
        "--csv",
        type=Path,
        metavar="FILE",
        help="CSV file with TM1 and TM2 columns for batch mode.",
    )
    p.add_argument(
        "--tm2",
        metavar="ELEM",
        help="TM₂ element symbol (required with --tm1).",
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

    if args.csv:
        if not args.csv.exists():
            logger.error("CSV not found: %s", args.csv)
            sys.exit(1)
        combos = _load_csv(args.csv)
        ok, fail = build_batch(
            combos,
            ads_list=args.ads,
            site_list=args.site,
            ads_root=args.ads_root,
            fmax=args.fmax,
            max_steps=args.steps,
            force=args.force,
        )
    else:
        if not args.tm2:
            logger.error("--tm2 is required with --tm1.")
            sys.exit(1)
        ok = fail = 0
        log_rows: list[dict] = []
        for ads in args.ads:
            for site in args.site:
                res = relax_one(
                    args.tm1, args.tm2, ads, site,
                    ads_root=args.ads_root,
                    fmax=args.fmax,
                    max_steps=args.steps,
                    force=args.force,
                )
                log_rows.append(res)
                if res["status"] in ("ok", "skip", "unconverged"):
                    ok += 1
                else:
                    fail += 1
        _write_log(log_rows, args.ads_root)
        logger.info("Finished: %d succeeded, %d failed", ok, fail)

    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
