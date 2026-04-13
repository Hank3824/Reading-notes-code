#!/usr/bin/env python3
"""Stage 01b — CHGNet pre-relaxation of bare TM₁-TM₂-N-C slab structures.

Runs a fast CHGNet/FIRE pre-relaxation on the initial slab POSCAR produced by
build_slab.py, then writes the relaxed POSCAR back into the same slab directory
together with a fresh INCAR and KPOINTS.  The whole directory is then ready to
upload to the HPC cluster for VASP precise optimisation.
The cell is held fixed throughout (atomic positions only); the original lattice
matrix is restored after relaxation to avoid numerical drift.

Workflow::

    data/structures/slabs/{system_id}/
        ├── POSCAR      ← initial structure from build_slab.py (overwritten)
        ├── INCAR       ← regenerated from INCAR_opt template
        └── KPOINTS     ← Gamma 3×3×1

        ↓  CHGNet FIRE  (fmax=0.05 eV/Å, cell fixed)
        ↓  overwrite POSCAR with pre-relaxed geometry
        ↓  upload entire directory to HPC, add POTCAR
        ↓  VASP precise optimisation

    data/structures/slabs/{system_id}/CONTCAR   ← download after VASP finishes

CONTCAR is then used by build_adsorption.py as the slab reference structure.

Structural checks logged for each system:

* Graphene-C z-std  — should be < 0.05 Å (flat layer indicator)
* TM dz             — z-offset of each TM atom from mean graphene-C plane

Usage::

    # Single system
    python scripts/build/prerelax_slab.py --tm1 Ni --tm2 Zn

    # Batch from CSV
    python scripts/build/prerelax_slab.py --csv data/sampled_combinations.csv

    # Override convergence criteria
    python scripts/build/prerelax_slab.py \\
        --csv data/sampled_combinations.csv --fmax 0.03 --steps 800

    # Regenerate INCAR/KPOINTS from an already-relaxed POSCAR (no CHGNet re-run)
    python scripts/build/prerelax_slab.py \\
        --csv data/sampled_combinations.csv --regen-incar
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

from scripts.utils.constants import MAGMOM_CONFIG, TM_SET
from scripts.vasp.write_incar import write_incar
from scripts.vasp.write_kpoints import write_kpoints
from scripts.vasp.write_poscar import write_poscar

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
FMAX: float = 0.05       # eV/Å — loose pre-relax; VASP does the precise opt
MAX_STEPS: int = 500
KPOINTS_MESH: tuple[int, int, int] = (3, 3, 1)

SLAB_ROOT_DEFAULT = Path("data/structures/slabs")


# ── Structure utilities ───────────────────────────────────────────────────────

def _attach_magmom(structure: Structure) -> None:
    """Add initial magnetic moments as a ``"magmom"`` site property (in-place).

    Args:
        structure: Pymatgen Structure (mutated in-place).
    """
    structure.add_site_property(
        "magmom",
        [MAGMOM_CONFIG.get(s.specie.symbol, 0.0) for s in structure],
    )


def _fix_lattice(relaxed: Structure, reference: Structure) -> Structure:
    """Replace the lattice of *relaxed* with the one from *reference*.

    CHGNet's FIRE optimiser acts on atomic positions only (no cell DOF), so
    the lattice should not change.  This function restores it exactly to guard
    against floating-point drift and ensure that the output POSCAR has the
    same cell vectors as the DFT-optimised graphene template.

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


def _check_geometry(structure: Structure) -> dict[str, float]:
    """Compute structural quality metrics after relaxation.

    Metrics:

    * ``c_std``     — std of graphene-C z-coordinates (Å); should be < 0.05
    * ``tm_{elem}`` — z-offset of each TM atom from the mean graphene-C plane (Å)

    Args:
        structure: Relaxed Structure in Cartesian coordinates.

    Returns:
        Dict of metric names → float values.
    """
    symbols = [s.specie.symbol for s in structure]
    z = [s.coords[2] for s in structure]

    c_z = [z[i] for i, sym in enumerate(symbols) if sym == "C"]
    c_mean = float(np.mean(c_z)) if c_z else 0.0
    c_std = float(np.std(c_z)) if c_z else 0.0

    metrics: dict[str, float] = {"c_std": c_std}
    for i, sym in enumerate(symbols):
        if sym in TM_SET:
            metrics[f"tm_{sym}"] = z[i] - c_mean

    return metrics


# ── Core relaxation ───────────────────────────────────────────────────────────

def relax_one(
    tm1: str,
    tm2: str,
    slab_root: Path,
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
    force: bool = False,
) -> dict:
    """Run CHGNet pre-relaxation for one TM₁-TM₂-N-C slab system.

    Reads ``POSCAR`` from ``{slab_root}/{system_id}/``, runs CHGNet/FIRE, then
    overwrites POSCAR in-place and regenerates INCAR + KPOINTS in the same
    directory.  The directory is then ready for direct upload to HPC.

    Args:
        tm1:       TM₁ element symbol.
        tm2:       TM₂ element symbol.
        slab_root: Root containing per-system slab directories.
        fmax:      Force convergence threshold (eV/Å).
        max_steps: Maximum FIRE steps.
        force:     Overwrite existing pre-relaxed outputs when True.

    Returns:
        Dict with keys ``system_id``, ``status``, ``n_steps``, ``fmax_final``,
        ``c_std``, and per-TM ``tm_{elem}`` offsets.
    """
    # Lazy import: CHGNet is only needed when actually running relaxation
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        from pymatgen.io.ase import AseAtomsAdaptor
        from ase.optimize import FIRE as ASE_FIRE
    except ImportError as exc:
        raise ImportError(
            "CHGNet and/or ASE are required for pre-relaxation. "
            "Install with: pip install chgnet ase"
        ) from exc

    system_id = f"{tm1}{tm2}_NC"
    out_dir = slab_root / system_id
    poscar_in = out_dir / "POSCAR"
    poscar_out = out_dir / "POSCAR"

    result: dict = {"system_id": system_id, "status": "fail",
                    "n_steps": 0, "fmax_final": 999.0, "c_std": 999.0}

    if not poscar_in.exists():
        logger.warning("[%s] POSCAR not found: %s", system_id, poscar_in)
        return result

    if not force and poscar_out.exists() and (out_dir / "INCAR").exists():
        logger.info("[%s] already relaxed — skipping (use --force to overwrite)", system_id)
        result["status"] = "skip"
        return result

    try:
        reference = Structure.from_file(str(poscar_in))
        logger.info(
            "[%s] %d atoms  a=%.4f Å  c=%.4f Å",
            system_id,
            reference.num_sites,
            reference.lattice.a,
            reference.lattice.c,
        )

        # ── CHGNet relaxation (atomic positions only, cell fixed) ──────────
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(reference)
        magmoms = [MAGMOM_CONFIG.get(s, 0.0) for s in atoms.get_chemical_symbols()]
        atoms.set_initial_magnetic_moments(magmoms)

        chgnet = CHGNet.load(check_cuda_mem=False)
        atoms.calc = CHGNetCalculator(model=chgnet, check_cuda_mem=False)

        opt = ASE_FIRE(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)

        n_steps = opt.get_number_of_steps()
        forces = atoms.get_forces()
        fmax_final = float(np.max(np.linalg.norm(forces, axis=1)))
        converged = n_steps < max_steps

        logger.info(
            "[%s] %s  steps=%d  fmax=%.4f eV/Å",
            system_id,
            "converged" if converged else "NOT converged",
            n_steps,
            fmax_final,
        )

        # ── Rebuild pymatgen Structure, restore original lattice ───────────
        relaxed_raw = adaptor.get_structure(atoms)
        relaxed = _fix_lattice(relaxed_raw, reference)
        _attach_magmom(relaxed)

        # ── Structural quality checks ──────────────────────────────────────
        metrics = _check_geometry(relaxed)
        if metrics["c_std"] > 0.05:
            logger.warning(
                "[%s] graphene-C z-std = %.4f Å (> 0.05) — check structure",
                system_id,
                metrics["c_std"],
            )
        else:
            logger.info("[%s] graphene-C z-std = %.4f Å ✓", system_id, metrics["c_std"])
        for key, val in metrics.items():
            if key.startswith("tm_"):
                logger.info("[%s] %s dz = %+.4f Å", system_id, key[3:], val)

        # ── Write VASP inputs ──────────────────────────────────────────────
        write_poscar(relaxed, output_dir=out_dir,
                     comment=f"{system_id}-prerelax")
        write_incar(mode="opt_slab", output_dir=out_dir,
                    system_id=system_id, structure=relaxed)
        write_kpoints(mesh=KPOINTS_MESH, output_dir=out_dir)

        logger.info("[%s] ✓ VASP inputs → %s", system_id, out_dir)

        result.update({
            "status": "ok" if converged else "unconverged",
            "n_steps": n_steps,
            "fmax_final": fmax_final,
            **metrics,
        })
        return result

    except Exception:
        logger.exception("[%s] unexpected error", system_id)
        return result


def regen_incar_one(
    tm1: str,
    tm2: str,
    slab_root: Path,
) -> bool:
    """Regenerate INCAR and KPOINTS from an existing pre-relaxed POSCAR.

    Useful after INCAR template changes without re-running CHGNet.

    Args:
        tm1:       TM₁ element symbol.
        tm2:       TM₂ element symbol.
        slab_root: Root containing ``{system_id}/POSCAR``.

    Returns:
        True on success, False if POSCAR not found.
    """
    system_id = f"{tm1}{tm2}_NC"
    out_dir = slab_root / system_id
    poscar = out_dir / "POSCAR"

    if not poscar.exists():
        logger.warning("[%s] POSCAR not found in %s — skipping", system_id, out_dir)
        return False

    structure = Structure.from_file(str(poscar))
    _attach_magmom(structure)
    write_incar(mode="opt_slab", output_dir=out_dir,
                system_id=system_id, structure=structure)
    write_kpoints(mesh=KPOINTS_MESH, output_dir=out_dir)
    logger.info("[%s] INCAR + KPOINTS regenerated → %s", system_id, out_dir)
    return True


# ── Batch runner ──────────────────────────────────────────────────────────────

def build_batch(
    combos: list[tuple[str, str]],
    slab_root: Path,
    fmax: float = FMAX,
    max_steps: int = MAX_STEPS,
    force: bool = False,
    regen_incar: bool = False,
) -> tuple[int, int]:
    """Run pre-relaxation (or INCAR regeneration) for a list of systems.

    Args:
        combos:      List of ``(TM₁, TM₂)`` pairs.
        slab_root:   Root containing per-system slab directories.
        fmax:        Force convergence threshold (eV/Å).
        max_steps:   Maximum FIRE steps.
        force:       Overwrite existing outputs.
        regen_incar: If True, only regenerate INCAR/KPOINTS (no CHGNet).

    Returns:
        ``(n_ok, n_fail)`` counts.
    """
    ok = fail = 0
    log_rows: list[dict] = []

    for tm1, tm2 in combos:
        if regen_incar:
            success = regen_incar_one(tm1, tm2, slab_root)
            ok += success
            fail += not success
        else:
            res = relax_one(tm1, tm2, slab_root, fmax, max_steps, force)
            log_rows.append(res)
            if res["status"] in ("ok", "skip", "unconverged"):
                ok += 1
            else:
                fail += 1

    if log_rows:
        _write_log(log_rows, slab_root)

    logger.info("Finished: %d succeeded, %d failed / %d total", ok, fail, ok + fail)
    return ok, fail


def _write_log(rows: list[dict], slab_root: Path) -> None:
    """Write a CSV summary of pre-relaxation results."""
    log_path = slab_root / "prerelax_log.csv"
    slab_root.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Collect all unique fieldnames across all rows — TM-specific keys (tm_X)
    # differ per system, so rows[0].keys() alone is insufficient.
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Log written → %s", log_path)


def _load_csv(csv_path: Path) -> list[tuple[str, str]]:
    """Read ``(TM1, TM2)`` pairs from a CSV file."""
    combos: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            combos.append((row["TM1"].strip(), row["TM2"].strip()))
    logger.info("Loaded %d combinations from %s", len(combos), csv_path)
    return combos


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CHGNet pre-relaxation of bare TM₁-TM₂-N-C slab structures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--slab-root",
        type=Path,
        default=SLAB_ROOT_DEFAULT,
        metavar="DIR",
        help=(
            f"Root containing per-system slab directories. "
            f"POSCAR is read from and written back to this location. "
            f"(default: {SLAB_ROOT_DEFAULT})"
        ),
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=FMAX,
        metavar="F",
        help=f"Force convergence threshold in eV/Å (default: {FMAX})",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=MAX_STEPS,
        metavar="N",
        help=f"Maximum FIRE steps (default: {MAX_STEPS})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    p.add_argument(
        "--regen-incar",
        action="store_true",
        help=(
            "Regenerate INCAR and KPOINTS from the existing pre-relaxed POSCAR "
            "without re-running CHGNet (useful after template changes)."
        ),
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tm1", metavar="ELEM", help="TM₁ element (requires --tm2).")
    mode.add_argument("--csv", type=Path, metavar="FILE",
                      help="CSV file with TM1 and TM2 columns.")
    p.add_argument("--tm2", metavar="ELEM", help="TM₂ element (used with --tm1).")
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
    else:
        if not args.tm2:
            logger.error("--tm2 is required with --tm1.")
            sys.exit(1)
        combos = [(args.tm1, args.tm2)]

    ok, fail = build_batch(
        combos,
        slab_root=args.slab_root,
        fmax=args.fmax,
        max_steps=args.steps,
        force=args.force,
        regen_incar=args.regen_incar,
    )
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
