#!/usr/bin/env python3
"""Physical constants, element data, and project-wide configuration.

Reference energies are NOT stored here — load them at runtime from
``data/reference_energies.json`` via :func:`load_reference_energies`.

Usage::

    from scripts.utils.constants import MAGMOM_CONFIG, TM_SET, N_SLAB_C
    from scripts.utils.constants import load_reference_energies

    refs = load_reference_energies()
    E_CO2 = refs["CO2"]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Thermodynamics ────────────────────────────────────────────────────────────

T: float = 298.15          # K
KB: float = 8.617e-5       # eV / K
U_EQ_OER: float = 1.23     # V vs. RHE
U_EQ_CO2RR: float = -0.10  # V vs. RHE

# ΔG(*OOH) ≈ ΔG(*OH) + 3.2 eV  (universal scaling relation)
DELTA_G_OOH_SCALING_OFFSET: float = 3.2   # eV

# Frequency scaling factor for the fine-tuned MLIP (vs. VASP IBRION=5)
FREQ_SCALING_FACTOR: float = 1.069

# ── Slab geometry ─────────────────────────────────────────────────────────────
# Bare slab: 1×TM₁ + 1×TM₂ + 62×C + 6×N  =  70 atoms total
N_SLAB_C: int = 62
N_SLAB_N: int = 6
N_SLAB_TM: int = 2

# ── Transition metals (Tc excluded) ───────────────────────────────────────────

TM_LIST: list[str] = [
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",  # 3d
    "Y",  "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",        # 4d
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",               # 5d
]
TM_SET: frozenset[str] = frozenset(TM_LIST)

# ── Magnetic moments (μ_B) — initial values for INCAR MAGMOM ─────────────────
# Values follow the convention used in the original slab construction scripts.

MAGMOM_CONFIG: dict[str, float] = {
    # 3d
    "Sc": 1.0, "Ti": 2.0, "V":  3.0, "Cr": 4.0, "Mn": 5.0,
    "Fe": 4.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0, "Zn": 0.0,
    # 4d
    "Y":  1.0, "Zr": 2.0, "Nb": 3.0, "Mo": 2.0, "Ru": 2.0,
    "Rh": 1.0, "Pd": 0.0, "Ag": 0.0, "Cd": 0.0,
    # 5d
    "Hf": 2.0, "Ta": 3.0, "W":  2.0, "Re": 2.0, "Os": 2.0,
    "Ir": 1.0, "Pt": 0.0, "Au": 0.0,
    # Ligand atoms
    "C":  0.0, "N":  0.0, "H":  0.0, "O":  0.0,
}

# ── Adsorbate atom-count definitions ──────────────────────────────────────────
# Maps each adsorbate label to the expected number of *non-slab* atoms per element.
# Used to validate that identify_adsorbate_flags() found the correct atoms.

ADS_COMPOSITION: dict[str, dict[str, int]] = {
    "CO2":  {"C": 1, "O": 2},
    "COOH": {"C": 1, "O": 2, "H": 1},
    "CO":   {"C": 1, "O": 1},
    "H":    {"H": 1},
    "OH":   {"O": 1, "H": 1},
    "O":    {"O": 1},
}


# ── Reference energies ────────────────────────────────────────────────────────

def load_reference_energies(
    path: str | Path | None = None,
) -> dict[str, float]:
    """Load self-consistent gas-phase reference energies from JSON.

    Never hardcode reference energies in scripts — all values must come from
    this file so they stay consistent with the VASP settings used throughout
    the project.

    Args:
        path: Explicit path to ``reference_energies.json``.  Defaults to
              ``<project_root>/data/reference_energies.json``.

    Returns:
        Dict mapping species label (e.g. ``"CO2"``, ``"H2O"``) to DFT
        total energy in eV.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "reference_energies.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference energies file not found: {path}\n"
            "Compute gas-phase references with the project VASP settings and "
            "populate this file before calculating adsorption energies."
        )
    with path.open(encoding="utf-8") as fh:
        data: dict[str, float] = json.load(fh)
    logger.info("Loaded %d reference energies from %s", len(data), path)
    return data
