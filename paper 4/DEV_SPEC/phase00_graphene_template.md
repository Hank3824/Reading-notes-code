# Phase 00 — Graphene Template Preparation

**Scope**: Build and optimize the NiZn-N-C 6×6 slab template used by `build_slab.py`.  
**Output**: `NiZn-N-C.cif` → copied to `data/structures/graphene_template.json` (pymatgen JSON)  
**No scripts required**: all structure construction is done in Materials Studio; DFT jobs are
submitted manually to the cluster.

---

## Overview

```
Step 1 │ Primitive graphene cell optimization  → a_DFT = 2.4670 Å
Step 2 │ 6×6 supercell + double-vacancy construction in Materials Studio
Step 3 │ NiZn-N-C defect structure construction in Materials Studio
Step 4 │ 6×6 NiZn-N-C slab optimization  → NiZn-N-C.cif (template)
```

The optimized template is used **as-is** by `build_slab.py`, which only replaces the Ni/Zn
sites with other TM elements.  Geometry coordinates come from this DFT-relaxed template,
so it is critical to converge this step properly.

---

## Step 1 — Primitive Cell Construction and DFT Optimization

### 1.1 Build in Materials Studio

```
Build → Crystals → Build Crystal
  Crystal system : Hexagonal
  a = b          : 2.4670 Å   ← initial guess, will be updated after DFT
  c              : 15.0 Å     ← vacuum layer included from the start
  γ              : 120°

Atoms (fractional coordinates):
  C1 : (0,     0,     0.5)
  C2 : (1/3, 2/3,   0.5)
```

### 1.2 INCAR — Primitive Cell Optimization

```fortran
SYSTEM = graphene_primitive_opt

! ── Initialization ──────────────────────────────────────────────────────────
ISTART = 0
ICHARG = 2

! ── Accuracy and cutoff ─────────────────────────────────────────────────────
ENCUT  = 520        ! Higher than the project standard (500 eV) because we are
                    ! optimizing the cell, and lattice-constant convergence
                    ! requires a slightly higher cutoff to suppress Pulay stress.
PREC   = Accurate
GGA    = PE         ! Explicit PBE functional specification

! ── Electronic steps ────────────────────────────────────────────────────────
ALGO   = Normal     ! Davidson diagonalization; adequate for non-magnetic carbon
NELM   = 200
NELMIN = 6
EDIFF  = 1E-6       ! Tighter than the project default (1E-5) because we need
                    ! a well-converged total energy to obtain a reliable a_DFT

! ── Smearing ────────────────────────────────────────────────────────────────
ISMEAR = 0          ! Gaussian smearing
                    ! Graphene is a zero-gap semiconductor (Dirac point); do NOT
                    ! use ISMEAR = -5 (tetrahedron, for insulators only) or
                    ! ISMEAR = 1 (Methfessel-Paxton, intended for metals).
SIGMA  = 0.05       ! Small broadening width (eV). Graphene's DOS near the Fermi
                    ! level is nearly zero, so a small SIGMA prevents artificial
                    ! smearing of the Dirac cone and reduces the associated error
                    ! in the total energy.

! ── Spin ────────────────────────────────────────────────────────────────────
ISPIN  = 1          ! Pure carbon, no magnetism

! ── Ionic relaxation ────────────────────────────────────────────────────────
IBRION = 2          ! Conjugate gradient
NSW    = 100
ISIF   = 3          ! Relax atomic positions + cell shape + cell volume
                    ! (ISIF=3 is required to optimize the lattice constant a)
EDIFFG = -0.001     ! Tight force threshold (eV/Å) so that a_DFT is accurate
POTIM  = 0.5

! ── Projection ──────────────────────────────────────────────────────────────
LREAL  = .FALSE.    ! Use reciprocal-space projectors.
                    ! The primitive cell has only 2 atoms; real-space projectors
                    ! (LREAL = Auto or .TRUE.) introduce significant errors for
                    ! such small systems. Switch to LREAL = Auto only when the
                    ! supercell exceeds ~50 atoms.

! ── van der Waals ───────────────────────────────────────────────────────────
IVDW   = 12         ! DFT-D3(BJ); consistent with the rest of the project

! ── Output ──────────────────────────────────────────────────────────────────
LWAVE  = .FALSE.
LCHARG = .FALSE.
LORBIT = 0          ! No DOS needed for lattice-constant optimization

! ── Parallelization ─────────────────────────────────────────────────────────
! See §Parallelization Notes at the end of this file.
NCORE  = 4          ! Suitable for the primitive cell (2 atoms, few bands).
                    ! For 56-core nodes: 56 / 4 = 14 NPAR groups.
```

### 1.3 KPOINTS — Primitive Cell

Generate with VASPKIT (recommended):

```bash
vaspkit -task 102 -kgamma T -kspacing 0.03
```

Or write manually (Γ-centered, fine mesh for accurate lattice constant):

```
Automatic Gamma
0
Gamma
17  17  1
0  0  0
```

> A Rk ≈ 0.03 Å⁻¹ spacing for a 2-atom primitive cell typically gives ~17×17×1.
> Do not use the coarser 3×3×1 mesh here — the lattice constant is sensitive to
> k-point sampling in the primitive cell.

### 1.4 Expected Result

After convergence, extract `a_DFT` from the CONTCAR:

```bash
grep -A1 "direct" CONTCAR | head -1   # read lattice vectors
```

**Target**: `a_DFT = 2.4670 Å`  (PBE + D3-BJ, consistent with the CIF template)

---

## Step 2 — 6×6 Supercell Construction

### 2.1 Build in Materials Studio

1. Open the optimized primitive cell CONTCAR (or re-input with `a = 2.4670 Å`).
2. Expand to supercell:
   ```
   Build → Symmetry → Supercell → 6 × 6 × 1
   ```
   Result: **72 C atoms**, a = b = 14.8020 Å, c = 15.0 Å, γ = 120°.

### 2.2 Create Double Vacancy

Remove **two adjacent C atoms** from the center of the supercell to form a
divacancy. These two vacated sites will be replaced by the TM₁ and TM₂ atoms.

> Choose a central location so the defect is far from periodic images
> (~7.4 Å to the nearest image — sufficient for a 6×6 cell).

---

## Step 3 — NiZn-N-C Template Construction

### 3.1 N₄ Coordination Environment

Replace the **four C atoms** nearest to the divacancy with N atoms.
This creates the N₄ pocket that coordinates the two TM sites.

After substitution the composition should be:
- **62 C** + **6 N** (4 coordinating N + 2 further-neighbor N from graphene band,
  check the CIF label list) + vacant TM sites

> The current template (`NiZn-N-C.cif`) contains exactly 62 C + 6 N + Ni + Zn = 70 atoms.

### 3.2 Place Ni and Zn

Insert Ni at TM₁ site and Zn at TM₂ site above the graphene plane.
Approximate starting z ≈ 1.7–2.0 Å above the C plane; the exact position will
be found by DFT relaxation in Step 4.

Export as `NiZn_NC_init.cif` or directly as POSCAR.

---

## Step 4 — 6×6 NiZn-N-C Slab Optimization

### 4.1 INCAR — Slab Optimization

```fortran
SYSTEM = NiZn_NC_6x6_slab_opt

! ── Initialization ──────────────────────────────────────────────────────────
ISTART = 0
ICHARG = 2

! ── Accuracy and cutoff ─────────────────────────────────────────────────────
ENCUT  = 500        ! Project standard cutoff (eV)
PREC   = Accurate
GGA    = PE

! ── Electronic steps ────────────────────────────────────────────────────────
ALGO   = All        ! RMM-DIIS + Davidson; recommended for transition metals
                    ! (ALGO = Normal can stall for Ni/Zn d-electrons)
NELM   = 200
NELMIN = 6
EDIFF  = 1E-5       ! Project standard convergence criterion

! ── Smearing ────────────────────────────────────────────────────────────────
ISMEAR = 0          ! Gaussian smearing; the slab is semi-metallic
SIGMA  = 0.05       ! Keep the same small value as the primitive cell

! ── Spin ────────────────────────────────────────────────────────────────────
ISPIN  = 2          ! Ni has d⁸ configuration and carries a local moment;
                    ! always use ISPIN = 2 for TM-containing slabs
MAGMOM = 62*0 6*0 2.0 0.0   ! Ni(magmom=2), Zn(magmom=0); all C and N set to 0
                              ! Atom order must match the POSCAR species order

! ── Ionic relaxation ────────────────────────────────────────────────────────
IBRION = 2
NSW    = 300
ISIF   = 2          ! Fix cell shape and volume; relax atomic positions only.
                    ! The lattice constant is already set to a_DFT = 2.4670 Å.
EDIFFG = -0.02      ! Standard force threshold for slab optimization (eV/Å)
POTIM  = 0.5

! ── Projection ──────────────────────────────────────────────────────────────
LREAL  = Auto       ! 70 atoms; real-space projection is efficient and accurate
                    ! enough for this system size (> ~50 atoms threshold)

! ── van der Waals ───────────────────────────────────────────────────────────
IVDW   = 12         ! DFT-D3(BJ); important for graphene stacking and TM binding

! ── Output ──────────────────────────────────────────────────────────────────
LWAVE  = .TRUE.     ! Keep WAVECAR for potential charge-density analysis
LCHARG = .TRUE.     ! Keep CHGCAR for charge-density difference plots
LORBIT = 11         ! Write DOSCAR; needed for d-band center analysis in Phase 06

! ── Implicit solvation (VASPsol) ────────────────────────────────────────────
! Not needed for the bare template optimization; enable in Phase 02 sp jobs.
! LSOL   = .TRUE.
! EB_K   = 78.4

! ── Parallelization ─────────────────────────────────────────────────────────
! See §Parallelization Notes at the end of this file.
NCORE  = 7          ! Recommended for 56-core nodes with the 70-atom slab.
                    ! 56 / 7 = 8 NPAR groups (see detailed notes below).
```

### 4.2 KPOINTS — 6×6 Slab

Generate with VASPKIT:

```bash
vaspkit -task 102 -kgamma T -kspacing 0.04
```

For a 6×6 graphene supercell (a = 14.8 Å), a 0.04 Å⁻¹ spacing gives **3×3×1**:

```
Automatic Gamma
0
Gamma
3  3  1
0  0  0
```

> A 3×3×1 Γ-centered mesh is sufficient for the 6×6 slab.
> The fine 17×17×1 mesh used in the primitive cell is not needed here.

### 4.3 Selective Dynamics (optional)

For the template slab, it is acceptable to relax all atoms (no frozen layers) because
the graphene sheet is only one atom thick and there is no bulk-like subsurface.
If CONTCAR shows unphysical C displacements, fix the outer ring of C atoms:

```
Selective dynamics
Direct
  x  y  z  F  F  F   ← TM, N, inner C: free
  x  y  z  F  F  F
  ...
  x  y  z  F  F  T   ← outer-ring C: fix z only
```

### 4.4 Expected Result

After convergence:

| Quantity | Expected range |
|---|---|
| Ni–N bond length | 1.9–2.1 Å |
| Zn–N bond length | 1.9–2.2 Å |
| Ni z-displacement above graphene | 0.8–1.2 Å |
| Zn z-displacement above graphene | 0.4–0.8 Å |
| Max residual force | < 0.02 eV/Å |

Export the converged CONTCAR as **`NiZn-N-C.cif`** from Materials Studio (or
use `pymatgen`):

```python
from pymatgen.core import Structure
s = Structure.from_file("CONTCAR")
s.to(fmt="json", filename="data/structures/graphene_template.json")
```

---

## Parallelization Notes — NCORE on 56-Core Nodes

### Background

`NCORE` controls how many MPI ranks collaborate on a single orbital (band).
`NPAR = NPROC / NCORE` is the number of parallel orbital groups.  The product
must equal the total number of cores used: `NCORE × NPAR = NPROC`.

```
NPROC = NCORE × NPAR
         ↑           ↑
 cores per orbital   number of groups
```

Choosing NCORE correctly matters for both **efficiency** (avoids idle CPUs) and
**correctness** (VASP will abort if NPROC is not divisible by NCORE).

### 56-Core Node Factorization

```
56 = 2³ × 7
Valid divisors (= valid NCORE values): 1, 2, 4, 7, 8, 14, 28, 56
```

> ⚠ Unlike most cluster nodes (32, 48, 64 cores), 56 is **not a power of 2**.
> NCORE = 16 or NCORE = 32 are commonly seen in tutorials but will cause VASP
> to abort on a 56-core node because 56 / 16 = 3.5 is not an integer.

### Recommended Values

| Calculation | System | NCORE | NPAR | Rationale |
|---|---|---|---|---|
| Step 1 primitive opt | 2 atoms, ~8 bands | **4** | 14 | Small band count; keep groups small to avoid idle cores |
| Step 4 slab opt | 70 atoms, ~180 bands | **7** | 8 | Balances band-level and k-point parallelism |
| Phase 02 SP (LFT label) | 70 atoms, ISPIN=2 | **7** | 8 | Same system size |
| Phase 02 SP (adsorption) | ~75 atoms, ISPIN=2 | **7** | 8 | Slightly more bands; 7 still fits |

### Why NCORE = 7 for the 70-Atom Slab?

```
70 atoms × ISPIN=2 → ~200–220 bands (up + down combined)
K-grid: 3×3×1 Γ-centered → ~6 irreducible k-points

With NCORE = 7, NPAR = 8:
  - 8 groups handle ~1 k-point each (6 active + 2 waiting at k-point level)
  - Within each group, 7 cores share the work on each orbital
  - Orbital-level scaling: reasonable for ~200 bands / 8 groups ≈ 25 bands/group

With NCORE = 8, NPAR = 7: also valid (56/8 = 7 ✓)
  - Slightly more efficient if the number of bands per group is a multiple of 8
  - Use this if you observe idle time with NCORE = 7

With NCORE = 14, NPAR = 4: good when k-points are the bottleneck
  - Only 4 groups for 6 k-points → some groups idle
  - Not recommended for 3×3×1 meshes
```

### Summary Table for Quick Reference

```
NPROC = 56 (one full node)

NCORE │ NPAR │ Use when
──────┼──────┼──────────────────────────────────────────────────────
  4   │  14  │ Small systems (< 20 atoms), many k-points (primitive cell)
  7   │   8  │ Medium slabs (50–100 atoms), 3×3×1 k-grid  ← DEFAULT
  8   │   7  │ Alternative to NCORE=7; try if Step 4 shows slow SCF
 14   │   4  │ Large slabs (> 100 atoms) with few k-points
 28   │   2  │ Very large systems only (> 200 atoms); rarely beneficial
```

> **Group server (even-core count)**: if your local workstation has, e.g., 32 cores,
> use NCORE = 4 (32/4 = 8) or NCORE = 8 (32/8 = 4).  Powers of 2 always divide evenly
> into a power-of-2 core count; choose NCORE ≈ √(NPROC) as a rule of thumb.
> For 56-core nodes, √56 ≈ 7.5 → round to the nearest valid divisor → **NCORE = 7**.

---

## Output Files and Next Steps

| File | Location | Used by |
|---|---|---|
| `CONTCAR` (primitive opt) | `work/Phase00/graphene_primitive/opt/` | Step 2 construction |
| `CONTCAR` (slab opt) | `work/Phase00/NiZn_NC/opt/` | Export as template |
| `NiZn-N-C.cif` | project root (version control) | Manual reference |
| `data/structures/graphene_template.json` | `data/structures/` | `build_slab.py` |

After exporting `graphene_template.json`, proceed to **Phase 01** (`phase01_slab.md`):

```bash
# Sample 25 initial combinations
python scripts/active_learning/sample_initial_combinations.py -n 25

# Build all sampled slabs from the template
python scripts/build/build_slab.py \
    --csv data/sampled_combinations.csv \
    --template data/structures/graphene_template.json \
    --outroot data/structures/slabs/
```
