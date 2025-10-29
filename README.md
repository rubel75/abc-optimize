# abc-optimize
Automates lattice optimization by total energy minimization using VASP or WIEN2k. The script parses structural files, drives single-point or short relaxations across trial lattices, and uses SciPy to optimise lattice parameters and angles. Results are written to a CSV table for post-processing.

*Note: This is a working progress and not well documented or tested at the moment.*

Features:
* Works with VASP (POSCAR or CONTCAR) and WIEN2k (case.struct)
* Supports multiple symmetry modes and parameterisations (for example: cubic a, tetragonal a,c, hexagonal a,c, rhombohedral a,α, orthorhombic a,b,c)
* SciPy-based optimisation with bracketed scans and local refinement
* Reuses previous calculations when possible to save CPU time
* Produces data.csv with lattice, energy, pressure or stress metadata
* Optional Slurm integration via srun for multi-job sweeps

Requirements:
* Python 3.9 or newer
* Packages: numpy, scipy, numdifftools
* External codes: a licensed VASP or WIEN2k installation available on PATH or through environment variables
