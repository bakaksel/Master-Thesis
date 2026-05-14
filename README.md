# Master Thesis – Composite Sandwich FSDT Analysis

Author: Aksel Gundersen  
Project: *Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design*

## Overview

This repository contains Python code developed for the structural analysis of composite laminated sandwich structures in a Formula Student chassis. The code focuses on the front suspension region of a composite sandwich monocoque and was developed as part of a master thesis.

The framework combines simplified vehicle load calculations, Classical Laminate Theory (CLT), First-Order Shear Deformation Theory (FSDT), finite element analysis, ply-level stress recovery and numerical verification.

The code is intended as a starting point for future Formula Student teams that want to evaluate, modify or build further on composite sandwich chassis analysis.

The model should be treated as an engineering support tool. It does not replace physical testing, SES documentation or final design validation.

## Repository structure

```text
.
├── Applied_loads.py
├── CLT_FSDT.py
├── FSDT_plate_model.py
├── MMS_plot
├── MMS_verification.py
├── README.md
├── Tsai_Pagano_plot.py
├── plot_convergence.py
└── postprocess.py
```

## File descriptions

### `Applied_loads.py`

Calculates representative suspension load cases from simplified vehicle dynamics.

The file includes:

- vehicle and suspension parameters
- longitudinal load case
- lateral load case
- force vectors for the load patches
- moment contributions used by the finite element model

The available load cases are stored in:

```python
LOAD_CASES["longitudinal"]
LOAD_CASES["lateral"]
```

These are imported and used by `FSDT_plate_model.py`.

---

### `CLT_FSDT.py`

Defines the laminate and core material properties and calculates the laminate stiffness matrices.

The file includes:

- lamina stiffness properties
- core shear properties
- ply and core strength values
- Tsai-Pagano based stiffness transformation
- CLT/FSDT stiffness matrix assembly
- layer-wise ply data used for stress recovery

The main function is:

```python
CLT_FSDT(bottom_skin, top_skin)
```

The function can return:

```text
A   extensional stiffness matrix
B   extension-bending coupling stiffness matrix
D   bending stiffness matrix
As  transverse shear stiffness matrix
h   total laminate thickness
ply_data  layer-wise data for post-processing
```

Ply angles are defined in radians.

Example:

```python
import numpy as np
from CLT_FSDT import CLT_FSDT

top = [np.pi / 4, np.pi / 2, 0]
bottom = [np.pi / 2, np.pi / 4]

A, B, D, As, h, ply_data = CLT_FSDT(
    bottom,
    top,
    return_total_thickness=True,
    return_ply_data=True,
)
```

---

### `FSDT_plate_model.py`

This is the main finite element analysis file.

The file:

- defines the laminate layup
- creates the plate geometry and tie-rod hole
- creates local suspension load patches
- generates the mesh using Gmsh
- sets up the FSDT finite element model in DOLFINx
- applies longitudinal or lateral load cases
- solves the linear static problem
- computes displacement, strain energy and external work
- computes ply stresses and failure indices through `postprocess.py`

The model uses five degrees of freedom per node:

```text
ux    in-plane displacement in x-direction
uy    in-plane displacement in y-direction
uz    transverse displacement
phix  rotation variable
phiy  rotation variable
```

Important user settings are located near the top of the file:

```python
PLOT_MESH = False
PRINT_CONVERGENCE_RESULTS = False
SAVE_CONVERGENCE_RESULTS = False
PRINT_PLY_STRESS_RANGES = False
PRINT_PLY_MAX_ABS = False
PLOT_SELECTED_PLY_STRESSES = False
PLOT_TRANSVERSE_DISPLACEMENT = False
```

Set these to `True` to activate optional output.

The load case is selected with:

```python
CASE = "longitudinal"
```

or:

```python
CASE = "lateral"
```

The load factor is controlled by:

```python
LoadFactor = 1.5
```

The mesh size and element order are controlled by:

```python
h = 10
Pe = 2
```

Run the main analysis with:

```bash
python FSDT_plate_model.py
```

or with MPI:

```bash
mpirun -n 1 python FSDT_plate_model.py
```

---

### `postprocess.py`

Contains post-processing functions used by the main FSDT model.

The file computes:

- local ply stresses at top and bottom ply surfaces
- stresses in local material coordinates
- Tsai-Wu failure index
- out-of-plane shear failure indices
- maximum absolute stress and failure-index values
- optional PyVista plots of selected stress fields

Available stress and failure components include:

```text
sigma1
sigma2
tau12
tau13
tau23
FI_TW
FI_tau13
FI_tau23
FI_oop_shear
```

Ply numbering is reversed in the output. Ply `0` is the topmost layer.

Example use inside `FSDT_plate_model.py`:

```python
ply_surface_fields, V0 = compute_ply_surface_local_stresses(
    domain=domain,
    uh=uh,
    ply_data=ply_data,
    eps_2D_0=eps_2D_0,
    chi=chi,
    gamma=gamma,
)
```

A selected stress or failure-index field can be plotted with:

```python
plot_ply_surface_stress(domain, ply_surface_fields, 5, "bot", "sigma1")
```

---

### `MMS_verification.py`

Runs a manufactured solution verification of the FSDT finite element formulation.

The file:

- defines an exact manufactured displacement field
- applies exact boundary conditions
- constructs the matching weak-form right-hand side
- solves the model for several mesh sizes
- calculates L2 errors and convergence rates
- writes verification results to CSV

Important settings:

```python
EXACT_SOLUTION_TYPE = "trig"
Pe = 2
h_sizes = [30, 20, 10]
APPLY_EXACT_ON_ALL_BOUNDARIES = True
```

Available manufactured solution types:

```text
trig
poly
```

Run with:

```bash
python MMS_verification.py
```

or with MPI:

```bash
mpirun -n 1 python MMS_verification.py
```

This file is used for code verification, not for design evaluation.

---

### `MMS_plot`

Plots convergence results from the manufactured solution verification.

The file reads MMS result CSV files and plots log-log convergence curves with reference slopes.

Run with:

```bash
python MMS_plot
```

If the file is renamed to `MMS_plot.py`, run:

```bash
python MMS_plot.py
```

---

### `Tsai_Pagano_plot.py`

Plots selected laminate stiffness terms as a function of ply angle.

This file is useful for studying how fibre orientation affects laminate stiffness and coupling behaviour.

Run with:

```bash
python Tsai_Pagano_plot.py
```

---

### `plot_convergence.py`

Plots convergence results generated by `FSDT_plate_model.py`.

Before using this script, set the following in `FSDT_plate_model.py`:

```python
SAVE_CONVERGENCE_RESULTS = True
```

Then run the main model for the desired mesh sizes and polynomial degrees.

The results are written to:

```text
convergence_results.csv
```

Run the plotting script with:

```bash
python plot_convergence.py
```

## Dependencies

The code requires:

```text
numpy
pandas
matplotlib
gmsh
mpi4py
dolfinx
ufl
pyvista
```

DOLFINx/FEniCSx must be installed separately using a suitable Conda, Docker or system installation.

Some plotting scripts use LaTeX rendering in Matplotlib. If LaTeX is not installed, change:

```python
"text.usetex": True
```

to:

```python
"text.usetex": False
```

in the relevant plotting file.

## Units

The code uses a millimetre-based unit system.

```text
Length:      mm
Force:       N
Stress:      MPa = N/mm²
A matrix:    MPa mm
B matrix:    MPa mm²
D matrix:    MPa mm³
```

All input values should use consistent units.

## Typical workflow

1. Update material and strength data in `CLT_FSDT.py`.
2. Update vehicle and suspension parameters in `Applied_loads.py`.
3. Define the laminate layup in `FSDT_plate_model.py`.
4. Select load case, mesh size and element order.
5. Run `FSDT_plate_model.py`.
6. Activate optional post-processing if needed.
7. Save convergence results and plot them using `plot_convergence.py`.
8. Use `MMS_verification.py` and `MMS_plot` to check the numerical implementation.

## Running the main model

Open `FSDT_plate_model.py` and select the desired load case:

```python
CASE = "longitudinal"
```

or:

```python
CASE = "lateral"
```

Select mesh size and polynomial degree:

```python
h = 10
Pe = 2
```

Then run:

```bash
python FSDT_plate_model.py
```

The model will solve the FSDT plate problem and compute the main displacement, energy and ply stress quantities.

## Running a convergence study

To run a convergence study:

1. Set this in `FSDT_plate_model.py`:

```python
SAVE_CONVERGENCE_RESULTS = True
```

2. Run the model for one mesh size:

```python
h = 20
Pe = 2
```

3. Change the mesh size and run again:

```python
h = 10
Pe = 2
```

4. Repeat for the desired mesh sizes and element orders.

The results are appended to:

```text
convergence_results.csv
```

Then plot the results with:

```bash
python plot_convergence.py
```

If starting a new convergence study, delete or rename the old CSV file first to avoid mixing results.

## Running MMS verification

To verify the finite element formulation, open `MMS_verification.py` and select:

```python
EXACT_SOLUTION_TYPE = "trig"
Pe = 2
h_sizes = [30, 20, 10]
```

Then run:

```bash
python MMS_verification.py
```

The script computes numerical errors and convergence rates.

The results can be plotted using:

```bash
python MMS_plot
```

or, if renamed:

```bash
python MMS_plot.py
```

## Output files

The scripts may generate CSV files depending on which options are activated.

### `convergence_results.csv`

Generated by `FSDT_plate_model.py` when:

```python
SAVE_CONVERGENCE_RESULTS = True
```

Typical contents:

```text
h
Pe
cells
dofs
ux_max
uy_max
uz_max
phix_max
phiy_max
strain_energy
ext_work
rel_balance
max_sigma1
max_sigma2
max_tau12
max_tau13
max_tau23
max_FI_TW
max_FI_tau13
max_FI_tau23
max_FI_oop_shear
critical ply values
```

### `mms_results.csv`

Generated by `MMS_verification.py`.

Typical contents:

```text
h
cells
dofs
L2
rate
disc_inf
ux
uy
uz
phix
phiy
```

## Notes for future Formula Student teams

This repository is intended as a foundation for further development. Future teams may extend the code by:

- adding new suspension load cases
- updating vehicle parameters for a new car
- testing different laminate layups
- changing core thickness and material data
- improving hardpoint and insert modelling
- automating convergence studies
- comparing numerical results with physical test data
- extending the failure criteria
- exporting results to other visualisation tools
- improving the representation of boundary conditions
- linking the model to CAD-derived geometry

The model assumptions should always be reviewed before using the results for design decisions.

## Model assumptions and limitations

The current model is based on the following assumptions:

- the analysed region is represented as a laminated sandwich plate
- the plate kinematics are based on First-Order Shear Deformation Theory
- material behaviour is linear elastic
- small displacements and small rotations are assumed
- loads are derived from simplified vehicle dynamics
- loads are applied as distributed loads over rectangular patches
- the outer boundary is modelled as fully clamped in the main analysis
- the tie-rod hole is included in the geometry

The current model does not include:

- nonlinear material behaviour
- progressive damage
- detailed insert modelling
- adhesive failure
- local core crushing
- full monocoque stiffness analysis
- experimental validation inside the codebase

The results should therefore be interpreted within the assumptions of the model.

## Citation

If this code is used in future work, cite the associated master thesis:

Aksel Gundersen, *Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design*.
