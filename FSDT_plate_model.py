"""
Author: Aksel Gundersen
Project: Master thesis – Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design
Content: Builds and solves an FSDT finite element plate model with local load patches, and computes displacement, energy and ply stress quantities for post-processing or convergence studies.
"""

import numpy as np
import pyvista
import ufl
import csv
import os
import gmsh

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio
from dolfinx import plot

from CLT_FSDT import CLT_FSDT
from Applied_loads import LOAD_CASES
from postprocess import (
    compute_ply_surface_local_stresses,
    print_ply_surface_stress_ranges,
    compute_ply_max_abs,
    print_ply_max_abs,
    plot_ply_surface_stress,
)


# Laminate definition.
top = [np.pi/4, np.pi/2, 0]
bottom = [np.pi/2, np.pi/4]

A, B, D, As, h, ply_data = CLT_FSDT(
    bottom, top,
    return_total_thickness=True,
    return_ply_data=True
)


# Geometry, mesh and load parameters.
L = 450
W = 220

h=10
Pe = 2

hole_center = (70.1050, -40.5294)
hole_radius = 15.0 

patch_w = 40.0
patch_h = 27.0

patch_centers = {
    1: (-141.4076,  81.7512),
    2: ( 158.5924,  81.7512),
    3: (-158.5924, -81.7512),
    4: ( 141.4076, -81.7512),
}

A_patch = patch_w * patch_h

CASE = "longitudinal"       # Load case "longitudinal" or "lateral" (braking or cornering)
LoadFactor = 1.5

case_data = LOAD_CASES[CASE]

P = {
    i + 1: case_data["P"][i].copy() * LoadFactor
    for i in range(4)
}

M = {
    i + 1: case_data["M"][i, :2].copy() * LoadFactor
    for i in range(4)
}


def create_plate_mesh_with_hole_and_patches(comm, h):
    """
    Create a 2D Gmsh mesh for the plate, tie-rod hole and load patches.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI communicator used when converting the Gmsh model to a DOLFINx mesh.
    h : float
        Global characteristic mesh size.

    Returns
    -------
    tuple
        DOLFINx mesh, cell tags and facet tags.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model
    occ = model.occ
    model.add("plate_with_hole")

    if comm.rank == 0:
        plate = occ.addRectangle(-L/2, -W/2, 0.0, L, W)
        hole = occ.addDisk(hole_center[0], hole_center[1], 0.0, hole_radius, hole_radius)

        cut = occ.cut([(2, plate)], [(2, hole)], removeObject=True, removeTool=True)
        occ.synchronize()

        base_surfaces = cut[0]

        patch_entities = []
        for tag, (xc, yc) in patch_centers.items():
            r = occ.addRectangle(xc - patch_w/2, yc - patch_h/2, 0.0, patch_w, patch_h)
            patch_entities.append((2, r))

        occ.synchronize()
        occ.fragment(base_surfaces, patch_entities)
        occ.synchronize()

        all_surfaces = model.getEntities(dim=2)

        patch_surface_map = {}
        for tag, (xc, yc) in patch_centers.items():
            found = None
            for dim, s in all_surfaces:
                com = occ.getCenterOfMass(dim, s)
                if abs(com[0] - xc) < 1e-8 and abs(com[1] - yc) < 1e-8:
                    found = s
                    break
            if found is None:
                raise RuntimeError(f"Could not find surface for patch {tag}")
            patch_surface_map[tag] = found

        patch_surface_ids = set(patch_surface_map.values())
        plate_surfaces = [s for dim, s in all_surfaces if s not in patch_surface_ids]

        model.addPhysicalGroup(2, plate_surfaces, 100)
        model.setPhysicalName(2, 100, "Plate")

        for tag, s in patch_surface_map.items():
            model.addPhysicalGroup(2, [s], tag)
            model.setPhysicalName(2, tag, f"Patch{tag}")

        boundary_curves = model.getBoundary(
            [(2, s) for _, s in all_surfaces],
            oriented=False,
            recursive=False
        )

        outer_curves = []
        hole_curves = []

        for dim, c in boundary_curves:
            cx, cy, cz = occ.getCenterOfMass(dim, c)
            r = np.sqrt((cx - hole_center[0])**2 + (cy - hole_center[1])**2)
            if abs(r - hole_radius) < 1e-6:
                hole_curves.append(c)
            else:
                outer_curves.append(c)

        outer_curves = list(set(outer_curves))
        hole_curves = list(set(hole_curves))

        if outer_curves:
            model.addPhysicalGroup(1, outer_curves, 201)
            model.setPhysicalName(1, 201, "OuterBoundary")

        if hole_curves:
            model.addPhysicalGroup(1, hole_curves, 202)
            model.setPhysicalName(1, 202, "HoleBoundary")

        points = model.getEntities(0)
        model.mesh.setSize(points, h)

        h_patch = h / 4
        pad_x = 0.25 * patch_w
        pad_y = 0.25 * patch_h

        field_ids = []

        for tag, (xc, yc) in patch_centers.items():
            f = model.mesh.field.add("Box")
            model.mesh.field.setNumber(f, "VIn", h_patch)
            model.mesh.field.setNumber(f, "VOut", h)
            model.mesh.field.setNumber(f, "XMin", xc - patch_w / 2 - pad_x)
            model.mesh.field.setNumber(f, "XMax", xc + patch_w / 2 + pad_x)
            model.mesh.field.setNumber(f, "YMin", yc - patch_h / 2 - pad_y)
            model.mesh.field.setNumber(f, "YMax", yc + patch_h / 2 + pad_y)
            model.mesh.field.setNumber(f, "Thickness", 1.0)
            field_ids.append(f)

        if field_ids:
            if len(field_ids) == 1:
                model.mesh.field.setAsBackgroundMesh(field_ids[0])
            else:
                fmin = model.mesh.field.add("Min")
                model.mesh.field.setNumbers(fmin, "FieldsList", field_ids)
                model.mesh.field.setAsBackgroundMesh(fmin)

            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        model.mesh.generate(2)

    mesh_data = gmshio.model_to_mesh(model, comm, rank=0, gdim=2)

    domain = mesh_data.mesh
    cell_tags = mesh_data.cell_tags
    facet_tags = mesh_data.facet_tags

    gmsh.finalize()
    return domain, cell_tags, facet_tags


domain, cell_tags, facet_tags = create_plate_mesh_with_hole_and_patches(MPI.COMM_WORLD, h)


# # Optional mesh plot.
# def plot_plate_mesh_with_patches(domain, cell_tags):
#     """
#     Plot the generated mesh and physical cell tags.
#     """
#     if domain.comm.rank != 0:
#         return

#     topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

#     num_local_cells = domain.topology.index_map(domain.topology.dim).size_local
#     values = np.zeros(num_local_cells, dtype=np.int32)

#     if cell_tags is not None:
#         tag_indices = cell_tags.indices
#         tag_values = cell_tags.values
#         values[tag_indices] = tag_values

#     grid.cell_data["cell_tags"] = values

#     plotter = pyvista.Plotter()
#     plotter.add_mesh(
#         grid,
#         scalars="cell_tags",
#         show_edges=True,
#         scalar_bar_args={"title": "Cell tags"},
#         show_scalar_bar=False,
#     )
#     plotter.view_xy()
#     plotter.show()
    
# plot_plate_mesh_with_patches(domain, cell_tags)


# Function space: [ux, uy, uz, phix, phiy].
V = fem.functionspace(domain, ("Lagrange", Pe, (5,)))

dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


def clamped_boundary(x):
    """
    Identify the outer plate boundary for fully clamped support conditions.

    Parameters
    ----------
    x : numpy.ndarray
        Coordinate array.

    Returns
    -------
    numpy.ndarray
        Boolean mask for points on the clamped boundary.
    """
    return (
        np.isclose(x[0], -L / 2)
        | np.isclose(x[0],  L / 2)
        | np.isclose(x[1], -W / 2)
        | np.isclose(x[1],  W / 2)
    )


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.zeros(5, dtype=default_scalar_type)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(u_D, boundary_dofs, V)


# FSDT strain measures.
u = ufl.TrialFunction(V)
du = ufl.TestFunction(V)


def eps_2D_0(w):
    """
    Compute the membrane strain vector.

    Parameters
    ----------
    w : ufl.Argument or dolfinx.fem.Function
        Displacement and rotation field.

    Returns
    -------
    ufl.Vector
        Membrane strain vector.
    """
    ux, uy, uz, phix, phiy = ufl.split(w)
    return ufl.as_vector([
        ux.dx(0),
        uy.dx(1),
        ux.dx(1) + uy.dx(0)
    ])


def chi(w):
    """
    Compute the curvature vector.

    Parameters
    ----------
    w : ufl.Argument or dolfinx.fem.Function
        Displacement and rotation field.

    Returns
    -------
    ufl.Vector
        Curvature vector.
    """
    ux, uy, uz, phix, phiy = ufl.split(w)
    return ufl.as_vector([
        phix.dx(0),
        phiy.dx(1),
        phix.dx(1) + phiy.dx(0)
    ])


def gamma(w):
    """
    Compute the transverse shear strain vector.

    Parameters
    ----------
    w : ufl.Argument or dolfinx.fem.Function
        Displacement and rotation field.

    Returns
    -------
    ufl.Vector
        Transverse shear strain vector.
    """
    ux, uy, uz, phix, phiy = ufl.split(w)
    return ufl.as_vector([
        uz.dx(0) + phix,
        uz.dx(1) + phiy
    ])


A_ufl = ufl.as_matrix(A.tolist())
B_ufl = ufl.as_matrix(B.tolist())
D_ufl = ufl.as_matrix(D.tolist())
As_ufl = ufl.as_matrix(As.tolist())

e0_u = eps_2D_0(u)
k_u = chi(u)
g_u = gamma(u)

e0_du = eps_2D_0(du)
k_du = chi(du)
g_du = gamma(du)

dx_full = ufl.Measure("dx", domain=domain)
dx_shear = ufl.Measure("dx", domain=domain)
# dx_shear = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2})

a = (
    ufl.dot(e0_du, A_ufl * e0_u + B_ufl * k_u)
    + ufl.dot(k_du, B_ufl * e0_u + D_ufl * k_u)
) * dx_full + ufl.dot(g_du, As_ufl * g_u) * dx_shear

L_form = 0

for i in [1, 2, 3, 4]:
    p_i = fem.Constant(domain, P[i] / A_patch)
    m_i = fem.Constant(domain, M[i] / A_patch)  

    du_trans = ufl.as_vector([du[0], du[1], du[2]])
    du_rot   = ufl.as_vector([du[3], du[4]])

    L_form += ufl.dot(p_i, du_trans) * dx(i)
    L_form += ufl.dot(m_i, du_rot) * dx(i)

problem = LinearProblem(
    a,
    L_form,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="linear_plate",
)


# Solve and compute convergence quantities.
uh = problem.solve()
uh.name = "u"

num_cells = domain.topology.index_map(domain.topology.dim).size_global
ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
h_char = h


# Global displacement quantities.
u_components = ["ux", "uy", "uz", "phix", "phiy"]
comp_max = {}

for i, name in enumerate(u_components):
    fi = uh.sub(i).collapse()
    val_local = np.max(np.abs(fi.x.array))
    comp_max[name] = domain.comm.allreduce(val_local, op=MPI.MAX)

uz_max = comp_max["uz"]


# Strain energy.
e0_h = eps_2D_0(uh)
k_h = chi(uh)
g_h = gamma(uh)

energy_form = (
    0.5 * (
        ufl.dot(e0_h, A_ufl * e0_h + B_ufl * k_h)
        + ufl.dot(k_h, B_ufl * e0_h + D_ufl * k_h)
    ) * dx_full
    + 0.5 * ufl.dot(g_h, As_ufl * g_h) * dx_shear
)

energy_local = fem.assemble_scalar(fem.form(energy_form))
strain_energy = domain.comm.allreduce(energy_local, op=MPI.SUM)


# External work check.
work_form = 0
for i in [1, 2, 3, 4]:
    p_i = P[i] / A_patch
    m_i = M[i] / A_patch

    work_form += (
        uh[0] * p_i[0]
        + uh[1] * p_i[1]
        + uh[2] * p_i[2]
        + uh[3] * m_i[0]
        + uh[4] * m_i[1]
    ) * dx(i)

external_work_local = fem.assemble_scalar(fem.form(work_form))
external_work = domain.comm.allreduce(external_work_local, op=MPI.SUM)

rel_balance = np.nan
if abs(external_work) > 0.0:
    rel_balance = abs(external_work - 2.0 * strain_energy) / abs(external_work)


# Ply stress and failure quantities.
ply_surface_fields, V0 = compute_ply_surface_local_stresses(
    domain=domain,
    uh=uh,
    ply_data=ply_data,
    eps_2D_0=eps_2D_0,
    chi=chi,
    gamma=gamma,
)

ply_max = compute_ply_max_abs(ply_surface_fields)

tracked_components = [
    "sigma1",
    "sigma2",
    "tau12",
    "tau13",
    "tau23",
    "FI_TW",
    "FI_tau13",
    "FI_tau23",
    "FI_oop_shear",
]

global_max = {comp: 0.0 for comp in tracked_components}
critical_ply = {comp: None for comp in tracked_components}

for ply_id, comps in ply_max.items():
    for comp in tracked_components:
        if comp in comps and comps[comp] > global_max[comp]:
            global_max[comp] = comps[comp]
            critical_ply[comp] = ply_id


# Optional convergence printout.
# if domain.comm.rank == 0:
#     print()
#     print("Convergence-study quantities for this run:")
#     print(
#         f"{'h':>10s} {'Pe':>6s} {'cells':>12s} {'dofs':>12s} "
#         f"{'max|uz|':>16s} {'strain_energy':>16s} {'ext_work':>16s} {'rel_bal':>12s}"
#     )
#     print(
#         f"{h_char:10.3f} {Pe:6d} {num_cells:12d} {ndofs:12d} "
#         f"{uz_max:16.8e} {strain_energy:16.8e} {external_work:16.8e} {rel_balance:12.4e}"
#     )

#     print()
#     print("Max absolute value of each DOF component:")
#     for name in u_components:
#         print(f"  max|{name}| = {comp_max[name]:.8e}")

#     print()
#     print("Global maximum stress / failure metrics across all plies:")
#     for comp in tracked_components:
#         print(
#             f"  max|{comp}| = {global_max[comp]:.8e}   "
#             f"(critical ply = {critical_ply[comp]})"
#         )

#     print()
#     print("Per-ply maximum absolute local stress components and failure indices:")
#     for ply_id, comps in ply_max.items():
#         print(f"\nPly {ply_id}:")
#         for comp in tracked_components:
#             if comp in comps:
#                 print(f"  {comp:12s}: {comps[comp]:.8e}")

#     print()
#     print("Work/energy consistency check (linear static):")
#     print(f"  external_work         = {external_work:.8e}")
#     print(f"  2 * strain_energy     = {2.0 * strain_energy:.8e}")
#     print(f"  relative difference   = {rel_balance:.8e}")


# Optional CSV storage for convergence results.
# results_file = "convergence_results.csv"

# row = {
#     "h": h_char,
#     "Pe": Pe,
#     "cells": num_cells,
#     "dofs": ndofs,
#     "ux_max": comp_max["ux"],
#     "uy_max": comp_max["uy"],
#     "uz_max": comp_max["uz"],
#     "phix_max": comp_max["phix"],
#     "phiy_max": comp_max["phiy"],
#     "strain_energy": strain_energy,
#     "ext_work": external_work,
#     "rel_balance": rel_balance,
#     "max_sigma1": global_max["sigma1"],
#     "max_sigma2": global_max["sigma2"],
#     "max_tau12": global_max["tau12"],
#     "max_tau13": global_max["tau13"],
#     "max_tau23": global_max["tau23"],
#     "max_FI_TW": global_max["FI_TW"],
#     "max_FI_tau13": global_max["FI_tau13"],
#     "max_FI_tau23": global_max["FI_tau23"],
#     "max_FI_oop_shear": global_max["FI_oop_shear"],
#     "crit_ply_sigma1": critical_ply["sigma1"],
#     "crit_ply_sigma2": critical_ply["sigma2"],
#     "crit_ply_tau12": critical_ply["tau12"],
#     "crit_ply_tau13": critical_ply["tau13"],
#     "crit_ply_tau23": critical_ply["tau23"],
#     "crit_ply_FI_TW": critical_ply["FI_TW"],
#     "crit_ply_FI_tau13": critical_ply["FI_tau13"],
#     "crit_ply_FI_tau23": critical_ply["FI_tau23"],
#     "crit_ply_FI_oop_shear": critical_ply["FI_oop_shear"],
# }

# if domain.comm.rank == 0:
#     file_exists = os.path.isfile(results_file)

#     fieldnames = [
#         "h",
#         "Pe",
#         "cells",
#         "dofs",
#         "ux_max",
#         "uy_max",
#         "uz_max",
#         "phix_max",
#         "phiy_max",
#         "strain_energy",
#         "ext_work",
#         "rel_balance",
#         "max_sigma1",
#         "max_sigma2",
#         "max_tau12",
#         "max_tau13",
#         "max_tau23",
#         "max_FI_TW",
#         "max_FI_tau13",
#         "max_FI_tau23",
#         "max_FI_oop_shear",
#         "crit_ply_sigma1",
#         "crit_ply_sigma2",
#         "crit_ply_tau12",
#         "crit_ply_tau13",
#         "crit_ply_tau23",
#         "crit_ply_FI_TW",
#         "crit_ply_FI_tau13",
#         "crit_ply_FI_tau23",
#         "crit_ply_FI_oop_shear",
#     ]

#     with open(results_file, "a", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         if not file_exists:
#             writer.writeheader()
#         writer.writerow(row)

#     print()
#     print(f"Saved convergence results to {results_file}")


# Post-processing.
ply_surface_fields, V0 = compute_ply_surface_local_stresses(
    domain=domain,
    uh=uh,
    ply_data=ply_data,
    eps_2D_0=eps_2D_0,
    chi=chi,
    gamma=gamma,
)

# print_ply_surface_stress_ranges(domain, ply_data, ply_surface_fields)

# ply_max = compute_ply_max_abs(ply_surface_fields)
# print_ply_max_abs(domain, ply_max)

# Optional stress and failure plots.
# plot_ply_surface_stress(domain, ply_surface_fields, 5, "bot", "sigma1")
# plot_ply_surface_stress(domain, ply_surface_fields, 0, "top", "FI_TW")
# plot_ply_surface_stress(domain, ply_surface_fields, 3, "top", "tau23")


# def plot_uz(domain, uh):
#     """
#     Plot transverse displacement u_z.
#     """
#     if domain.comm.rank != 0:
#         return

#     uz = uh.sub(2).collapse()

#     V0 = fem.functionspace(domain, ("DG", 0))
#     uz_cell = fem.Function(V0)
#     uz_cell.interpolate(uz)

#     topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

#     num_local_cells = domain.topology.index_map(domain.topology.dim).size_local
#     name = "u_z"

#     grid.cell_data[name] = uz_cell.x.array[:num_local_cells]

#     plotter = pyvista.Plotter()
#     plotter.add_mesh(
#         grid,
#         scalars=name,
#         show_edges=False,
#         scalar_bar_args={
#             "title": name + "\n\n",
#             "vertical": True,
#             "position_x": 0.88,
#             "position_y": 0.1,
#             "width": 0.06,
#             "height": 0.8,
#             "fmt": "%.2f",
#             "label_font_size": 40,
#         },
#     )
#     plotter.show_axes()
#     plotter.view_xy()
#     plotter.show()
    
# plot_uz(domain, uh)