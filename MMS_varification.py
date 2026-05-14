"""
Author: Aksel Gundersen
Project: Master thesis – Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design
Content: Runs a manufactured solution verification of the CLT/FSDT plate model and writes convergence errors to CSV.
"""

import numpy as np
import ufl
import gmsh
import csv

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem

from CLT_FSDT import CLT_FSDT


# Verification settings.
EXACT_SOLUTION_TYPE = "trig"

Pe = 2

h_sizes = [30, 20, 10, 5, 4, 3, 2, 1]

APPLY_EXACT_ON_ALL_BOUNDARIES = True

AMP_UX = 0.10
AMP_UY = -0.07
AMP_UZ = 1.20
AMP_PHIX = 0.03
AMP_PHIY = -0.02


# Laminate definition used in the verification model.
top = [np.pi / 4, 0, np.pi / 2, 0, np.pi / 4]
bottom = [np.pi / 2, np.pi / 4]

A, B, D, As, h_lam, ply_data = CLT_FSDT(
    bottom,
    top,
    return_total_thickness=True,
    return_ply_data=True,
)


# Plate geometry and load-patch locations.
L = 450.0
W = 220.0

hole_center = (70.1050, -40.5294)
hole_radius = 15.0

patch_w = 40.0
patch_h = 27.0

patch_centers = {
    1: (-141.4076, 81.7512),
    2: (158.5924, 81.7512),
    3: (-158.5924, -81.7512),
    4: (141.4076, -81.7512),
}


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
    ux = w[0]
    uy = w[1]
    return ufl.as_vector([
        ux.dx(0),
        uy.dx(1),
        ux.dx(1) + uy.dx(0),
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
    phix = w[3]
    phiy = w[4]
    return ufl.as_vector([
        phix.dx(0),
        phiy.dx(1),
        phix.dx(1) + phiy.dx(0),
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
    uz = w[2]
    phix = w[3]
    phiy = w[4]
    return ufl.as_vector([
        uz.dx(0) + phix,
        uz.dx(1) + phiy,
    ])


def build_exact_field_trig_ufl(domain):
    """
    Build the sinusoidal manufactured displacement field.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.

    Returns
    -------
    ufl.Vector
        Exact manufactured field.
    """
    x = ufl.SpatialCoordinate(domain)

    sx = ufl.sin(np.pi * (x[0] + L / 2.0) / L)
    sy = ufl.sin(np.pi * (x[1] + W / 2.0) / W)
    s = sx * sy

    return ufl.as_vector([
        AMP_UX * s,
        AMP_UY * s,
        AMP_UZ * s,
        AMP_PHIX * s,
        AMP_PHIY * s,
    ])


def build_exact_field_poly_ufl(domain):
    """
    Build the polynomial manufactured displacement field.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.

    Returns
    -------
    ufl.Vector
        Exact manufactured field.
    """
    x = ufl.SpatialCoordinate(domain)
    xh = x[0] / L
    yh = x[1] / W

    p_ux = 1.0 + xh + 2.0 * yh + xh * yh + xh**2
    p_uy = 1.0 - 2.0 * xh + yh + xh * yh + yh**2
    p_uz = 1.0 + xh - yh + xh**2 + yh**2 + xh * yh
    p_phix = 1.0 - xh + 2.0 * yh + xh**2 - xh * yh
    p_phiy = 1.0 + 2.0 * xh - yh + yh**2 + xh * yh

    return ufl.as_vector([
        AMP_UX * p_ux,
        AMP_UY * p_uy,
        AMP_UZ * p_uz,
        AMP_PHIX * p_phix,
        AMP_PHIY * p_phiy,
    ])


def build_exact_field_ufl(domain, solution_type):
    """
    Select the manufactured displacement field.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.
    solution_type : str
        Exact field type, either ``"trig"`` or ``"poly"``.

    Returns
    -------
    ufl.Vector
        Exact manufactured field.
    """
    solution_type = solution_type.lower()

    if solution_type == "trig":
        return build_exact_field_trig_ufl(domain)
    if solution_type == "poly":
        return build_exact_field_poly_ufl(domain)

    raise ValueError(
        f"Unknown EXACT_SOLUTION_TYPE='{solution_type}'. "
        "Use 'trig' or 'poly'."
    )


def create_plate_mesh_with_hole_and_patches(comm, h_mesh):
    """
    Create the plate mesh with tie-rod hole and load-patch subdomains.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI communicator used for mesh conversion.
    h_mesh : float
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
    model.add("mms_plate_with_hole_and_patches")

    if comm.rank == 0:
        plate = occ.addRectangle(-L / 2.0, -W / 2.0, 0.0, L, W)
        hole = occ.addDisk(hole_center[0], hole_center[1], 0.0, hole_radius, hole_radius)

        cut = occ.cut([(2, plate)], [(2, hole)], removeObject=True, removeTool=True)
        occ.synchronize()

        base_surfaces = cut[0]

        patch_entities = []
        for tag, (xc, yc) in patch_centers.items():
            r = occ.addRectangle(xc - patch_w / 2.0, yc - patch_h / 2.0, 0.0, patch_w, patch_h)
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
            r = np.sqrt((cx - hole_center[0]) ** 2 + (cy - hole_center[1]) ** 2)
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
        model.mesh.setSize(points, h_mesh)

        h_patch = h_mesh / 4.0
        pad_x = 0.25 * patch_w
        pad_y = 0.25 * patch_h

        field_ids = []

        for tag, (xc, yc) in patch_centers.items():
            f = model.mesh.field.add("Box")
            model.mesh.field.setNumber(f, "VIn", h_patch)
            model.mesh.field.setNumber(f, "VOut", h_mesh)
            model.mesh.field.setNumber(f, "XMin", xc - patch_w / 2.0 - pad_x)
            model.mesh.field.setNumber(f, "XMax", xc + patch_w / 2.0 + pad_x)
            model.mesh.field.setNumber(f, "YMin", yc - patch_h / 2.0 - pad_y)
            model.mesh.field.setNumber(f, "YMax", yc + patch_h / 2.0 + pad_y)
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


def interpolate_exact_to_space(V_target, u_exact_ufl):
    """
    Interpolate the exact field into a finite element space.

    Parameters
    ----------
    V_target : dolfinx.fem.FunctionSpace
        Target function space.
    u_exact_ufl : ufl.Vector
        Exact field expression.

    Returns
    -------
    dolfinx.fem.Function
        Interpolated exact field.
    """
    u_ex = fem.Function(V_target)
    expr = fem.Expression(u_exact_ufl, V_target.element.interpolation_points)
    u_ex.interpolate(expr)
    return u_ex


def build_problem(h_mesh, solution_type):
    """
    Build the manufactured solution verification problem.

    Parameters
    ----------
    h_mesh : float
        Global characteristic mesh size.
    solution_type : str
        Exact field type, either ``"trig"`` or ``"poly"``.

    Returns
    -------
    tuple
        Domain, function space, linear problem and exact field.
    """
    domain, cell_tags, facet_tags = create_plate_mesh_with_hole_and_patches(
        MPI.COMM_WORLD, h_mesh
    )

    V = fem.functionspace(domain, ("Lagrange", Pe, (5,)))

    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    u_exact = build_exact_field_ufl(domain, solution_type)

    fdim = domain.topology.dim - 1

    if APPLY_EXACT_ON_ALL_BOUNDARIES:
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
    else:
        def outer_boundary(x):
            """
            Identify the outer plate boundary.

            Parameters
            ----------
            x : numpy.ndarray
                Coordinate array.

            Returns
            -------
            numpy.ndarray
                Boolean mask for points on the outer boundary.
            """
            return (
                np.isclose(x[0], -L / 2.0)
                | np.isclose(x[0], L / 2.0)
                | np.isclose(x[1], -W / 2.0)
                | np.isclose(x[1], W / 2.0)
            )

        boundary_facets = mesh.locate_entities_boundary(domain, fdim, outer_boundary)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = interpolate_exact_to_space(V, u_exact)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    u = ufl.TrialFunction(V)
    du = ufl.TestFunction(V)

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

    # Use the same shear integration rule here as in the production model.
    # dx_shear = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 1})

    a = (
        ufl.dot(e0_du, A_ufl * e0_u + B_ufl * k_u)
        + ufl.dot(k_du, B_ufl * e0_u + D_ufl * k_u)
    ) * dx_full + ufl.dot(g_du, As_ufl * g_u) * dx_shear

    e0_ex = eps_2D_0(u_exact)
    k_ex = chi(u_exact)
    g_ex = gamma(u_exact)

    L_form = (
        ufl.dot(e0_du, A_ufl * e0_ex + B_ufl * k_ex)
        + ufl.dot(k_du, B_ufl * e0_ex + D_ufl * k_ex)
    ) * dx_full + ufl.dot(g_du, As_ufl * g_ex) * dx_shear

    problem = LinearProblem(
        a,
        L_form,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="linear_plate_mms",
    )

    return domain, V, problem, u_exact


def compute_l2_error(domain, uh, u_exact_ufl):
    """
    Compute the total L2 error of the numerical solution.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.
    uh : dolfinx.fem.Function
        Numerical solution.
    u_exact_ufl : ufl.Vector
        Exact manufactured field.

    Returns
    -------
    float
        Total L2 error.
    """
    p_err = max(Pe + 3, 5)
    V_err = fem.functionspace(domain, ("Lagrange", p_err, (5,)))
    dx_err = ufl.Measure("dx", domain=domain)

    uh_err = fem.Function(V_err)
    uh_err.interpolate(uh)

    uex_err = interpolate_exact_to_space(V_err, u_exact_ufl)

    e = fem.Function(V_err)
    e.x.array[:] = uh_err.x.array - uex_err.x.array

    local = fem.assemble_scalar(fem.form(ufl.inner(e, e) * dx_err))
    global_val = domain.comm.allreduce(local, op=MPI.SUM)
    return np.sqrt(global_val)


def compute_component_l2_errors(domain, uh, u_exact_ufl):
    """
    Compute component-wise L2 errors.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.
    uh : dolfinx.fem.Function
        Numerical solution.
    u_exact_ufl : ufl.Vector
        Exact manufactured field.

    Returns
    -------
    dict
        L2 error for each displacement and rotation component.
    """
    p_err = max(Pe + 3, 5)
    V_err = fem.functionspace(domain, ("Lagrange", p_err, (5,)))
    dx_err = ufl.Measure("dx", domain=domain)

    uh_err = fem.Function(V_err)
    uh_err.interpolate(uh)

    uex_err = interpolate_exact_to_space(V_err, u_exact_ufl)

    e = fem.Function(V_err)
    e.x.array[:] = uh_err.x.array - uex_err.x.array

    names = ["ux", "uy", "uz", "phix", "phiy"]
    out = {}

    for i, name in enumerate(names):
        local = fem.assemble_scalar(fem.form((e[i] ** 2) * dx_err))
        out[name] = np.sqrt(domain.comm.allreduce(local, op=MPI.SUM))

    return out


def compute_discrete_inf_error(uh, V, u_exact_ufl):
    """
    Compute the discrete infinity norm of the error.

    Parameters
    ----------
    uh : dolfinx.fem.Function
        Numerical solution.
    V : dolfinx.fem.FunctionSpace
        Function space for the numerical solution.
    u_exact_ufl : ufl.Vector
        Exact manufactured field.

    Returns
    -------
    float
        Discrete infinity error.
    """
    u_exact_V = interpolate_exact_to_space(V, u_exact_ufl)
    e = fem.Function(V)
    e.x.array[:] = uh.x.array - u_exact_V.x.array
    return np.linalg.norm(e.x.array, ord=np.inf)


def convergence_rate(err_coarse, err_fine, h_coarse, h_fine):
    """
    Compute the observed convergence rate between two mesh sizes.

    Parameters
    ----------
    err_coarse : float
        Error on the coarser mesh.
    err_fine : float
        Error on the finer mesh.
    h_coarse : float
        Coarser mesh size.
    h_fine : float
        Finer mesh size.

    Returns
    -------
    float
        Observed convergence rate.
    """
    return np.log(err_coarse / err_fine) / np.log(h_coarse / h_fine)


def main():
    """
    Run the convergence study and write results to CSV.

    Returns
    -------
    None
        Prints convergence results and writes ``mms_results.csv``.
    """
    results = []

    if MPI.COMM_WORLD.rank == 0:
        import csv

        print("=" * 88)
        print("Manufactured solution verification for CLT/FSDT plate model")
        print("=" * 88)
        print(f"Exact solution type              : {EXACT_SOLUTION_TYPE}")
        print(f"Polynomial degree Pe            : {Pe}")
        print(f"Laminate total thickness        : {h_lam}")
        print(f"Apply exact BC on all boundaries: {APPLY_EXACT_ON_ALL_BOUNDARIES}")
        print(f"Hole center                     : {hole_center}")
        print(f"Hole radius                     : {hole_radius}")
        print("-" * 88)

        with open("mms_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "h", "cells", "dofs", "L2", "rate", "disc_inf",
                "ux", "uy", "uz", "phix", "phiy"
            ])

    prev_row = None

    for h_mesh in h_sizes:
        domain, V, problem, u_exact = build_problem(h_mesh, EXACT_SOLUTION_TYPE)
        uh = problem.solve()
        uh.name = "u"

        l2_error = compute_l2_error(domain, uh, u_exact)
        comp_errors = compute_component_l2_errors(domain, uh, u_exact)
        disc_inf_error = compute_discrete_inf_error(uh, V, u_exact)

        num_cells = domain.topology.index_map(domain.topology.dim).size_global
        ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

        row = {
            "h": h_mesh,
            "cells": num_cells,
            "dofs": ndofs,
            "L2": l2_error,
            "disc_inf": disc_inf_error,
            **comp_errors,
        }

        results.append(row)

        if domain.comm.rank == 0:
            print(f"Mesh h = {h_mesh:8.3f} | cells = {num_cells:8d} | dofs = {ndofs:8d}")
            print(f"  L2 error                    = {l2_error:.8e}")
            print(f"  Discrete infinity error     = {disc_inf_error:.8e}")
            for name in ["ux", "uy", "uz", "phix", "phiy"]:
                print(f"  {name:>4s} component L2 error      = {comp_errors[name]:.8e}")

            if prev_row is None:
                rate_val = None
                rate_str = "-"
            else:
                rate_val = convergence_rate(prev_row["L2"], row["L2"], prev_row["h"], row["h"])
                rate_str = f"{rate_val:.4f}"

            print(f"  Convergence rate            = {rate_str}")
            print("-" * 88)

            import csv
            with open("mms_results.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    row["h"],
                    row["cells"],
                    row["dofs"],
                    row["L2"],
                    rate_val,
                    row["disc_inf"],
                    row["ux"],
                    row["uy"],
                    row["uz"],
                    row["phix"],
                    row["phiy"],
                ])
                f.flush()

        prev_row = row

    if MPI.COMM_WORLD.rank == 0:
        print("Convergence summary")
        print(
            f"{'h':>12s} {'cells':>12s} {'dofs':>12s} "
            f"{'L2 error':>18s} {'rate':>10s} {'disc_inf':>18s}"
        )

        prev = None
        for row in results:
            if prev is None:
                rate_str = "-"
            else:
                rate = convergence_rate(prev["L2"], row["L2"], prev["h"], row["h"])
                rate_str = f"{rate:.4f}"

            print(
                f"{row['h']:12.4f} {row['cells']:12d} {row['dofs']:12d} "
                f"{row['L2']:18.8e} {rate_str:>10s} {row['disc_inf']:18.8e}"
            )
            prev = row

        print("Results written incrementally to mms_results.csv")


if __name__ == "__main__":
    main()