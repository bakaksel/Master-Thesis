"""
Author: Aksel Gundersen
Project: Master thesis – Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design
Content: Computes, summarises and plots ply-level stresses and failure indices from the FSDT plate model.
"""

import numpy as np
import pyvista
import ufl

from dolfinx import fem, plot


def compute_ply_surface_local_stresses(domain, uh, ply_data, eps_2D_0, chi, gamma):
    """
    Compute local stress components and failure indices at ply surfaces.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.
    uh : dolfinx.fem.Function
        Solved displacement and rotation field.
    ply_data : list of dict
        Layer-wise geometry, stiffness and strength data.
    eps_2D_0, chi, gamma : callable
        Functions returning membrane, curvature and shear strain measures.

    Returns
    -------
    tuple
        Dictionary of projected ply surface fields and the DG0 function space.
    """
    e0_h = eps_2D_0(uh)
    k_h = chi(uh)
    g_h = gamma(uh)

    V0 = fem.functionspace(domain, ("DG", 0))
    X0 = V0.element.interpolation_points

    ply_surface_fields = {}

    def project_expr_to_dg0(expr, name):
        """
        Project a UFL expression to a cell-wise constant field.
        """
        f = fem.Function(V0, name=name)
        expr_obj = fem.Expression(expr, X0, comm=domain.comm)
        f.interpolate(expr_obj)
        return f

    ply_data_reordered = list(reversed(ply_data))

    for ply_id, ply in enumerate(ply_data_reordered):
        Qbar_ufl = ufl.as_matrix(np.array(ply["Qbar"], dtype=float).tolist())
        Qsbar_ufl = ufl.as_matrix(np.array(ply["Qsbar"], dtype=float).tolist())

        z_bot = float(ply["z_bot"])
        z_top = float(ply["z_top"])

        theta = 0.0 if ply["theta"] is None else float(ply["theta"])
        m = np.cos(theta)
        n = np.sin(theta)

        strength = ply.get("strength", {})
        Xt = strength.get("Xt", None)
        Xc = strength.get("Xc", None)
        Yt = strength.get("Yt", None)
        Yc = strength.get("Yc", None)
        S12 = strength.get("S12", None)
        S13 = strength.get("S13", None)
        S23 = strength.get("S23", None)

        eps_bot = e0_h + z_bot * k_h
        eps_top = e0_h + z_top * k_h

        sig_bot_g = Qbar_ufl * eps_bot
        sig_top_g = Qbar_ufl * eps_top
        tau_g = Qsbar_ufl * g_h

        sx_b = sig_bot_g[0]
        sy_b = sig_bot_g[1]
        txy_b = sig_bot_g[2]

        sx_t = sig_top_g[0]
        sy_t = sig_top_g[1]
        txy_t = sig_top_g[2]

        txz = tau_g[0]
        tyz = tau_g[1]

        sigma1_bot = m * m * sx_b + n * n * sy_b + 2.0 * m * n * txy_b
        sigma2_bot = n * n * sx_b + m * m * sy_b - 2.0 * m * n * txy_b
        tau12_bot = -m * n * sx_b + m * n * sy_b + (m * m - n * n) * txy_b

        sigma1_top = m * m * sx_t + n * n * sy_t + 2.0 * m * n * txy_t
        sigma2_top = n * n * sx_t + m * m * sy_t - 2.0 * m * n * txy_t
        tau12_top = -m * n * sx_t + m * n * sy_t + (m * m - n * n) * txy_t

        tau13 = m * txz + n * tyz
        tau23 = -n * txz + m * tyz

        if None not in (Xt, Xc, Yt, Yc, S12):
            F1 = 1.0 / Xt - 1.0 / Xc
            F2 = 1.0 / Yt - 1.0 / Yc
            F11 = 1.0 / (Xt * Xc)
            F22 = 1.0 / (Yt * Yc)
            F66 = 1.0 / (S12 * S12)

            F12 = -0.5 * np.sqrt(F11 * F22)

            FI_TW_bot = (
                F1 * sigma1_bot
                + F2 * sigma2_bot
                + F11 * sigma1_bot * sigma1_bot
                + F22 * sigma2_bot * sigma2_bot
                + 2.0 * F12 * sigma1_bot * sigma2_bot
                + F66 * tau12_bot * tau12_bot
            )

            FI_TW_top = (
                F1 * sigma1_top
                + F2 * sigma2_top
                + F11 * sigma1_top * sigma1_top
                + F22 * sigma2_top * sigma2_top
                + 2.0 * F12 * sigma1_top * sigma2_top
                + F66 * tau12_top * tau12_top
            )
        else:
            FI_TW_bot = ufl.as_ufl(0.0)
            FI_TW_top = ufl.as_ufl(0.0)

        if S13 is not None:
            FI_tau13 = abs(tau13) / S13
        else:
            FI_tau13 = ufl.as_ufl(0.0)

        if S23 is not None:
            FI_tau23 = abs(tau23) / S23
        else:
            FI_tau23 = ufl.as_ufl(0.0)

        FI_oop_shear = ufl.max_value(FI_tau13, FI_tau23)

        ply_surface_fields[(ply_id, "bot", "sigma1")] = project_expr_to_dg0(
            sigma1_bot, f"ply{ply_id}_bot_sigma1"
        )
        ply_surface_fields[(ply_id, "bot", "sigma2")] = project_expr_to_dg0(
            sigma2_bot, f"ply{ply_id}_bot_sigma2"
        )
        ply_surface_fields[(ply_id, "bot", "tau12")] = project_expr_to_dg0(
            tau12_bot, f"ply{ply_id}_bot_tau12"
        )
        ply_surface_fields[(ply_id, "bot", "tau13")] = project_expr_to_dg0(
            tau13, f"ply{ply_id}_bot_tau13"
        )
        ply_surface_fields[(ply_id, "bot", "tau23")] = project_expr_to_dg0(
            tau23, f"ply{ply_id}_bot_tau23"
        )
        ply_surface_fields[(ply_id, "bot", "FI_TW")] = project_expr_to_dg0(
            FI_TW_bot, f"ply{ply_id}_bot_FI_TW"
        )
        ply_surface_fields[(ply_id, "bot", "FI_tau13")] = project_expr_to_dg0(
            FI_tau13, f"ply{ply_id}_bot_FI_tau13"
        )
        ply_surface_fields[(ply_id, "bot", "FI_tau23")] = project_expr_to_dg0(
            FI_tau23, f"ply{ply_id}_bot_FI_tau23"
        )
        ply_surface_fields[(ply_id, "bot", "FI_oop_shear")] = project_expr_to_dg0(
            FI_oop_shear, f"ply{ply_id}_bot_FI_oop_shear"
        )

        ply_surface_fields[(ply_id, "top", "sigma1")] = project_expr_to_dg0(
            sigma1_top, f"ply{ply_id}_top_sigma1"
        )
        ply_surface_fields[(ply_id, "top", "sigma2")] = project_expr_to_dg0(
            sigma2_top, f"ply{ply_id}_top_sigma2"
        )
        ply_surface_fields[(ply_id, "top", "tau12")] = project_expr_to_dg0(
            tau12_top, f"ply{ply_id}_top_tau12"
        )
        ply_surface_fields[(ply_id, "top", "tau13")] = project_expr_to_dg0(
            tau13, f"ply{ply_id}_top_tau13"
        )
        ply_surface_fields[(ply_id, "top", "tau23")] = project_expr_to_dg0(
            tau23, f"ply{ply_id}_top_tau23"
        )
        ply_surface_fields[(ply_id, "top", "FI_TW")] = project_expr_to_dg0(
            FI_TW_top, f"ply{ply_id}_top_FI_TW"
        )
        ply_surface_fields[(ply_id, "top", "FI_tau13")] = project_expr_to_dg0(
            FI_tau13, f"ply{ply_id}_top_FI_tau13"
        )
        ply_surface_fields[(ply_id, "top", "FI_tau23")] = project_expr_to_dg0(
            FI_tau23, f"ply{ply_id}_top_FI_tau23"
        )
        ply_surface_fields[(ply_id, "top", "FI_oop_shear")] = project_expr_to_dg0(
            FI_oop_shear, f"ply{ply_id}_top_FI_oop_shear"
        )

    return ply_surface_fields, V0


def print_ply_surface_stress_ranges(domain, ply_data, ply_surface_fields):
    """
    Print minimum and maximum ply surface stresses and failure indices.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.
    ply_data : list of dict
        Layer-wise ply data.
    ply_surface_fields : dict
        Projected ply surface stress and failure-index fields.

    Returns
    -------
    None
        Prints values on MPI rank 0.
    """
    if domain.comm.rank != 0:
        return

    ply_data_reordered = list(reversed(ply_data))

    components = [
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

    print("\nLayer-wise local stress components and failure indices at ply surfaces:")
    for ply_id, ply in enumerate(ply_data_reordered):
        print(
            f"\nPly {ply_id}: name={ply['name']}, theta={ply['theta']}, "
            f"z_bot={ply['z_bot']:.6f}, z_top={ply['z_top']:.6f}"
        )
        for side in ["bot", "top"]:
            print(f"  {side}:")
            for comp in components:
                arr = ply_surface_fields[(ply_id, side, comp)].x.array
                print(f"    {comp:12s}: min = {arr.min():10.4f}, max = {arr.max():10.4f}")


def compute_ply_max_abs(ply_surface_fields):
    """
    Compute maximum absolute values for each ply and component.

    Parameters
    ----------
    ply_surface_fields : dict
        Projected ply surface stress and failure-index fields.

    Returns
    -------
    dict
        Maximum absolute value for each ply and component.
    """
    ply_max = {}

    for (k, side, comp), field in ply_surface_fields.items():
        val = np.max(np.abs(field.x.array))
        ply_max.setdefault(k, {})[comp] = max(
            ply_max.get(k, {}).get(comp, 0.0), val
        )

    return ply_max


def print_ply_max_abs(domain, ply_max):
    """
    Print maximum absolute stress and failure-index values for each ply.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.
    ply_max : dict
        Maximum absolute value for each ply and component.

    Returns
    -------
    None
        Prints values on MPI rank 0.
    """
    if domain.comm.rank != 0:
        return

    print("\nPer-ply maximum absolute local stress components and failure indices:")
    for k, comps in ply_max.items():
        print(f"\nPly {k}:")
        for c, v in comps.items():
            print(f"  {c:12s}: {v:10.4f}")


def plot_ply_surface_stress(domain, ply_surface_fields, ply_id, side, component):
    """
    Plot a stress component or failure index for one ply surface.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
        Computational domain.
    ply_surface_fields : dict
        Projected ply surface stress and failure-index fields.
    ply_id : int
        Ply number, where ply 0 is the topmost layer.
    side : str
        Ply surface, either ``"bot"`` or ``"top"``.
    component : str
        Component or index to plot.

    Returns
    -------
    None
        Displays a PyVista plot on MPI rank 0.
    """
    if domain.comm.rank != 0:
        return

    topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    num_local_cells = domain.topology.index_map(domain.topology.dim).size_local

    key = (ply_id, side, component)
    field = ply_surface_fields[key]
    name = f"ply{ply_id} {side} {component}"

    grid.cell_data[name] = field.x.array[:num_local_cells]

    plotter = pyvista.Plotter()
    plotter.add_mesh(
        grid,
        scalars=name,
        show_edges=False,
        scalar_bar_args={
            "title": name + "\n\n",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.1,
            "width": 0.06,
            "height": 0.8,
            "fmt": "%.1f",
            "label_font_size": 40,
        },
    )
    plotter.show_axes()
    plotter.view_xy()
    plotter.show()