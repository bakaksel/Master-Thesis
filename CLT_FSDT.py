"""
Author: Aksel Gundersen
Project: Master thesis – Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design
Content: Defines laminate and core material properties, transforms ply stiffness matrices, and assembles CLT/FSDT stiffness matrices for sandwich laminates.
"""

import numpy as np


# Lamina, core and strength properties.
E11 = 125000.0
E22 = 8410.0
G12 = 4230.0
G13 = 5170.0
G23 = 3920.0
nu12 = 0.31
nu21 = nu12 * E22 / E11

kappa = 5.0 / 6.0
tL = 0.152

Gc13 = 482.6
Gc23 = 213.7
tc = 14

SafetyFactor = 1.5

Xt = 2172 / SafetyFactor
Xc = 1448 / SafetyFactor
Yt = 52.9 / SafetyFactor
Yc = 199.0 / SafetyFactor
S12 = 154 / SafetyFactor
S13 = 154 / SafetyFactor
S23 = 30.54 / SafetyFactor

Sc13 = 2.34 / SafetyFactor
Sc23 = 1.52 / SafetyFactor


# Reduced stiffnesses in the material coordinate system.
Q11 = E11 / (1.0 - nu12 * nu21)
Q22 = E22 / (1.0 - nu12 * nu21)
Q12 = nu12 * E22 / (1.0 - nu12 * nu21)

Q44 = G23
Q55 = G13
Q66 = G12


def rotated_Q_matrix(Q11, Q22, Q12, Q44, Q55, Q66, theta):
    """
    Transform reduced lamina stiffness matrices to a ply angle.

    Parameters
    ----------
    Q11, Q22, Q12, Q44, Q55, Q66 : float
        Reduced lamina stiffness terms in the material coordinate system.
    theta : float
        Ply angle in radians.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Transformed in-plane stiffness matrix and transverse shear stiffness matrix.
    """
    u1 = (3.0 * Q11 + 3.0 * Q22 + 2.0 * Q12 + 4.0 * Q66) / 8.0
    u2 = (Q11 - Q22) / 2.0
    u3 = (Q11 + Q22 - 2.0 * Q12 - 4.0 * Q66) / 8.0
    u4 = (Q11 + Q22 + 6.0 * Q12 - 4.0 * Q66) / 8.0
    u5 = (Q11 + Q22 - 2.0 * Q12 + 4.0 * Q66) / 8.0
    u6 = (Q44 + Q55) / 2.0
    u7 = (Q44 - Q55) / 2.0

    c2 = np.cos(2.0 * theta)
    c4 = np.cos(4.0 * theta)
    s2 = np.sin(2.0 * theta)
    s4 = np.sin(4.0 * theta)
    cs = np.cos(theta) * np.sin(theta)

    Qbar11 = u1 + u2 * c2 + u3 * c4
    Qbar12 = u4 - u3 * c4
    Qbar22 = u1 - u2 * c2 + u3 * c4
    Qbar16 = 0.5 * u2 * s2 + u3 * s4
    Qbar26 = 0.5 * u2 * s2 - u3 * s4
    Qbar66 = u5 - u3 * c4

    Qbar44 = u6 + u7 * c2
    Qbar45 = u7 * cs
    Qbar55 = u6 - u7 * c2

    Qbar_inplane = np.array([
        [Qbar11, Qbar12, Qbar16],
        [Qbar12, Qbar22, Qbar26],
        [Qbar16, Qbar26, Qbar66]
    ], dtype=float)

    Qbar_shear = np.array([
        [Qbar44, Qbar45],
        [Qbar45, Qbar55]
    ], dtype=float)

    return Qbar_inplane, Qbar_shear


def CLT_FSDT(
    bottom_skin,
    top_skin,
    return_total_thickness=False,
    return_ply_data=False
):
    """
    Assemble laminate stiffness matrices using CLT and FSDT.

    Parameters
    ----------
    bottom_skin : sequence of float
        Ply angles in radians for the bottom skin.
    top_skin : sequence of float
        Ply angles in radians for the top skin.
    return_total_thickness : bool, optional
        If True, include the total laminate thickness in the returned tuple.
    return_ply_data : bool, optional
        If True, include layer-wise geometry, stiffness and strength data.

    Returns
    -------
    tuple
        Laminate stiffness matrices A, B, D and As, with optional total thickness
        and ply data.
    """
    layers = []

    for theta in bottom_skin:
        Qb, Qs = rotated_Q_matrix(Q11, Q22, Q12, Q44, Q55, Q66, theta)
        layers.append({
            "name": "ply",
            "theta": float(theta),
            "thickness": float(tL),
            "Q_inplane": Qb,
            "Q_shear": Qs,
            "strength": {
                "Xt": float(Xt),
                "Xc": float(Xc),
                "Yt": float(Yt),
                "Yc": float(Yc),
                "S12": float(S12),
                "S13": float(S13),
                "S23": float(S23),
            }
        })

    Qcore_shear = np.array([
        [Gc23, 0.0],
        [0.0, Gc13]
    ], dtype=float)

    layers.append({
        "name": "core",
        "theta": None,
        "thickness": float(tc),
        "Q_inplane": np.zeros((3, 3), dtype=float),
        "Q_shear": Qcore_shear,
        "strength": {
            "Xt": None,
            "Xc": None,
            "Yt": None,
            "Yc": None,
            "S12": None,
            "S13": float(Sc13),
            "S23": float(Sc23),
        }
    })

    for theta in top_skin:
        Qb, Qs = rotated_Q_matrix(Q11, Q22, Q12, Q44, Q55, Q66, theta)
        layers.append({
            "name": "ply",
            "theta": float(theta),
            "thickness": float(tL),
            "Q_inplane": Qb,
            "Q_shear": Qs,
            "strength": {
                "Xt": float(Xt),
                "Xc": float(Xc),
                "Yt": float(Yt),
                "Yc": float(Yc),
                "S12": float(S12),
                "S13": float(S13),
                "S23": float(S23),
            }
        })

    total_thickness = sum(layer["thickness"] for layer in layers)

    z = [-total_thickness / 2.0]
    for layer in layers:
        z.append(z[-1] + layer["thickness"])
    z = np.array(z, dtype=float)

    A = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    D = np.zeros((3, 3), dtype=float)
    As = np.zeros((2, 2), dtype=float)

    ply_data = []

    for k, layer in enumerate(layers):
        zk = z[k]
        zk1 = z[k + 1]
        dz = zk1 - zk

        Qk = layer["Q_inplane"]
        Qsk = layer["Q_shear"]

        A += Qk * dz
        B += 0.5 * Qk * (zk1**2 - zk**2)
        D += (1.0 / 3.0) * Qk * (zk1**3 - zk**3)
        As += kappa * Qsk * dz

        ply_data.append({
            "name": layer["name"],
            "theta": layer["theta"],
            "thickness": dz,
            "z_bot": zk,
            "z_top": zk1,
            "Qbar": Qk.copy(),
            "Qsbar": Qsk.copy(),
            "strength": layer["strength"].copy()
        })

    tol = 1e-12
    A[np.abs(A) < tol] = 0.0
    B[np.abs(B) < tol] = 0.0
    D[np.abs(D) < tol] = 0.0
    As[np.abs(As) < tol] = 0.0

    out = [A, B, D, As]

    if return_total_thickness:
        out.append(total_thickness)
    if return_ply_data:
        out.append(ply_data)

    return tuple(out)