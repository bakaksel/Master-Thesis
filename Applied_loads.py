"""
Author: Aksel Gundersen
Project: Master thesis – Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design
Content: Computes wishbone force components and moment contributions for longitudinal and lateral load cases.
"""

import numpy as np


# Geometry, vehicle and load parameters.
l = 1550
w = 1240
l1 = 289.7
l2 = 288.9
l3 = 351.9
l4 = 352.6
lx = 300
h1 = 118.2
h2 = 170
mu = 1.6
g = 9.81
mf = 142.5
mt = 285
h_CG_U = 372.7

# Moment arm from the load application point.
R = np.array([0.0, 7.31, 22.5], dtype=float)

# Unit direction vectors for the four wishbone members.
u = np.array([
    [ 0.52 ,  0.306,  0.797],
    [ 0.516, -0.307, -0.799],
    [-0.424, -0.281, -0.861],
    [-0.428,  0.28 ,  0.859]
], dtype=float)


def longitudinal_loads():
    """
    Calculate forces and moments for the longitudinal load case.

    Returns
    -------
    dict
        Dictionary containing the load case name, scalar member forces,
        force vectors and moment vectors.
    """
    mass_term = mf + (mt * mu * h_CG_U) / l
    k_upper = (h1 * mu * g) / (2 * h2 * lx)
    k_lower = ((h1 + h2) * mu * g) / (2 * h2 * lx)

    F = np.array([
        l1 * k_upper * mass_term,
        l2 * k_upper * mass_term,
        l3 * k_lower * mass_term,
        l4 * k_lower * mass_term
    ], dtype=float)

    P = F[:, None] * u
    m = np.cross(R, u)
    M = -(F[:, None] * m)

    return {
        "name": "longitudinal",
        "F": F,
        "P": P,
        "M": M
    }


def lateral_loads():
    """
    Calculate forces and moments for the lateral load case.

    Returns
    -------
    dict
        Dictionary containing the load case name, scalar member forces,
        force vectors and moment vectors.
    """
    beta_u = np.array([
        np.arccos((l1**2 + lx**2 - l2**2) / (2 * l1 * lx)),
        np.arccos((l2**2 + lx**2 - l1**2) / (2 * l2 * lx))
    ])

    beta_l = np.array([
        np.arccos((l3**2 + lx**2 - l4**2) / (2 * l3 * lx)),
        np.arccos((l4**2 + lx**2 - l3**2) / (2 * l4 * lx))
    ])

    mass_term_c = mf + (mu * mt * h_CG_U) / w
    k_upper_c = (h1 * mu * g) / (2 * h2)
    k_lower_c = (mu * g / 2) * (1 + h1 / h2)

    F1c = -k_upper_c * mass_term_c * (l2 * np.cos(beta_u[1]) / (lx * np.sin(beta_u[0])))
    F2c =  k_upper_c * mass_term_c * (l1 * np.cos(beta_u[0]) / (lx * np.sin(beta_u[1])))
    F3c =  k_lower_c * mass_term_c * (l4 * np.cos(beta_l[1]) / (lx * np.sin(beta_l[0])))
    F4c = -k_lower_c * mass_term_c * (l3 * np.cos(beta_l[0]) / (lx * np.sin(beta_l[1])))

    F = np.array([F1c, F2c, F3c, F4c], dtype=float)

    P = F[:, None] * u
    m = np.cross(R, u)
    M = -(F[:, None] * m)

    return {
        "name": "lateral",
        "F": F,
        "P": P,
        "M": M
    }


# Available load cases evaluated by this script.
LOAD_CASES = {
    "longitudinal": longitudinal_loads(),
    "lateral": lateral_loads()
}


if __name__ == "__main__":
    for case_name, case in LOAD_CASES.items():
        print(f"\n--- {case_name.upper()} ---")
        for i in range(4):
            print(
                f'F{i+1} = {case["F"][i]:.0f}\n'
                f'P{i+1} = {np.round(case["P"][i], 0)}\n'
                f'M{i+1} = {np.round(case["M"][i], 1)}\n'
            )