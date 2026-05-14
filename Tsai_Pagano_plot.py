"""
Author: Aksel Gundersen
Project: Master thesis – Finite Element Analysis of Composite Laminated Sandwich Structures in Formula Student Chassis Design
Content: Plots laminate stiffness terms as a function of ply angle using the CLT/FSDT formulation with Tsai-Pagano based stiffness transformation.
"""

import numpy as np
import matplotlib.pyplot as plt

from CLT_FSDT import CLT_FSDT


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "font.size": 25,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25
})


def plot_laminate_sweep():
    """
    Plot selected laminate stiffness terms for a sweep of ply angles.

    Returns
    -------
    None
        Displays a figure containing extensional, bending, shear and coupling stiffness terms.
    """
    angles_deg = np.linspace(-90, 90, 181)
    angles_rad = np.deg2rad(angles_deg)

    A11, A22, A66 = [], [], []
    D11, D22, D66 = [], [], []
    As11, As22 = [], []
    B11, B22 = [], []

    for theta in angles_rad:
        top = [np.pi/4, np.pi/2, theta, 0]
        bottom = [0, np.pi/2, np.pi/4]

        A, B, D, As = CLT_FSDT(bottom, top)

        A11.append(A[0, 0])
        A22.append(A[1, 1])
        A66.append(A[2, 2])

        D11.append(D[0, 0])
        D22.append(D[1, 1])
        D66.append(D[2, 2])

        As11.append(As[0, 0])
        As22.append(As[1, 1])

        B11.append(B[0, 0])
        B22.append(B[1, 1])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axs[0, 0].plot(angles_deg, A11, label=r"$A_{11}$")
    axs[0, 0].plot(angles_deg, A22, label=r"$A_{22}$")
    axs[0, 0].plot(angles_deg, A66, label=r"$A_{66}$")
    axs[0, 0].set_title(r"Extensional stiffness")
    axs[0, 0].set_ylabel(r"Stiffness [MPa mm]")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[0, 1].plot(angles_deg, D11, label=r"$D_{11}$")
    axs[0, 1].plot(angles_deg, D22, label=r"$D_{22}$")
    axs[0, 1].plot(angles_deg, D66, label=r"$D_{66}$")
    axs[0, 1].set_title(r"Bending stiffness")
    axs[0, 1].set_ylabel(r"Stiffness [MPa mm$^3$]")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 0].plot(angles_deg, As11, label=r"$A^{s}_{11}$")
    axs[1, 0].plot(angles_deg, As22, label=r"$A^{s}_{22}$")
    axs[1, 0].set_title(r"Shear stiffness")
    axs[1, 0].set_xlabel(r"Angle [deg]")
    axs[1, 0].set_ylabel(r"Stiffness [MPa mm]")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[1, 1].plot(angles_deg, B11, label=r"$B_{11}$")
    axs[1, 1].plot(angles_deg, B22, label=r"$B_{22}$")
    axs[1, 1].set_title(r"Coupling ($B$ matrix)")
    axs[1, 1].set_xlabel(r"Angle [deg]")
    axs[1, 1].set_ylabel(r"Stiffness [MPa mm$^2$]")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_laminate_sweep()
