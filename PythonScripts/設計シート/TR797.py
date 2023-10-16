# -*- coding: utf-8 -*-
# -----------
# Optimum Design of Nonplanar wings-Minimum Induced Drag
# for A Given Lift and Wing Root Bending Moment (NAL TR-797)
#
# Created by Takahiro Inagawa on 2018-03-24.
# Copyright (c) 2018 Takahiro Inagawa. All rights reserved.
# -----------

import numpy as np
from numpy import sin, cos, tan, pi, sqrt
import matplotlib.pyplot as plt


class Wing:
    def __init__(self):
        self.lift = 1000  # Lift [N]
        self.Uinf = 7.2  # Aircraft speed [m/s]
        self.rho = 1.2  # Air density [kg/m3]
        self.span = 30  # Span width [m]
        self.le = self.span / 2  # length from root to tip
        self.N = 100  # Partition number
        self.beta = 0.85  # Coefficient related to the bending moment
        self.delta_S = (
            np.ones(self.N) * self.le / self.N / 2
        )  # hafl of partition, Equal distance
        self.phi = np.zeros(self.N)  # local dihedral angle [rad]

    def calc(self):
        N = self.N
        le = self.le
        delta_S = self.delta_S
        lift = self.lift
        Uinf = self.Uinf
        rho = self.rho
        span = self.span
        beta = self.beta

        # ---- Geometric Condition ----
        # yz plane
        # y axis is vertical, z axis is horizontal.
        # phi is the local dihedral angel.
        # y is the center of the partition.
        # -----------------------------
        y = np.linspace(delta_S[0], le - delta_S[N - 1], N)
        z = np.zeros(N)
        phi = self.phi

        ydash = np.zeros((N, N))
        zdash = np.zeros((N, N))
        y2dash = np.zeros((N, N))
        z2dash = np.zeros((N, N))
        R2puls = np.zeros((N, N))
        R2minus = np.zeros((N, N))
        Rdash2puls = np.zeros((N, N))
        Rdash2minus = np.zeros((N, N))
        Q = np.zeros((N, N))
        q = np.zeros((N, N))

        # ---- Geometric variable number ----
        # Variables required to seek the Q.
        for i in range(N):
            for j in range(N):
                ydash[i, j] = (y[i] - y[j]) * cos(phi[j]) + (z[i] - z[j]) * sin(phi[j])
                zdash[i, j] = (y[i] - y[j]) * sin(phi[j]) + (z[i] - z[j]) * cos(phi[j])
                y2dash[i, j] = (y[i] + y[j]) * cos(phi[j]) - (z[i] - z[j]) * sin(phi[j])
                z2dash[i, j] = (y[i] + y[j]) * sin(phi[j]) + (z[i] - z[j]) * cos(phi[j])

                R2puls[i, j] = (ydash[i, j] - delta_S[j]) ** 2 + zdash[i, j] ** 2
                R2minus[i, j] = (ydash[i, j] + delta_S[j]) ** 2 + zdash[i, j] ** 2
                Rdash2puls[i, j] = (y2dash[i, j] + delta_S[j]) ** 2 + z2dash[i, j] ** 2
                Rdash2minus[i, j] = (y2dash[i, j] - delta_S[j]) ** 2 + z2dash[i, j] ** 2

        for i in range(N):
            for j in range(N):
                Q[i, j] = (
                    -1
                    / (2 * pi)
                    * (
                        (
                            (ydash[i, j] - delta_S[j]) / R2puls[i, j]
                            - (ydash[i, j] + delta_S[j]) / R2minus[i, j]
                        )
                        * cos(phi[i] - phi[j])
                        + (zdash[i, j] / R2puls[i, j] - zdash[i, j] / R2minus[i, j])
                        * sin(phi[i] - phi[j])
                        + (
                            (y2dash[i, j] - delta_S[j]) / Rdash2minus[i, j]
                            - (y2dash[i, j] + delta_S[j]) / Rdash2puls[i, j]
                        )
                        * cos(phi[i] + phi[j])
                        + (
                            z2dash[i, j] / Rdash2minus[i, j]
                            - z2dash[i, j] / Rdash2puls[i, j]
                        )
                        * sin(phi[i] + phi[j])
                    )
                )

        # ---- Normalization ----
        # Variables required to seek q.
        delta_sigma = delta_S / le
        eta = y / le
        etadash = ydash / le
        eta2dash = y2dash / le
        zeta = z / le
        zetadash = zdash / le
        zeta2dash = z2dash / le
        gamma2puls = R2puls / (le**2)
        gamma2minus = R2minus / (le**2)
        gammadash2puls = Rdash2puls / (le**2)
        gammadash2minus = Rdash2minus / (le**2)

        for i in range(N):
            for j in range(N):
                q[i, j] = (
                    -1
                    / (2 * pi)
                    * (
                        (
                            (etadash[i, j] - delta_sigma[j]) / gamma2puls[i, j]
                            - (etadash[i, j] + delta_sigma[j]) / gamma2minus[i, j]
                        )
                        * cos(phi[i] - phi[j])
                        + (
                            zetadash[i, j] / gamma2puls[i, j]
                            - zetadash[i, j] / gamma2minus[i, j]
                        )
                        * sin(phi[i] - phi[j])
                        + (
                            (eta2dash[i, j] - delta_sigma[j]) / gammadash2minus[i, j]
                            - (eta2dash[i, j] + delta_sigma[j]) / gammadash2puls[i, j]
                        )
                        * cos(phi[i] + phi[j])
                        + (
                            zeta2dash[i, j] / gammadash2minus[i, j]
                            - zeta2dash[i, j] / gammadash2puls[i, j]
                        )
                        * sin(phi[i] + phi[j])
                    )
                )

        # ---- elliptic loading aerodynamic force ----
        # Vn is Induced vertical velocisy.
        # Vn is constant when elliptical circulation distribution.
        bending_moment_elpl = 2 / 3 / pi * le * lift
        induced_drag_elpl = lift**2 / (2 * pi * rho * Uinf**2 * le**2)
        Vn_elpl = lift / (2 * pi * rho * Uinf * le**2)

        # ---- Creating the optimization equation ----
        c = 2 * cos(phi) * delta_sigma
        b = 3 * pi / 2 * (eta * cos(phi) + zeta * sin(phi)) * delta_sigma
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                A[i, j] = pi * q[i, j] * delta_sigma[i]

        # ---- solve optimization problem ----
        AAA = A + A.T
        ccc = -c
        cccT = ccc.reshape(N, 1)
        ccc0 = np.append(ccc, np.zeros(2)).reshape(1, N + 2)
        bbb = -b
        bbbT = bbb.reshape(N, 1)
        bbb0 = np.append(bbb, np.zeros(2)).reshape(1, N + 2)
        AAAcb = np.concatenate((AAA, cccT, bbbT), axis=1)
        # import pdb; pdb.set_trace()
        left_matrix = np.concatenate((AAAcb, ccc0, bbb0), axis=0)
        right_matrix = np.append(np.zeros(N), np.array([-1, -beta]))

        solve_matrix = np.linalg.solve(left_matrix, right_matrix)
        g = solve_matrix[0:N]
        mu = solve_matrix[N : N + 2]  # Lagrange multiplier

        # ---- After Solve ----
        # efficient : span efficiency
        # Gamma : Local circulation
        # InducedDrag : Sum of induced drag
        # Vn : Local induced vertical velocisy
        # Lift0_elpl : Lift at root culcurated by area of the ellipse circulation distribution
        # Gamma0_elpl : Circulation at root
        # Gamma_elpl : Analytical ellipse circulation distribution @ beta = 1
        # ---------------------
        efficiency_inverse = np.dot(np.dot(g, A), g)
        efficiency = 1 / efficiency_inverse
        Gamma = g * lift / (2 * le * rho * Uinf)
        induced_drag = efficiency_inverse * induced_drag_elpl

        local_lift = 4 * rho * Uinf * Gamma.T * cos(phi)
        Vn = np.zeros(N)
        for i in range(N):
            for j in range(N):
                Vn[i] += Q[i, j] * Gamma[j]
        local_induced_drag = rho * Gamma * Vn

        # ---- Aerodynamic force when Elliptical cierculation distribution----
        lift0_elpl = 2 * lift / pi / le
        lift_elpl = 4 * lift0_elpl * sqrt(1 - (y / le) ** 2)
        Gamma0_elpl = lift0_elpl / (rho * Uinf * cos(phi[0]))
        Gamma_elpl = Gamma0_elpl * sqrt(1 - (y / le) ** 2)
        local_induced_drag_elpl = 2 * rho * Gamma_elpl * Vn_elpl

        # ---- Bending Moment ----
        local_bending_moment = np.zeros(N)
        local_bending_moment_elpl = np.zeros(N)
        for i in range(N):
            tmp1 = 0
            tmp2 = 0
            for j in range(i, N):
                tmp1 += local_lift[j] * (y[j] - y[i])
                tmp2 += lift_elpl[j] * (y[j] - y[i])
            local_bending_moment[i] = tmp1
            local_bending_moment_elpl[i] = tmp2

        # ---- Display Input and Output ----
        print("==== Input ====")
        print("Lift\t\t\t: %.1f [N]" % (lift))
        print("Aircraft speed\t\t: %.2f [m/s]" % (Uinf))
        print("Wing span width\t\t: %.1f [m]" % (span))
        print("Partition number\t: %d" % (N))
        print("beta\t\t\t: %.2f [-]" % (beta))
        print("==== Output ====")
        print("Induced Drag\t\t: %.2f [N]" % (induced_drag))
        print("efficiency\t\t: %.1f [%%]" % (efficiency * 100))
        print("==== cf. ellipse circulation distribution ====")
        print("Induced Drag\t\t: %.2f [N]" % (induced_drag_elpl))

        # ---- Plot ----
        plt.ion()
        plt.close("all")

        plt.figure()
        plt.plot(y, Gamma, label="beta=%.2f" % (beta))
        plt.plot(y, Gamma_elpl, label="ellipse circulation distribution")
        plt.xlabel("span [m]")
        plt.ylabel("Circulation")
        plt.grid()
        plt.legend()
        plt.title("Circulation")
        plt.savefig("circulation.png")

        plt.figure()
        plt.plot(y, local_lift, label="beta=%.2f" % (beta))
        plt.plot(y, lift_elpl, label="ellipse circulation distribution")
        plt.xlabel("span [m]")
        plt.ylabel("Lift [N]")
        plt.grid()
        plt.legend()
        plt.title("Lift")
        plt.savefig("lift.png")

        plt.figure()
        plt.plot(y, local_bending_moment, label="beta=%.2f" % (beta))
        plt.plot(y, local_bending_moment_elpl, label="ellipse circulation distribution")
        plt.xlabel("span [m]")
        plt.ylabel("Bending Moment [Nm]")
        plt.grid()
        plt.legend()
        plt.title("Bending Moment")
        plt.savefig("bending_moment.png")

        plt.figure()
        plt.plot(y, Vn, label="beta=%.2f" % (beta))
        plt.plot(y, np.ones(N) * Vn_elpl, label="ellipse circulation distribution")
        plt.xlabel("span [m]")
        plt.ylabel("velocity [m/s]")
        plt.grid()
        plt.legend()
        plt.title("Induced vertical velocity")
        plt.savefig("induced_vertical_velocity.png")

        plt.figure()
        plt.plot(y, local_induced_drag, label="beta=%.2f" % (beta))
        plt.plot(y, local_induced_drag_elpl, label="ellipse circulation distribution")
        plt.xlabel("span [m]")
        plt.ylabel("Drag [N]")
        plt.grid()
        plt.legend()
        plt.title("Induced Drag")
        plt.savefig("induced_drag.png")

        plt.show()


if __name__ == "__main__":
    wing = Wing()
    wing.calc()
