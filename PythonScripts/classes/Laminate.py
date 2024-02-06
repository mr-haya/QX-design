import xlwings as xw
import numpy as np
import pandas as pd

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf

from .Prepreg import Prepreg


class Laminate:
    def __init__(self, laminate_name):
        wb = xw.Book.caller()
        laminate_sheet = wb.sheets[sn.laminate]
        laminate_df = (
            laminate_sheet[ca.laminate_cell].options(pd.DataFrame, index=1).value
        )

        elements = laminate_df["積層構成"][laminate_name].split(",")
        self.total_count = len(elements)
        self.obi_count = sum("オビ" in element for element in elements)
        self.angles = [int(element.replace("オビ", "")) for element in elements]

        self.prepreg = Prepreg(laminate_df["プリプレグ"][laminate_name])
        self.thickness = self.prepreg.t * len(self.angles)
        self.thickness_zenshu = self.prepreg.t * (self.total_count - self.obi_count)
        self.E_equiv, self.nu_equiv, self.G_equiv = stiffness_index(
            self.angles, self.prepreg
        )


def stiffness_index(angles, prepreg):
    # Initialize ABD matrix
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))

    # Iterate over each layer
    for i, angle in enumerate(angles):
        # Convert the angle to radians
        theta = np.radians(angle)

        # Get the material properties for this layer
        EL = prepreg.ELc
        ET = prepreg.ETc
        nuTc = prepreg.nuTc
        nuLc = prepreg.nuLc
        GLT = prepreg.GLT

        # Calculate the transformation matrix
        l = np.cos(theta)
        m = np.sin(theta)
        # T = np.array(
        #     [
        #         [m**2, n**2, 2 * m * n],
        #         [n**2, m**2, -2 * m * n],
        #         [-m * n, m * n, m**2 - n**2],
        #     ]
        # )

        # # Calculate the stiffness matrix for this layer
        # Q = np.array(
        #     [
        #         [EL / (1 - nuTc * nuLc), (EL * nuTc) / (1 - nuTc * nuLc), 0],
        #         [(EL * nuTc) / (1 - nuTc * nuLc), ET / (1 - nuTc * nuLc), 0],
        #         [0, 0, GLT],
        #     ]
        # )
        Q11 = EL / (1 - nuTc * nuLc)
        Q22 = ET / (1 - nuTc * nuLc)
        Q12 = (EL * nuTc) / (1 - nuTc * nuLc)
        Q66 = GLT
        Q11_t = Q11 * l**4 + Q22 * m**4 + 2 * (Q12 + 2 * Q66) * l**2 * m**2
        Q22_t = Q11 * m**4 + Q22 * l**4 + 2 * (Q12 + 2 * Q66) * l**2 * m**2
        Q12_t = (Q11 + Q22 - 4 * Q66) * l**2 * m**2 + Q12 * (l**4 + m**4)
        Q16_t = (Q11 - Q12 - 2 * Q66) * l**3 * m + (Q12 - Q22 + 2 * Q66) * l * m**3
        Q26_t = (Q11 - Q12 - 2 * Q66) * l * m**3 + (Q12 - Q22 + 2 * Q66) * l**3 * m
        Q66_t = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * l**2 * m**2 + Q66 * (
            l**4 + m**4
        )
        Q = np.array(
            [[Q11_t, Q12_t, Q16_t], [Q12_t, Q22_t, Q26_t], [Q16_t, Q26_t, Q66_t]]
        )

        z_k = prepreg.t * 10**3 * (i - len(angles) / 2)
        z_k_1 = prepreg.t * 10**3 * (i - len(angles) / 2 - 1)

        # Add the contribution of this layer to the overall stiffness matrix
        A += Q * (z_k - z_k_1) * 10**9
        B += Q * (z_k**2 - z_k_1**2) / 2 * 10**9
        D += Q * (z_k**3 - z_k_1**3) / 3 * 10**9

    # Calculate the equivalent elastic modulus
    thickness = prepreg.t * 10**3 * len(angles)
    # S_Matrix = np.block([[A, B], [B, D]])
    Q_bar = A / thickness
    S = np.linalg.inv(Q_bar)
    E_equiv = 1 / S[0, 0] * 10**-9
    nu_equiv = -S[0, 1] / S[0, 0]
    G_equiv = 1 / S[2, 2] * 10**-9
    return E_equiv, nu_equiv, G_equiv
