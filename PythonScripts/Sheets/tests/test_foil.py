import numpy as np
from sympy import symbols, Matrix

m, n = symbols("m n")
q11, q12, q13, q21, q22, q23, q31, q32, q33 = symbols(
    "q11 q12 q13 q21 q22 q23 q31 q32 q33"
)

Q = Matrix([[q11, q12, 0], [q12, q22, 0], [0, 0, q33]])

T = Matrix(
    [
        [m**2, n**2, 2 * m * n],
        [n**2, m**2, -2 * m * n],
        [-m * n, m * n, m**2 - n**2],
    ]
)

T_inv = T.inv()
Qbar = T_inv * Q * T_inv.T

print(Qbar[0, 0])
