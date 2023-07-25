import os
import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import linregress

from classes.Airfoil import Airfoil
from classes.Rib import Rib
from classes.Wing import Wing

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf

file_path = cf.get_file_path()
xw.Book(file_path).set_mock_caller()
foil = Airfoil("QX0023")

# # Initialize the data array
# data = []

# # Loop over all files
# for i in range(len(foil.Re_list)):
#     # Create the file name
#     file_name = (
#         foil.name
#         + "_T1_Re"
#         + format((foil.Re_min + foil.Re_step * i) / 1000000, ".3f")
#         + "_M0.00_N9.0.txt"
#     )
#     file_path = os.path.join(foil.path, file_name)

#     # Open the file
#     with open(file_path, "r") as f:
#         lines = f.readlines()

#         for j, line in enumerate(lines):
#             if j > 10 and line.strip() != "":
#                 data_tmp = line.split()
#                 data_tmp = [x for x in data_tmp if x != ""]  # remove empty strings
#                 if foil.alpha_min <= float(data_tmp[0]) <= foil.alpha_max:
#                     row = [float(x) for x in data_tmp[:4]]
#                     row.append(foil.Re_min + foil.Re_step * i)
#                     data.append(row)

# # Convert the list of lists to a numpy array
# data = np.array(data)

# # Extract alpha and Re columns
# alpha = data[:, 0]
# Re = data[:, 4]

# coef = np.zeros((3, 15))

# # Approximate each column with a polynomial
# for k in range(3):
#     y = data[:, k + 1]
#     if k == 0:
#         X = np.array([alpha**i for i in range(9)]).T
#         tmp, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
#         coef[k, :9] = tmp
#     else:
#         X = np.array(
#             [alpha**i for i in range(9)] + [Re**j for j in range(1, 7)]
#         ).T
#         print(X.shape)
#         # m2 = X @ X.T
#         # m3 = np.linalg.inv(m2)
#         # m4 = m3 @ X
#         # m5 = m4 @ y
#         # coef[k, :] = m5
#         tmp, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
#         coef[k, :] = tmp
#     # coef = np.linalg.inv(X @ X.T) @ X @ y
# print(coef)
coefsht = np.array(
    [
        [
            6.102e-01,
            1.106e-01,
            6.717e-04,
            2.610e-04,
            -6.273e-05,
            -6.682e-06,
            1.349e-06,
            -6.877e-08,
            1.153e-09,
            0.000e00,
            0.000e00,
            0.000e00,
            0.000e00,
            0.000e00,
            0.000e00,
        ],
        [
            5.845e-02,
            -2.113e-04,
            1.094e-04,
            -2.076e-05,
            5.734e-07,
            5.332e-07,
            -5.571e-08,
            2.196e-09,
            -3.137e-11,
            -5.275e-07,
            2.563e-12,
            -6.521e-18,
            8.871e-24,
            -6.126e-30,
            1.688e-36,
        ],
        [
            -8.337e-02,
            2.833e-04,
            -1.112e-04,
            -4.567e-05,
            1.272e-05,
            8.245e-07,
            -2.352e-07,
            1.311e-08,
            -2.319e-10,
            -6.329e-07,
            3.065e-12,
            -7.629e-18,
            1.019e-23,
            -6.949e-30,
            1.897e-36,
        ],
    ]
)
alpha = np.linspace(-5, 20, 100)
Re = np.linspace(1e6, 1e5, 100)
X1, Y1 = np.meshgrid(alpha, Re, indexing="ij")
CL = np.zeros((len(alpha), len(Re)))
CD = np.zeros((len(alpha), len(Re)))
for i in range(len(alpha)):
    for j in range(len(Re)):
        for k in range(9):
            CL[i, j] += coefsht[0, k] * alpha[i] ** k
            CD[i, j] += coefsht[1, k] * alpha[i] ** k
        for l in range(1, 7):
            CD[i, j] += coefsht[1, 8 + l] * Re[j] ** l
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(X1, Y1, CL / CD, alpha=0.3, color="orange")

X2, Y2 = np.meshgrid(foil.alpha_list, foil.Re_list, indexing="ij")
# ax.scatter(X2, Y2, foil.xflr5[0] / foil.xflr5[1])

CL = np.array(foil.coefs["CL"]((X2, Y2)))
CD = np.array(foil.coefs["CD"]((X2, Y2)))
ax.plot_wireframe(X2, Y2, CL / CD, color=(0, 0, 1, 0.3))

plt.xlabel("alpha")
plt.ylabel("Re")

plt.show()
