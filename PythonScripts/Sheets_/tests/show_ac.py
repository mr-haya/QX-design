import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from classes.Airfoil import Airfoil

import config.config as cf


file_path = cf.get_file_path()
xw.Book(file_path).set_mock_caller()

foil = Airfoil("QX0023")
dev = 100
alpha = np.linspace(-5, 6, dev)  # 迎角
Re = np.full(dev, 1e6)  # レイノルズ数
array = np.array([[alpha[i], Re[i]] for i in range(len(alpha))])
CL = np.array(foil.coefs["CL"](array))  # CLの値
CD = np.array(foil.coefs["CD"](array))  # CDの値
Cmc_4 = np.array(foil.coefs["Cm"](array))  # Cmc/4の値

# CLとCDの合力を計算
Cf = np.sqrt(CL**2 + CD**2)

# Cn（法線分力）を計算
Cn = Cf * np.cos(np.radians(alpha))

# Cm0（前縁まわりのピッチングモーメント係数）を計算
Cm0 = -0.25 * Cn + Cmc_4

# Cm0とCnの散布図を作成
plt.scatter(Cn, Cm0)

# # 近似直線をフィット
slope, intercept, r_value, p_value, std_err = linregress(Cn, Cm0)
print(f"近似直線の傾き: {slope}")
print(f"近似直線の切片: {intercept}")

plt.plot(Cn, intercept + slope * Cn, "r")
plt.xlabel("Cn")
plt.ylabel("Cm0")
plt.show()
