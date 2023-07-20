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


def main():
    # シートを取得
    wb = xw.Book.caller()
    sht_wing = wb.sheets[sn.wing]
    sht_foil = wb.sheets[sn.foil]

    # wing = Wing()
    # wing.llt()
    # print("Di: ", wing.Induced_drag, "Dp: ", wing.Dp, "D: ", wing.Drag)
    # print("L: ", wing.Lift, "L/D: ", wing.Lift / wing.Drag)
    # print("S_def: ", wing.S_def)
    # alRe = [
    #     [wing.panels[i].alpha_effective, wing.panels[i].Re]
    #     for i in range(len(wing.panels))
    # ]
    # print(alRe)
    foil = Airfoil("NACA 0012")

    # 迎角範囲とそれに対応するCL, CD, Cmc/4の値を提供する必要があります。
    # 以下は仮のデータです。
    dev = 100
    alpha = np.linspace(-4, 0, dev)  # 迎角
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

    y = Cn * -0.25

    # Cm0とCnの散布図を作成
    plt.scatter(Cn, Cm0)

    plt.plot(Cn, y, "g")

    # # 近似直線をフィット
    slope, intercept, r_value, p_value, std_err = linregress(Cn, Cm0)
    plt.plot(Cn, intercept + slope * Cn, "r")

    plt.xlabel("Cn")
    plt.ylabel("Cm0")
    plt.show()

    print(f"近似直線の傾き: {slope}")
    print(f"近似直線の切片: {intercept}")


if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
