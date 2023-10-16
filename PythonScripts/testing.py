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


if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
