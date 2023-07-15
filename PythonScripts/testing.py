import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt

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

    foil1 = Airfoil("QX0123")
    print(foil1.perimeter(-1, 1))
    # wing = Wing()
    # wing.llt()
    # print([panel.Re for panel in wing.panels])


if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
