"""
主翼計算
"""


import os
import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from scipy import interpolate

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf
from geometry.Airfoil import GeometricalAirfoil
from classes.Spar import WingSpar


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sht = wb.sheets[sn.wing]

    # シートから値を読み取る
    foil1name_arr = sht.range(ca.foil1name).expand("down").value
    foil1rate_arr = sht.range(ca.foil1rate).expand("down").value
    foil2name_arr = sht.range(ca.foil2name).expand("down").value
    foil2rate_arr = sht.range(ca.foil2rate).expand("down").value
    chordlen_arr = sht.range(ca.chordlen).expand("down").value
    taper_arr = sht.range(ca.taper).expand("down").value
    diam_z_arr = sht.range(ca.diam_z).expand("down").value
    diam_x_arr = sht.range(ca.diam_x).expand("down").value
    spar_position_arr = sht.range(ca.spar_position).expand("down").value
    alpha_rib_arr = sht.range(ca.alpha_rib).expand("down").value

    # 主桁のインスタンスを生成
    wing_spar = WingSpar()
    results = []

    for i, y in enumerate(sheet.range(ca.spar_yn_cell).expand("down").value):
        # 値を反映
        results.append(
            list(wing_spar.zixizoxo(y))
            + list(wing_spar.section_modulus(y))
            + [wing_spar.E(y), wing_spar.G(y)]
        )

    # エクセルに書き込み
    sheet.range(ca.spar_export_cell).value = results


if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
