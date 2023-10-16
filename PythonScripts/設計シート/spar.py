"""
桁のデータを読み込み、エクセルに書き込むプログラム

"""

import os
import xlwings as xw

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf

from classes.Spar import WingSpar


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sheet = wb.sheets[sn.spar]

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
