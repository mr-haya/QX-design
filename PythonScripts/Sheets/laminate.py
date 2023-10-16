import xlwings as xw
import pandas as pd
import numpy as np

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf
from classes.Laminate import Laminate


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sheet = wb.sheets[sn.laminate]
    df = sheet[ca.laminate_cell].options(pd.DataFrame, index=1).value

    for i, laminate_name in enumerate(list(df.index[2:])):
        # 積層板のインスタンスを生成
        laminate = Laminate(laminate_name)
        df["積層数"][laminate_name] = laminate.total_count
        df["全周"][laminate_name] = laminate.total_count - laminate.obi_count
        df["オビ"][laminate_name] = laminate.obi_count
        df["厚さ"][laminate_name] = laminate.thickness
        df["全周厚さ"][laminate_name] = laminate.thickness_zenshu
        df["相当縦弾性率"][laminate_name] = laminate.E_equiv
        df["相当横弾性率"][laminate_name] = laminate.G_equiv
        df["ポアソン比"][laminate_name] = laminate.nu_equiv

    # エクセルに書き込み
    sheet.range(ca.laminate_cell).value = df


if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
