import os
import xlwings as xw
import pandas as pd
import numpy as np

import config.cell_adress as ca
import config.sheet_name as sn
import config.settings as st
from classes.Laminate import Laminate


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sheet = wb.sheets[sn.laminate]
    df = sheet[ca.laminate_cell].options(pd.DataFrame, index=1).value

    for i, laminate_name in enumerate(list(df.index[2:])):
        # 積層板のインスタンスを生成
        laminate = Laminate(
            laminate_name, df["プリプレグ"][laminate_name], df["積層構成"][laminate_name]
        )
        print(laminate.angles)
        df["積層数"][laminate_name] = laminate.total_count
        df["全周"][laminate_name] = laminate.total_count - laminate.obi_count
        df["オビ"][laminate_name] = laminate.obi_count
        df["厚さ"][laminate_name] = laminate.thickness
        df["全周厚さ"][laminate_name] = laminate.thickness_zenshu
        df["相当縦弾性率"][laminate_name] = laminate.E_equiv
        df["相当横弾性率"][laminate_name] = laminate.G_equiv
        df["ポアソン比"][laminate_name] = laminate.nu_equiv

    #     foil = Airfoil(
    #         foil_name, alpha_min, alpha_max, alpha_step, Re_min, Re_max, Re_step
    #     )
    #     # 値を反映
    #     thickness, tmax_at = foil.max_thickness()
    #     df["翼厚"][foil_name], df["tmax_at"][foil_name] = (
    #         round((thickness * 100), 1),
    #         tmax_at * 100,
    #     )
    #     cam, cmax_at = foil.max_camber()
    #     df["キャンバー"][foil_name], df["cmax_at"][foil_name] = (
    #         round(cam * 100, 1),
    #         cmax_at * 100,
    #     )
    #     df["前縁半径"][foil_name] = foil.leading_edge_radius()
    #     df["後縁角"][foil_name] = foil.trairing_edge_angle()
    #     # df["最大揚力係数"][foil_name] = foil.CL_max(df["Re数"][foil_name])
    #     # df["最大揚抗比"][foil_name], df["l_dmax_at"][foil_name] = foil.L_D_max(
    #     #     df["Re数"][foil_name]
    #     # )
    #     df["揚力係数"][foil_name] = foil.CL(df["迎角"][foil_name], df["Re数"][foil_name])
    #     df["揚抗比"][foil_name] = foil.L_D(df["迎角"][foil_name], df["Re数"][foil_name])
    #     df["ピッチングモーメント係数"][foil_name] = foil.Cm(
    #         df["迎角"][foil_name], df["Re数"][foil_name]
    #     )
    #     df["風圧中心"][foil_name] = (
    #         foil.XCp(df["迎角"][foil_name], df["Re数"][foil_name]) * 100
    #     )
    #     df["遷移位置"][foil_name] = (
    #         foil.Top_Xtr(df["迎角"][foil_name], df["Re数"][foil_name]) * 100
    #     )
    #     # 翼型の外形を描画
    #     sheet.pictures.add(
    #         foil.outline(),
    #         name=foil_name,
    #         update=True,
    #         left=sheet.range(13 + i, 19).left + 5,
    #         top=sheet.range(13 + i, 19).top + 7,
    #     )
    # エクセルに書き込み
    sheet[ca.laminate_cell].value = df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = st.book_name
    file_path = os.path.join(script_dir, "..", file_name)

    xw.Book(file_path).set_mock_caller()
    main()
