"""
翼型のデータを読み込み、エクセルに書き込むプログラム

前提条件
・翼型のデータは、(エクセルの存在するフォルダ)\翼型\(翼型名のフォルダ)内に格納されている
・翼型の座標データは、翼型名.datというファイル名で格納されている
・翼型の解析データは、翼型名_T1_Re0.000_M0.00_N9.0.txtというファイル名で格納されている

"""

import os
import xlwings as xw
import pandas as pd

import config.cell_adress as ca
import config.sheet_name as sn
import config.settings as st
from classes.Airfoil import Airfoil


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sheet = wb.sheets[sn.foil]
    df = sheet["B10:S16"].options(pd.DataFrame, index=1).value

    # 解析範囲を読み込む
    alpha_min = sheet.range(ca.alpha_min_cell).value
    alpha_max = sheet.range(ca.alpha_max_cell).value
    alpha_step = sheet.range(ca.alpha_step_cell).value
    Re_min = sheet.range(ca.Re_min_cell).value
    Re_max = sheet.range(ca.Re_max_cell).value
    Re_step = sheet.range(ca.Re_step_cell).value

    for i, foil_name in enumerate(list(df.index[2:])):
        # 翼型のインスタンスを生成
        print(foil_name)
        foil = Airfoil(
            foil_name, alpha_min, alpha_max, alpha_step, Re_min, Re_max, Re_step
        )
        # 値を反映
        thickness, tmax_at = foil.max_thickness()
        df["翼厚"][foil_name], df["tmax_at"][foil_name] = (
            round((thickness * 100), 1),
            tmax_at * 100,
        )
        cam, cmax_at = foil.max_camber()
        df["キャンバー"][foil_name], df["cmax_at"][foil_name] = (
            round(cam * 100, 1),
            cmax_at * 100,
        )
        df["前縁曲率半径比"][foil_name] = foil.leading_edge_radius()
        df["後縁角"][foil_name] = foil.trairing_edge_angle()
        # df["最大揚力係数"][foil_name] = foil.CL_max(df["Re数"][foil_name])
        # df["最大揚抗比"][foil_name], df["l_dmax_at"][foil_name] = foil.L_D_max(
        #     df["Re数"][foil_name]
        # )
        df["揚力係数"][foil_name] = foil.CL(df["迎角"][foil_name], df["Re数"][foil_name])
        df["揚抗比"][foil_name] = foil.L_D(df["迎角"][foil_name], df["Re数"][foil_name])
        df["ピッチングモーメント係数"][foil_name] = foil.Cm(
            df["迎角"][foil_name], df["Re数"][foil_name]
        )
        df["風圧中心"][foil_name] = (
            foil.XCp(df["迎角"][foil_name], df["Re数"][foil_name]) * 100
        )
        df["遷移位置"][foil_name] = (
            foil.Top_Xtr(df["迎角"][foil_name], df["Re数"][foil_name]) * 100
        )
        # 翼型の外形を描画
        sheet.pictures.add(
            foil.outline(),
            name=foil_name,
            update=True,
            left=sheet.range(13 + i, 19).left + 5,
            top=sheet.range(13 + i, 19).top + 7,
        )
    # エクセルに書き込み
    sheet["B10:S16"].value = df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = st.book_name
    file_path = os.path.join(script_dir, "..", file_name)

    xw.Book(file_path).set_mock_caller()
    main()
