"""
翼型のデータを読み込み、エクセルに書き込むプログラム

前提条件
・翼型のデータは、(エクセルの存在するフォルダ)\翼型\(翼型名のフォルダ)内に格納されている
・翼型の座標データは、翼型名.datというファイル名で格納されている
・翼型の解析データは、翼型名_T1_Re0.000_M0.00_N9.0.txtというファイル名で格納されている

"""

import os
import xlwings as xw

from calc_airfoil import coef_interp
from calc_airfoil import calcClmax
from calc_airfoil import calcClCdmax
from calc_airfoil import export_outline
from tools import show_data

# 値読み込み先セル一覧
Re_min_cell = "C4"
Re_max_cell = "D4"
Re_step_cell = "E4"
alpha_min_cell = "C3"
alpha_max_cell = "D3"
alpha_step_cell = "E3"
ref_unit_cell = "G3"
ref_point_cell = "H3"
foil_name_cell = "B13"

# 値書き込み先セル一覧
thickness_cell = "C13"
thickness_at_cell = "D13"
camber_cell = "E13"
camber_at_cell = "F13"
points_number_cell = "G13"
whole_length_cell = "H13"
area_cell = "I13"
ClCdmax_cell = "J13"
Clmax_cell = "L13"
Cm_ref_cell = "M13"
Cma_ref_cell = "N13"
XCp_ref_cell = "O13"
Xtr_top_cell = "P13"
outline_cell = "Q13"

# 定数


def main():
    # 翼型名を取得
    wb = xw.Book.caller()
    sheet = wb.sheets["翼型"]
    foil_name = str(sheet.range(foil_name_cell).value)

    # Cl_interp = coef_interp(foil_name, "CL", 1000000, 5)

    # エクセルに書き込み
    # sheet.range(thickness_cell).value, sheet.range(thickness_at_cell).value  = thickness_of(foil_data)
    # sheet.range(camber_cell).value, sheet.range(camber_at_cell).value = camber_of(foil_data)
    # sheet.range(points_number_cell).value = points_of(foil_name)
    # sheet.range(whole_length_cell).value = calcWholeLength(foil_data)
    # sheet.range(area_cell).value = calcArea(foil_data)
    sheet.range(ClCdmax_cell).value = calcClCdmax(foil_name)
    sheet.range(Clmax_cell).value = calcClmax(foil_name)
    # sheet.range(Cm_ref_cell).value = calcCm_ref(foil_name, "Cm", Re_ref, alpha_ref)
    # sheet.range(Cma_ref_cell).value = calcCma_ref(foil_data)
    # sheet.range(XCp_ref_cell).value = calcXCp_ref(foil_data)
    # sheet.range(Xtr_top_cell).value = calcXtr_top(foil_data)

    # 翼型の外形を描画
    foil_outline = export_outline(foil_name)
    sheet.pictures.add(
        foil_outline,
        name="MyPlot",
        update=True,
        left=sheet.range(outline_cell).left + 5,
        top=sheet.range(outline_cell).top + 5,
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "QX-23_Eclipse_v1.xlsm"
    file_path = os.path.join(script_dir, "..", file_name)

    xw.Book(file_path).set_mock_caller()
    main()
