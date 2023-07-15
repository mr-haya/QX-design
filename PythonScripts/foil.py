"""
翼型のデータを読み込み、エクセルに書き込むプログラム

"""

import xlwings as xw

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf
from classes.Airfoil import Airfoil


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sheet = wb.sheets[sn.foil]

    # 出力用配列
    output = []

    for i, array in enumerate(sheet.range(ca.foil_detail_cell).expand("table").value):
        # 翼型のインスタンスを生成
        foil = Airfoil(array[0])
        # 値を反映
        output.append(
            [
                foil.name,
                foil.thickness_max * 100,
                foil.tmax_at * 100,
                foil.camber_max * 100,
                foil.cmax_at * 100,
                foil.leading_edge_radius,
                foil.trailing_edge_angle,
                array[7],
                foil.CL_max(array[7]),
            ]
            + list(foil.L_D_max(array[7]))
            + [
                array[11],
                foil.CL(array[11], array[7]),
                foil.L_D(array[11], array[7]),
                foil.Cm(array[11], array[7]),
                foil.XCp(array[11], array[7]) * 100,
                foil.Top_Xtr(array[11], array[7]) * 100,
            ]
        )
        # 翼型の外形を描画
        outline_cell = sheet.range(ca.foil_detail_cell).offset(i, 17)
        fig = foil.outline()
        width, height = fig.get_size_inches() * 72
        sheet.pictures.add(
            fig,
            name=array[0],
            update=True,
            left=outline_cell.left + 10,
            top=outline_cell.top + outline_cell.height / 2 - foil.thickness_max * 335,
        )
    # エクセルに書き込み
    sheet.range(ca.foil_detail_cell).value = output


if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
