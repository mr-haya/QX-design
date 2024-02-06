"""
リブ付け迎角合わせの治具をsvgとして出力するプログラム
"""

# DPIの設定
DPI = 96
mm = DPI / 25.4

import os
import xlwings as xw
import numpy as np
import svgwrite

import classes.Config as cf
from classes.Geometry import GeometricalAirfoil


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sht = wb.sheets[cf.Wing.name]

    # シートから値を読み取る
    foil1name_arr = sht.range(cf.Wing.foil1name).expand("down").value
    foil1rate_arr = sht.range(cf.Wing.foil1rate).expand("down").value
    foil2name_arr = sht.range(cf.Wing.foil2name).expand("down").value
    foil2rate_arr = sht.range(cf.Wing.foil2rate).expand("down").value
    chordlen_arr = sht.range(cf.Wing.chordlen).expand("down").value
    taper_arr = sht.range(cf.Wing.taper).expand("down").value
    spar_arr = sht.range(cf.Wing.spar).expand("down").value
    ishalf_arr = [
        item == "half" for item in sht.range(cf.Wing.ishalf).expand("down").value
    ]
    diam_z_arr = sht.range(cf.Wing.diam_z).expand("down").value
    diam_x_arr = sht.range(cf.Wing.diam_x).expand("down").value
    spar_position_arr = sht.range(cf.Wing.spar_position).expand("down").value
    alpha_rib_arr = sht.range(cf.Wing.alpha_rib).expand("down").value

    half_x_start_rate = -0.75
    half_x_end_rate = 0.4
    plank_start_rate = -0.75
    plank_end_rate = 0.25
    plank_thickness = 2 * mm
    ribcap_thickness = 0.45 * mm

    setline1_sp2 = -150 * mm
    setline2_sp2 = 200 * mm
    setline1_sp3 = -100 * mm
    setline2_sp3 = 150 * mm

    space = 120 * mm
    line_len = 5 * mm

    # 必要なdatファイルを呼び出して格納
    dat_dict = {}

    for foilname in ["rev_root_140"]:
        # フォルダのパスを取得
        foilpath = os.path.join(cf.Settings.AIRFOIL_PATH, foilname)

        # datデータを取得
        dat = np.loadtxt(
            fname=os.path.join(foilpath, foilname + ".dat"),
            dtype="float",
            skiprows=1,
        )

        # 辞書に追加
        dat_dict[foilname] = dat

    point_ref_count = 0

    for id in range(22, 69 + 1):  # リブ番号の範囲を指定
        chord = chordlen_arr[id] * mm
        section_spar = spar_arr[id]
        spar_position = spar_position_arr[id]
        alpha_rib = alpha_rib_arr[id]
        point_ref = np.array([60, 0]) * mm  # np.array([200, 150]) * mm
        is_half = ishalf_arr[id]
        point_ref_count += 1

        # SVGファイルの設定
        dwg = svgwrite.Drawing("output.svg")

        # スタイルの設定
        dwg.add(
            dwg.style(
                ".red { fill:none; stroke:red; stroke-width:3px; } .black { fill:none; stroke:black; stroke-width:3px; }"
            )
        )

        # 翼型のdatデータを取得
        dat_raw = dat_dict["rev_root_140"]

        # 幾何値参照用オブジェクトを作成
        geo = GeometricalAirfoil(dat_raw, chord_ref=chord)

        # 桁中心を計算
        spar_x = spar_position * chord
        spar_center = np.array([spar_x, geo.camber(spar_x)])

        # リブの段差オフセットを定義
        if is_half:
            offset_base = 0
        else:
            offset_base = ribcap_thickness
        offset_arr = np.array(
            [[plank_start_rate, plank_end_rate, plank_thickness]]
        )  # [[xstart, xend, depth],...]

        # リブオフセット
        dat_offset = geo.offset_foil(offset_base, offset_arr).copy()

        # ハーフリブの後ろ側を削除
        if is_half:
            half_x_start = half_x_start_rate * geo.chord_ref
            half_x_end = half_x_end_rate * geo.chord_ref
            half_start = (
                np.array([abs(half_x_start), geo.y(half_x_start)])
                + geo.nvec(half_x_start) * plank_thickness
            )
            half_end = np.array([abs(half_x_end), geo.y(half_x_end)])

            start_index = (
                next(
                    (
                        i
                        for i, point in enumerate(geo.dat_extended)
                        if point[0] > half_x_start
                    ),
                    None,
                )
                + 1
            )
            end_index = (
                next(
                    (
                        i
                        for i, point in enumerate(geo.dat_extended)
                        if point[0] > half_x_end
                    ),
                    None,
                )
                - 1
            ) + 4
            dat_out = np.vstack(
                [
                    [half_start],
                    dat_offset[start_index : end_index + 1],
                    [half_end],
                    [half_start],
                ]
            )
        else:
            dat_out = dat_offset.copy()

        # 桁中心を原点に移動＆迎角だけ回転
        dat_out = rotate_points(dat_out - spar_center, (0, 0), alpha_rib)

        if section_spar == "2番":
            setline1_offset = setline1_sp2
            setline2_offset = setline2_sp2

        elif section_spar == "3番":
            setline1_offset = setline1_sp3
            setline2_offset = setline2_sp3

        elif section_spar == "端リブ":
            setline1_offset = setline1_sp3
            setline2_offset = setline2_sp3

        # リブ描写
        # dwg.add(dwg.polygon(dat_out + point_ref, class_="red"))

        # 治具の作図
        for i, offset in enumerate([setline1_offset, setline2_offset]):
            intersections = find_line_intersection(dat_out, (offset, 0), 0)
            jig_points = (
                np.array(
                    [
                        [-60, 0],
                        [-60, 60],
                        [0, 140 + intersections[1][1] / mm],
                        [60, 60],
                        [60, 0],
                        [30, 0],
                        [30, 30],
                        [-30, 30],
                        [-30, 0],
                    ]
                )
                * mm
            )
            if i == 50:
                print(110 + intersections[1][1] / mm)

            # jig_points += np.array([offset, -140 * mm])
            jig_points += np.array([space, 0]) * i
            dwg.add(dwg.polygon(jig_points + point_ref, class_="red"))
            dwg.add(
                dwg.line(
                    jig_points[2] + point_ref,
                    jig_points[2] - np.array([0, line_len]) + point_ref,
                    class_="red",
                ),
            )
            # テキストを作成
            text = "LE" if i == 0 else "TE"
            text_location = np.array([30, 45]) * mm + np.array([space, 0]) * i
            dwg.add(
                dwg.text(
                    str(id) + text,
                    insert=text_location + point_ref,
                    class_="red",
                    font_size=10 * mm,
                )
            )

        file_name = (
            "C:/Users/soyha/OneDrive - Kyushu University/AircraftDesign/QX-design/Outputs/master/alpha_master/alpha_master_"
            + str(id)
        )

        # SVGファイルに保存
        file_name = file_name + ".svg"
        dwg.saveas(file_name)

    # メッセージを表示して終了
    print("ファイルが保存されました。")


def rotate_points(points, center, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)],
        ]
    )
    translated_points = points - center
    rotated_points = np.dot(translated_points, rotation_matrix)
    rotated_points += center

    return rotated_points


def find_line_intersection(curve, point, alpha):
    intersections = []

    # alphaが0の場合、垂直な直線を処理
    if alpha == 0:
        x_vertical = point[0]

        # 曲線の各座標間で垂直な直線との交点を確認
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i + 1]

            # もし垂直な直線のxの値が曲線のxの値の間にあるか確認
            if (x1 <= x_vertical <= x2) or (x2 <= x_vertical <= x1):
                # 交点を求める（線形補間を使用）
                y_intersection = y1 + (y2 - y1) * (x_vertical - x1) / (x2 - x1)
                intersections.append((x_vertical, y_intersection))
    else:
        a = np.tan(np.radians(90 + alpha))
        b = point[1] - a * point[0]

        # 曲線の各座標間で直線との交点を確認
        for i in range(len(curve) - 1):
            x1, y1 = curve[i]
            x2, y2 = curve[i + 1]

            # もし直線のyの値が曲線のyの値の間にあるか確認
            if (a * x1 + b <= y1 and a * x2 + b >= y2) or (
                a * x1 + b >= y1 and a * x2 + b <= y2
            ):
                # 交点を求める（線形補間を使用）
                x_intersection = (x2 * (y1 - b) - x1 * (y2 - b)) / (
                    a * (x2 - x1) - (y2 - y1)
                )
                y_intersection = a * x_intersection + b

                intersections.append((x_intersection, y_intersection))

    intersections = np.array(intersections)

    return intersections


if __name__ == "__main__":
    file_path = cf.Settings.BOOK_PATH
    xw.Book(file_path).set_mock_caller()
    main()
