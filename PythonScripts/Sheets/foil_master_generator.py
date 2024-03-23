"""
翼型のデータを読み込み、型紙を出力するプログラム
mode = "print" : 印刷用(DXF)
       "lasercut" : レーザーカット用(DXF)
       "jig" : ジグレーザーカット用(SVG)
       "plot": matplotlibでプロット
"""

import os
import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.enums import TextEntityAlignment

import classes.Config as cf
from classes.Geometry import GeometricalAirfoil

mode = "print"  # "print", "lasercut", "jig", "plot"
file_name = r"C:\Users\soyha\OneDrive - Kyushu University\AircraftDesign\QX-design\Outputs\master\rib_master_qx24_240322.dxf"


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
    stringer_arr = sht.range(cf.Wing.stringer).expand("down").value

    ribzai_thickness = sht.range(cf.Wing.ribzai_thickness).value
    plank_thickness = sht.range(cf.Wing.plank_thickness).value
    plank_start = sht.range(cf.Wing.plank_start).value
    plank_end = sht.range(cf.Wing.plank_end).value
    halfline_start = sht.range(cf.Wing.halfline_start).value
    halfline_end = sht.range(cf.Wing.halfline_end).value
    ribset_line = np.array(sht.range(cf.Wing.ribset_line).value)
    hole_margin = sht.range(cf.Wing.hole_margin).value
    refline_offset = sht.range(cf.Wing.refline_offset).value

    total_rib = len(foil1name_arr)

    # 必要なdatファイルを呼び出して格納
    dat_dict = {}
    foilnames = np.unique(np.concatenate((foil1name_arr, foil2name_arr)))

    for foilname in foilnames:
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

    if mode == "print":
        doc = ezdxf.new("R2007", setup=True)
        msp = doc.modelspace()
        doc.styles.add("MSゴシック", font="romans")
        # リブを作図するためのレイヤー
        doc.layers.new(
            name="FoilLayer",
            dxfattribs={
                "color": 0,
                "lineweight": 50,
            },
        )
        # ストリンガーを作図するためのレイヤー
        doc.layers.new(
            name="StringerLayer",
            dxfattribs={
                "color": 0,
                "lineweight": 50,
            },
        )
        # コードラインやダミーライン、円弧など補助図形を作図するためのレイヤー
        doc.layers.new(
            name="SubLayer",
            dxfattribs={
                "color": 0,
                "lineweight": 50,
            },
        )
        # 文字を入力するためのレイヤー
        doc.layers.new(
            name="TextLayer",
            dxfattribs={
                "color": 0,
                "lineweight": 50,
            },
        )

    point_ref_count = 0

    for id in range(total_rib):  # リブ番号の範囲を指定
        chord = chordlen_arr[id]
        taper = taper_arr[id]
        spar = spar_arr[id]
        is_half = ishalf_arr[id]
        diam_z = diam_z_arr[id]
        diam_x = diam_x_arr[id]
        spar_position = spar_position_arr[id]
        foil1name = foil1name_arr[id]
        foil1rate = foil1rate_arr[id]
        foil2name = foil2name_arr[id]
        foil2rate = foil2rate_arr[id]
        alpha_rib = alpha_rib_arr[id]

        # if mode == "print":
        #     point_ref = np.array([-spar_position * chord, -point_ref_count * 200])
        # else:
        #     point_ref = np.array([-spar_position * chord, 0])

        # 翼型のdatデータを取得
        dat_raw = dat_dict[foil1name] * foil1rate + dat_dict[foil2name] * foil2rate
        # 幾何値参照用オブジェクトを作成
        geo = GeometricalAirfoil(dat_raw, chord_ref=chord)
        # プランク情報を取得
        offset_arr = np.array([[plank_start, plank_end, plank_thickness]])
        # 桁中心を計算
        spar_x = spar_position * chord
        spar_center = np.array([spar_x, geo.camber(spar_x)])
        # リブの段差オフセットを定義
        if is_half:
            offset_base = 0
        else:
            offset_base = ribzai_thickness
        offset_arr = np.array(
            [[plank_start, plank_end, plank_thickness]]
        )  # [[xstart, xend, depth],...]
        # リブオフセット
        dat_offset = geo.offset_foil(offset_base, offset_arr).copy()
        # ハーフリブの後ろ側を削除
        if is_half:
            half_x_start = halfline_start * geo.chord_ref
            half_x_end = halfline_end * geo.chord_ref
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

        if mode == "print":
            point_ref = np.array(
                [spar_position * chord, -(100 + point_ref_count * 200)]
            )
            # リブ
            msp.add_lwpolyline(
                dat_out + point_ref,
                format="xy",
                close=True,
                dxfattribs={"layer": "FoilLayer"},
            )
            # ストリンガー
            for stringer in stringer_arr:
                add_tangedsquare(
                    msp,
                    geo,
                    point_ref,
                    plank_thickness,
                    stringer[0],
                    stringer[1],
                    stringer[2],
                    spar_center,
                    alpha_rib,
                )
            if not is_half:
                # 外形
                rotated_outline = rotate_points(
                    geo.dat_ref - spar_center, (0, 0), alpha_rib
                )
                msp.add_lwpolyline(
                    rotated_outline + point_ref,
                    format="xy",
                    close=True,
                    dxfattribs={"layer": "FoilLayer"},
                )
                # コードライン
                rotated_chordline = rotate_points(
                    np.array([[0, 0], [chord, 0]]) - spar_center, (0, 0), alpha_rib
                )
                msp.add_line(
                    rotated_chordline[0] + point_ref,
                    rotated_chordline[1] + point_ref,
                    dxfattribs={"layer": "SubLayer"},
                )
            # 桁線
            intersections_center = find_line_intersection(dat_out, (0, 0), 0)
            msp.add_line(
                intersections_center[0] + point_ref,
                intersections_center[1] + point_ref,
                dxfattribs={"layer": "SubLayer", "linetype": "CENTER"},
            )
            # 桁穴
            if diam_z != 0:
                if diam_z >= diam_x:
                    msp.add_ellipse(
                        center=(0, 0) + point_ref,
                        major_axis=(diam_z / 2, 0),
                        ratio=diam_x / diam_z,
                        dxfattribs={"layer": "SubLayer"},
                    )
                else:
                    msp.add_ellipse(
                        center=(0, 0) + point_ref,
                        major_axis=(0, diam_x / 2),
                        ratio=diam_z / diam_x,
                        dxfattribs={"layer": "SubLayer"},
                    )
            # ダミーライン
            intersections_dummy = find_line_intersection(dat_out, (0, 0), 90)
            msp.add_line(
                intersections_dummy[0] + point_ref,
                intersections_dummy[1] + point_ref,
                dxfattribs={"layer": "SubLayer"},
            )
            # 後縁円弧
            if not is_half:
                add_TEarc(msp, geo, point_ref, 20, spar_center, alpha_rib)
                add_TEarc(msp, geo, point_ref, 40, spar_center, alpha_rib)
                add_TEarc(msp, geo, point_ref, 80, spar_center, alpha_rib)
            # オフセット線
            if taper == "基準":
                refline_offset = -80
                intersections = find_line_intersection(dat_out, (refline_offset, 0), 0)
                msp.add_line(
                    intersections[0] + point_ref,
                    intersections[1] + point_ref,
                    dxfattribs={"layer": "SubLayer", "linetype": "CENTER"},
                )
            if spar == "1番":
                for distance in np.ravel(ribset_line[:, 0]):
                    intersections = find_line_intersection(dat_out, (distance, 0), 0)
                    msp.add_line(
                        intersections[0] + point_ref,
                        intersections[1] + point_ref,
                        dxfattribs={"layer": "SubLayer"},
                    )
            elif spar == "2番":
                for distance in np.ravel(ribset_line[:, 1]):
                    intersections = find_line_intersection(dat_out, (distance, 0), 0)
                    msp.add_line(
                        intersections[0] + point_ref,
                        intersections[1] + point_ref,
                        dxfattribs={"layer": "SubLayer"},
                    )
            elif spar == "3番":
                for distance in np.ravel(ribset_line[:, 2]):
                    intersections = find_line_intersection(dat_out, (distance, 0), 0)
                    msp.add_line(
                        intersections[0] + point_ref,
                        intersections[1] + point_ref,
                        dxfattribs={"layer": "SubLayer"},
                    )
            # テキスト
            label_location = np.array([0.1 * chord - spar_x, 0]) + point_ref
            label_text = str(id)
            if taper == "基準":
                label_text = label_text + " ref"
            if spar == "端リブ":
                label_text = label_text + " end"
            label_height = 15
            msp.add_text(
                label_text,
                height=label_height,
                dxfattribs={
                    "layer": "TextLayer",
                    "style": "MSゴシック",
                },
            ).set_placement(label_location, align=TextEntityAlignment.BOTTOM_LEFT)
            # if is_half:
            #     info_location = (
            #         rotate_points(
            #             np.array([0.65 * chord, geo.y(0.65 * chord)]), (0, 0), alpha_rib
            #         )
            #         + point_ref
            #     )
            #     info_align = TextEntityAlignment.BOTTOM_RIGHT
            # else:
            #     info_location = np.array([0.8 * chord, geo.y(0.8 * chord)]) + point_ref
            #     info_align = TextEntityAlignment.BOTTOM_RIGHT
            info_text = str(np.round(chord * 1e3) / 1e3) + "mm"
            info_height = 10
            msp.add_text(
                info_text,
                height=info_height,
                dxfattribs={
                    "layer": "TextLayer",
                    "style": "MSゴシック",
                },
            ).set_placement(
                (label_location[0], label_location[1] - 5),
                align=TextEntityAlignment.TOP_LEFT,
            )

        point_ref_count += 1

    doc.saveas(file_name)

    # メッセージを表示して終了
    print("ファイルが保存されました。")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    plt.show()


def add_tangedsquare(msp, geo, point_ref, gap, x, width, depth, spar_center, alpha_rib):
    x = x * geo.chord_ref
    x_abs = abs(x)
    y = geo.y(x)
    nvec = geo.nvec(x)

    # 接点の計算
    contact_point = np.array([x_abs, y]) + nvec * gap

    # 四角形の頂点の計算
    half_width_vector = np.array([-nvec[1], nvec[0]]) * width / 2
    depth_vector = nvec * depth

    # 上下左右の頂点
    top_left = contact_point - half_width_vector + depth_vector
    top_right = contact_point + half_width_vector + depth_vector
    bottom_left = contact_point - half_width_vector
    bottom_right = contact_point + half_width_vector

    points = np.array([top_left, top_right, bottom_right, bottom_left, top_left])
    rotated_points = rotate_points(points - spar_center, (0, 0), alpha_rib)

    msp.add_lwpolyline(
        rotated_points + point_ref,
        dxfattribs={"layer": "StringerLayer"},
    )


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

            # 垂直な直線のxの値が曲線のxの値の間にあるか確認
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


# 翼型下部のあるｘ位置からあるｘ位置までを分割して返す関数
# もしｘ位置が翼型の範囲外の場合は、範囲内の最も近い点を返す
def divide_dat(dat, start, end):
    dat_lower = dat[np.argmin(dat[:, 0]) :].copy()
    start_index = np.argmin(np.abs(dat_lower[:, 0] - start))
    if dat_lower[start_index][0] < start:
        start_index += 1
    end_index = np.argmin(np.abs(dat_lower[:, 0] - end))
    if dat_lower[end_index][0] > end:
        end_index -= 1
    return dat_lower[start_index : end_index + 1]


def find_circle_intersection(curve, center, radius):
    intersections = []

    # 曲線の各座標間で円との交点を確認
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]

        # セグメントの端点と円の中心との距離を計算
        d1 = np.sqrt((x1 - center[0]) ** 2 + (y1 - center[1]) ** 2)
        d2 = np.sqrt((x2 - center[0]) ** 2 + (y2 - center[1]) ** 2)

        # 一方の端点が円の内部にあり、もう一方が外部にあるか確認
        if (d1 < radius and d2 > radius) or (d1 > radius and d2 < radius):
            # 交点を求める（線形補間を使用）
            t = (radius - d1) / (d2 - d1)
            x_intersection = (1 - t) * x1 + t * x2
            y_intersection = (1 - t) * y1 + t * y2

            intersections.append((x_intersection, y_intersection))

    intersections = np.array(intersections)
    return intersections


def add_TEarc(msp, geo, point_ref, radius, spar_center, alpha_rib):
    TE_center = np.array([geo.chord_ref, 0])
    intersections = find_circle_intersection(geo.dat_out, TE_center, radius)
    rotated_TE_center = rotate_points(TE_center - spar_center, (0, 0), alpha_rib)
    rotated_intersections = rotate_points(
        intersections - spar_center, (0, 0), alpha_rib
    )
    start = rotated_intersections[0]
    start_angle = np.rad2deg(
        np.arctan2(start[1] - rotated_TE_center[1], start[0] - rotated_TE_center[0])
    )
    end = rotated_intersections[1]
    end_angle = np.rad2deg(
        np.arctan2(end[1] - rotated_TE_center[1], end[0] - rotated_TE_center[0])
    )

    msp.add_arc(
        center=rotated_TE_center + point_ref,
        radius=radius,
        start_angle=start_angle,
        end_angle=end_angle,
        dxfattribs={"layer": "SubLayer"},
    )


if __name__ == "__main__":
    file_path = cf.Settings.BOOK_PATH
    xw.Book(file_path).set_mock_caller()
    main()
