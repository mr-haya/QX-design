"""
翼型のデータを読み込み、dxfデータとして出力するプログラム
ezdxfというモジュールを使用
https://ezdxf.readthedocs.io/en/stable/index.html
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
from datetime import datetime

# 保存先のパス
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = "C:/Users/soyha/OneDrive - Kyushu University/AircraftDesign/QX-design/Outputs/master/rib_master_lasercut_240322/"
# output_path = (
#     cf.Settings.OUTPUTS_PATH + f"\\master\\rib_master_lasercut_{current_time}\\"
# )


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

    plank_thickness = sht.range(cf.Wing.plank_thickness_cell).value
    plank_start = sht.range(cf.Wing.plank_start_cell).value
    plank_end = sht.range(cf.Wing.plank_end_cell).value
    halfline_start = sht.range(cf.Wing.halfline_start_cell).value
    halfline_end = sht.range(cf.Wing.halfline_end_cell).value
    ribzai_thickness = sht.range(cf.Wing.ribzai_thickness_cell).value
    ribset_line = np.array(sht.range(cf.Wing.ribset_line).value)
    print(np.ravel(ribset_line[:, 1]))

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

    # 全てのリブを書き込む用のDXFファイルを作成
    doc_all = ezdxf.new()
    msp_all = doc_all.modelspace()
    doc_all.styles.add("MSゴシック", font="romans")
    # レイヤーを設定
    doc_all.layers.new(
        name="FoilLayer",
        dxfattribs={
            "color": 0,
            "lineweight": 50,
        },
    )
    doc_all.layers.new(
        name="StringerLayer",
        dxfattribs={
            "color": 0,
            "lineweight": 50,
        },
    )
    doc_all.layers.new(
        name="SubLayer",
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
        point_ref = np.array([-spar_position * chord, 0])
        spar = spar_arr[id]
        point_ref_count += 1

        # 個別のリブを書き込む用のDXFファイルを作成
        doc = ezdxf.new()
        msp = doc.modelspace()
        doc.styles.add("MSゴシック", font="romans")
        doc.layers.new(
            name="FoilLayer",
            dxfattribs={
                "color": 1,
                "lineweight": 50,
            },
        )
        doc.layers.new(
            name="StringerLayer",
            dxfattribs={
                "color": 1,
                "lineweight": 50,
            },
        )
        doc.layers.new(
            name="SubLayer",
            dxfattribs={
                "color": 1,
                "lineweight": 50,
            },
        )

        # 翼型のdatデータを取得
        dat_raw = dat_dict[foil1name] * foil1rate + dat_dict[foil2name] * foil2rate
        # 幾何値参照用オブジェクトを作成
        geo = GeometricalAirfoil(dat_raw, chord_ref=chord)

        offset_arr = np.array([[plank_start, plank_end, plank_thickness]])
        if is_half:
            offset_base = (
                0  # プランク以外の部分でのオフセット値（基本的にリブ材の厚み）
            )
            half_x_start = halfline_start * geo.chord_ref
            half_x_end = halfline_end * geo.chord_ref
            half_start = (
                np.array([abs(half_x_start), geo.y(half_x_start)])
                + geo.nvec(half_x_start) * plank_thickness
            )
            half_end = np.array([abs(half_x_end), geo.y(half_x_end)])
            msp.add_line(
                half_start + point_ref,
                half_end + point_ref,
                dxfattribs={"layer": "FoilLayer"},
            )
        else:
            offset_base = ribzai_thickness
        # リブオフセット
        dat_out = geo.offset_foil(offset_base, offset_arr)
        # リブ描写
        add_polygon_with_lines(msp, dat_out + point_ref, close=True)
        # ストリンガー用の長方形を作図
        for stringer in stringer_arr:
            add_tangedsquare(
                msp,
                geo,
                point_ref,
                plank_thickness,
                stringer[0],
                stringer[1],
                stringer[2],
            )
        # コードラインを追加
        msp.add_line(
            np.array([0, 0]) + point_ref,
            np.array([4, 0]) + point_ref,
            dxfattribs={"layer": "SubLayer"},
        )
        msp.add_line(
            np.array([0.45 * chord - 2, 0]) + point_ref,
            np.array([0.46 * chord + 2, 0]) + point_ref,
            dxfattribs={"layer": "SubLayer"},
        )
        # 桁の十字線追加
        spar_x = spar_position * chord
        spar_center = np.array([abs(spar_x), geo.camber(spar_x)]) + point_ref
        add_cross(msp, spar_center, diam_x, diam_z, alpha_rib)

        # ダミーラインを作図
        dummy_center = spar_center - np.array([0, geo.thickness(spar_x) * 0.2])
        add_distanttiltline(
            msp, geo.dat_out + point_ref, dummy_center, 0, 90 + alpha_rib
        )

        # 円弧を追加
        add_TEarc(msp, geo, point_ref, 20)
        add_TEarc(msp, geo, point_ref, 40)
        add_TEarc(msp, geo, point_ref, 80)

        # 80mmオフセット線を追加
        if taper == "基準":
            refline_offset = -80
            add_distanttiltline(
                msp,
                geo.dat_out + point_ref,
                spar_center,
                refline_offset,
                alpha_rib,
            )
        if spar == "端リブ":
            for distance in np.ravel(ribset_line):
                add_distanttiltline(
                    msp,
                    geo.dat_out + point_ref,
                    spar_center,
                    distance,
                    alpha_rib,
                )
        elif spar == "1番":
            for distance in np.ravel(ribset_line[:, 0]):
                add_distanttiltline(
                    msp,
                    geo.dat_ref + point_ref,
                    spar_center,
                    distance,
                    alpha_rib,
                )
        elif spar == "2番":
            for distance in np.ravel(ribset_line[:, 1]):
                add_distanttiltline(
                    msp,
                    geo.dat_ref + point_ref,
                    spar_center,
                    distance,
                    alpha_rib,
                )
        elif spar == "3番":
            for distance in np.ravel(ribset_line[:, 2]):
                add_distanttiltline(
                    msp,
                    geo.dat_ref + point_ref,
                    spar_center,
                    distance,
                    alpha_rib,
                )
        # if spar == "2番":
        #     setline1_offset = -150
        #     setline2_offset = 200
        #     add_distanttiltline(
        #         msp, geo.dat_out + point_ref, spar_center, setline1_offset, alpha_rib
        #     )
        #     add_distanttiltline(
        #         msp, geo.dat_out + point_ref, spar_center, setline2_offset, alpha_rib
        #     )
        # elif spar == "3番":
        #     setline1_offset = -100
        #     setline2_offset = 150
        #     add_distanttiltline(
        #         msp, geo.dat_out + point_ref, spar_center, setline1_offset, alpha_rib
        #     )
        #     add_distanttiltline(
        #         msp, geo.dat_out + point_ref, spar_center, setline2_offset, alpha_rib
        #     )
        # elif spar == "端リブ":
        #     setline1_offset = -100
        #     setline2_offset = 150
        #     add_distanttiltline(
        #         msp, geo.dat_out + point_ref, spar_center, setline1_offset, alpha_rib
        #     )
        #     add_distanttiltline(
        #         msp, geo.dat_out + point_ref, spar_center, setline2_offset, alpha_rib
        #     )

        # dxfファイルに保存
        # file_name = cf.Settings.OUTPUTS_PATH + "\\master\\rib_master_spar3_230930" + ".dxf"
        file_name = output_path + str(id) + ".dxf"

        doc.saveas(file_name)

    # メッセージを表示して終了
    print("ファイルが保存されました。")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    plt.show()


# スプラインではなくポリラインで描画
def add_polygon_with_lines(msp, points, close=True):
    num_points = len(points)
    for i in range(num_points - 1):
        msp.add_line(points[i], points[i + 1], dxfattribs={"layer": "FoilLayer"})
    # If the polygon should be closed, add a line from the last point to the first
    if close:
        msp.add_line(points[-1], points[0], dxfattribs={"layer": "FoilLayer"})


# def offset_foil(geo, offset_arr, offset_base):
#     # offset_arr = [[start,end,depth], [start,end,depth], [start,end,depth...
#     dat = geo.dat_extended.copy()
#     n = len(offset_arr)
#     depth_arr = np.ones(len(dat)) * offset_base
#     for i in range(n):
#         start = np.array(
#             [offset_arr[i, 0] * geo.chord_ref, geo.y(offset_arr[i, 0] * geo.chord_ref)]
#         )
#         end = np.array(
#             [offset_arr[i, 1] * geo.chord_ref, geo.y(offset_arr[i, 1] * geo.chord_ref)]
#         )
#         idx_start = np.searchsorted(dat[:, 0], start[0])
#         idx_end = np.searchsorted(dat[:, 0], end[0])
#         # datに挿入
#         dat = np.insert(dat, [idx_start, idx_start], [start, start], axis=0)
#         dat = np.insert(dat, [idx_end + 2, idx_end + 2], [end, end], axis=0)
#         # depth行列を更新
#         depth_arr = np.insert(depth_arr, [idx_start, idx_start, idx_end, idx_end], 0)
#         depth_arr[idx_start] = depth_arr[idx_start - 1]
#         depth_arr[idx_start + 1 : idx_end + 3] = offset_arr[i, 2]
#         depth_arr[idx_end + 3] = depth_arr[idx_end + 4]

#     # オフセット
#     move = geo.nvec(dat[:, 0]) * depth_arr[:, np.newaxis]
#     dat[:, 0] = np.abs(dat[:, 0])
#     dat = dat + move
#     return dat


def add_tangedsquare(msp, geo, point_ref, gap, x, width, depth):
    x = x * geo.chord_ref
    x_abs = abs(x)
    y = geo.y(x)
    nvec = geo.nvec(x)

    # 接点の計算
    contact_point = np.array([x_abs, y]) + nvec * gap + point_ref

    # 四角形の頂点の計算
    half_width_vector = np.array([-nvec[1], nvec[0]]) * width / 2
    depth_vector = nvec * depth

    # 上下左右の頂点
    top_left = contact_point - half_width_vector + depth_vector
    top_right = contact_point + half_width_vector + depth_vector
    bottom_left = contact_point - half_width_vector
    bottom_right = contact_point + half_width_vector

    msp.add_line(top_left, top_right, dxfattribs={"layer": "StringerLayer"})
    msp.add_line(top_right, bottom_right, dxfattribs={"layer": "StringerLayer"})
    msp.add_line(bottom_right, bottom_left, dxfattribs={"layer": "StringerLayer"})
    msp.add_line(bottom_left, top_left, dxfattribs={"layer": "StringerLayer"})


def rotate_point(point, center, alpha_rad):
    """Rotate a point around a center by alpha_rad radians."""
    # Translate point to origin
    point_translated = point - center

    # Apply rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(alpha_rad), -np.sin(alpha_rad)],
            [np.sin(alpha_rad), np.cos(alpha_rad)],
        ]
    )
    rotated_translated = np.dot(rotation_matrix, point_translated)

    # Translate point back to original location
    rotated_point = rotated_translated + center
    return rotated_point


def add_square(msp, center, width, height, alpha=0):
    alpha_rad = np.radians(alpha)
    dx = width / 2
    dy = height / 2
    corners = np.array(
        [
            [center[0] - dx, center[1] - dy],
            [center[0] + dx, center[1] - dy],
            [center[0] + dx, center[1] + dy],
            [center[0] - dx, center[1] + dy],
        ]
    )
    rotated_corners = [rotate_point(corner, center, alpha_rad) for corner in corners]
    for i in range(4):
        start_point = rotated_corners[i]
        end_point = rotated_corners[(i + 1) % 4]
        msp.add_line(start_point, end_point, dxfattribs={"layer": "SubLayer"})


def add_cross(msp, center, width, height, alpha):
    alpha_rad = np.radians(alpha)
    left_start = center - np.array(
        [
            (width / 2) * np.cos(alpha_rad),
            (width / 2) * np.sin(alpha_rad),
        ]
    )
    left_end = center - np.array(
        [
            (width / 2 - 8) * np.cos(alpha_rad),
            (width / 2 - 8) * np.sin(alpha_rad),
        ]
    )
    right_start = center + np.array(
        [
            (width / 2 - 8) * np.cos(alpha_rad),
            (width / 2 - 8) * np.sin(alpha_rad),
        ]
    )
    right_end = center + np.array(
        [
            (width / 2) * np.cos(alpha_rad),
            (width / 2) * np.sin(alpha_rad),
        ]
    )
    top_start = center + np.array(
        [
            -(height / 2 - 3) * np.sin(alpha_rad),
            (height / 2 - 3) * np.cos(alpha_rad),
        ]
    )
    top_end = center + np.array(
        [
            -(height / 2 - 11) * np.sin(alpha_rad),
            (height / 2 - 11) * np.cos(alpha_rad),
        ]
    )
    bottom_start = center - np.array(
        [
            -(height / 2 - 3) * np.sin(alpha_rad),
            (height / 2 - 3) * np.cos(alpha_rad),
        ]
    )
    bottom_end = center - np.array(
        [
            -(height / 2 - 11) * np.sin(alpha_rad),
            (height / 2 - 11) * np.cos(alpha_rad),
        ]
    )
    msp.add_line(left_start, left_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(right_start, right_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(top_start, top_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(bottom_start, bottom_end, dxfattribs={"layer": "SubLayer"})

    add_square(msp, (left_start + left_end) / 2, 4, 4, alpha)
    add_square(msp, (right_start + right_end) / 2, 4, 4, alpha)
    add_square(msp, (top_start + top_end) / 2, 4, 4, alpha)
    add_square(msp, (bottom_start + bottom_end) / 2, 4, 4, alpha)


def find_line_intersection(curve, point, alpha):
    a = np.tan(np.radians(90 + alpha))
    b = point[1] - a * point[0]

    intersections = []

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


def interpolate_point(P1, P2, t):
    x = (1 - t) * P1[0] + t * P2[0]
    y = (1 - t) * P1[1] + t * P2[1]
    return (x, y)


def add_distanttiltline(msp, dat, center, distance, alpha):
    dist_center = center + np.array(
        [
            distance * np.cos(np.radians(alpha)),
            distance * np.sin(np.radians(alpha)),
        ]
    )
    intersections = find_line_intersection(dat, dist_center, alpha)
    start = intersections[0]
    end = intersections[1]
    start_start = start + np.array(
        [
            2 * -np.sin(np.radians(alpha)),
            2 * np.cos(np.radians(alpha)),
        ]
    )
    start_end = start - np.array(
        [
            2 * -np.sin(np.radians(alpha)),
            2 * np.cos(np.radians(alpha)),
        ]
    )
    end_start = end + np.array(
        [
            2 * -np.sin(np.radians(alpha)),
            2 * np.cos(np.radians(alpha)),
        ]
    )
    end_end = end - np.array(
        [
            2 * -np.sin(np.radians(alpha)),
            2 * np.cos(np.radians(alpha)),
        ]
    )
    msp.add_line(start_start, start_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(end_start, end_end, dxfattribs={"layer": "SubLayer"})
    t = 0.2
    point_1 = t * start + (1 - t) * end
    point_2 = t * end + (1 - t) * start
    point_center = (start + end) / 2
    point_1_start = point_1 + np.array(
        [
            3 * -np.sin(np.radians(alpha)),
            3 * np.cos(np.radians(alpha)),
        ]
    )
    point_1_end = point_1 - np.array(
        [
            3 * -np.sin(np.radians(alpha)),
            3 * np.cos(np.radians(alpha)),
        ]
    )
    point_2_start = point_2 + np.array(
        [
            3 * -np.sin(np.radians(alpha)),
            3 * np.cos(np.radians(alpha)),
        ]
    )
    point_2_end = point_2 - np.array(
        [
            3 * -np.sin(np.radians(alpha)),
            3 * np.cos(np.radians(alpha)),
        ]
    )
    point_center_start = point_center + np.array(
        [
            3 * -np.sin(np.radians(alpha)),
            3 * np.cos(np.radians(alpha)),
        ]
    )
    point_center_end = point_center - np.array(
        [
            3 * -np.sin(np.radians(alpha)),
            3 * np.cos(np.radians(alpha)),
        ]
    )
    msp.add_line(point_1_start, point_1_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(point_2_start, point_2_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(point_center_start, point_center_end, dxfattribs={"layer": "SubLayer"})
    add_square(msp, point_1, 4, 4, alpha)
    add_square(msp, point_2, 4, 4, alpha)
    add_square(msp, point_center, 4, 4, alpha)


def add_TEarc(msp, geo, point_ref, radius):
    TE_center = np.array([geo.chord_ref, 0]) + point_ref
    intersections = find_circle_intersection(geo.dat_out + point_ref, TE_center, radius)
    start = intersections[0]
    start_angle = np.rad2deg(
        np.arctan2(start[1] - TE_center[1], start[0] - TE_center[0])
    )
    end = intersections[1]
    end_angle = np.rad2deg(np.arctan2(end[1] - TE_center[1], end[0] - TE_center[0]))
    start_start = start + np.array(
        [
            0.5 * -np.sin(np.radians(start_angle)),
            0.5 * np.cos(np.radians(start_angle)),
        ]
    )
    start_end = start - np.array(
        [
            0.5 * -np.sin(np.radians(start_angle)),
            0.5 * np.cos(np.radians(start_angle)),
        ]
    )
    end_start = end + np.array(
        [
            0.5 * -np.sin(np.radians(end_angle)),
            0.5 * np.cos(np.radians(end_angle)),
        ]
    )
    end_end = end - np.array(
        [
            0.5 * -np.sin(np.radians(end_angle)),
            0.5 * np.cos(np.radians(end_angle)),
        ]
    )
    msp.add_line(start_start, start_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(end_start, end_end, dxfattribs={"layer": "SubLayer"})

    # msp.add_arc(
    #     center=TE_center,
    #     radius=radius,
    #     start_angle=start_angle - 1,
    #     end_angle=start_angle + 2,
    #     dxfattribs={"layer": "SubLayer"},
    # )
    # msp.add_arc(
    #     center=TE_center,
    #     radius=radius,
    #     start_angle=end_angle - 2,
    #     end_angle=end_angle + 1,
    #     dxfattribs={"layer": "SubLayer"},
    # )


if __name__ == "__main__":
    file_path = cf.Settings.BOOK_PATH
    xw.Book(file_path).set_mock_caller()
    main()
