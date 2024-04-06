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

    # 新しいDXFファイルを作成
    doc = ezdxf.new()
    # 作図空間を作成
    msp = doc.modelspace()
    doc.styles.add("MSゴシック", font="romans")
    # DXFエンティティのレイヤーを設定
    # プランクやリブ材段差加工後のリブを作図するためのレイヤー
    doc.layers.new(
        name="FoilLayer",
        dxfattribs={
            "color": 0,
            "lineweight": 50,
        },
    )
    # 翼型の外形を作図するためのレイヤー
    doc.layers.new(
        name="RefFoilLayer",
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

    for id in range(1, total_rib):  # リブ番号の範囲を指定
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
        point_ref = np.array([-spar_position * chord, -point_ref_count * 200])
        point_ref_count += 1

        # 翼型のdatデータを取得
        dat_raw = dat_dict[foil1name] * foil1rate + dat_dict[foil2name] * foil2rate
        # 幾何値参照用オブジェクトを作成
        geo = GeometricalAirfoil(dat_raw, chord_ref=chord)
        # 外形リブを描写
        msp.add_lwpolyline(
            geo.dat_ref + point_ref,
            format="xy",
            close=True,
            dxfattribs={"layer": "RefFoilLayer"},
        )

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
        msp.add_lwpolyline(
            dat_out + point_ref,
            format="xy",
            close=True,
            dxfattribs={"layer": "FoilLayer"},
        )
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
            np.array([chord, 0]) + point_ref,
            dxfattribs={"layer": "SubLayer"},
        )
        # 桁の十字線追加
        spar_x = spar_position * chord
        spar_center = np.array([abs(spar_x), geo.camber(spar_x)]) + point_ref
        add_cross(msp, spar_center, diam_x, diam_z, alpha_rib)
        # 桁穴を追加
        # msp.add_ellipse(
        #     (10, 10),
        #     major_axis=(5, 0),
        #     ratio=0.5,
        #     start_param=0,
        #     end_param=math.pi,
        #     dxfattribs=attribs,
        # )
        # ダミーラインを作図
        dummy_center = spar_center  # - np.array([0, geo.thickness(spar_x) * 0.2])
        add_distanttiltline(
            msp, geo.dat_ref + point_ref, dummy_center, 0, 90 + alpha_rib
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
                geo.dat_ref + point_ref,
                spar_center,
                refline_offset,
                alpha_rib,
            )
        if spar == "端リブ":
            for distance in np.ravel(ribset_line):
                add_distanttiltline(
                    msp,
                    geo.dat_ref + point_ref,
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

        # テキストを作成
        label_location = np.array([0.1 * chord, geo.camber(-0.1 * chord)]) + point_ref
        if is_half:
            label_text = str(id) + " H"
        else:
            label_text = str(id) + " F"
        if taper == "基準":
            label_text = label_text + " ref"
        if spar == "端リブ":
            label_text = label_text + " end"
        label_height = 15
        msp.add_text(
            label_text,
            dxfattribs={
                "layer": "TextLayer",
                "style": "MSゴシック",
                "height": label_height,
                "insert": label_location,
            },
        )
        if is_half:
            info_location = (
                np.array([0.65 * chord, geo.thickness(0.65 * chord) * 0.6]) + point_ref
            )
            info_align = TextEntityAlignment.BOTTOM_RIGHT
        else:
            info_location = np.array([0.8 * chord, geo.y(0.8 * chord)]) + point_ref
            info_align = TextEntityAlignment.BOTTOM_RIGHT
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
            (info_location[0], info_location[1]),
            align=info_align,
        )

    # dxfファイルに保存
    # file_name = cf.Settings.OUTPUTS_PATH + "\\master\\rib_master_spar3_230930" + ".dxf"
    file_name = r"C:\Users\soyha\OneDrive - Kyushu University\AircraftDesign\QX-design\Outputs\master\rib_master_qx24_240320.dxf"
    doc.saveas(file_name)

    # メッセージを表示して終了
    print("ファイルが保存されました。")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    plt.show()


def offset_foil(geo, offset_arr, offset_base):
    # offset_arr = [[start,end,depth], [start,end,depth], [start,end,depth...
    dat = geo.dat_extended.copy()
    n = len(offset_arr)
    depth_arr = np.ones(len(dat)) * offset_base
    for i in range(n):
        start = np.array(
            [offset_arr[i, 0] * geo.chord_ref, geo.y(offset_arr[i, 0] * geo.chord_ref)]
        )
        end = np.array(
            [offset_arr[i, 1] * geo.chord_ref, geo.y(offset_arr[i, 1] * geo.chord_ref)]
        )
        idx_start = np.searchsorted(dat[:, 0], start[0])
        idx_end = np.searchsorted(dat[:, 0], end[0])
        # datに挿入
        dat = np.insert(dat, [idx_start, idx_start], [start, start], axis=0)
        dat = np.insert(dat, [idx_end + 2, idx_end + 2], [end, end], axis=0)
        # depth行列を更新
        depth_arr = np.insert(depth_arr, [idx_start, idx_start, idx_end, idx_end], 0)
        depth_arr[idx_start] = depth_arr[idx_start - 1]
        depth_arr[idx_start + 1 : idx_end + 3] = offset_arr[i, 2]
        depth_arr[idx_end + 3] = depth_arr[idx_end + 4]

    # オフセット
    move = geo.nvec(dat[:, 0]) * depth_arr[:, np.newaxis]
    dat[:, 0] = np.abs(dat[:, 0])
    dat = dat + move
    return dat


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

    msp.add_lwpolyline(
        [top_left, top_right, bottom_right, bottom_left, top_left],
        dxfattribs={"layer": "StringerLayer"},
    )


def add_cross(msp, center, width, height, alpha):
    left = center - np.array(
        [width / 2 * np.cos(np.radians(alpha)), width / 2 * np.sin(np.radians(alpha))]
    )
    right = center + np.array(
        [width / 2 * np.cos(np.radians(alpha)), width / 2 * np.sin(np.radians(alpha))]
    )
    bottom = center - np.array(
        [
            -height / 2 * np.sin(np.radians(alpha)),
            height / 2 * np.cos(np.radians(alpha)),
        ]
    )
    top = center + np.array(
        [
            -height / 2 * np.sin(np.radians(alpha)),
            height / 2 * np.cos(np.radians(alpha)),
        ]
    )
    msp.add_line(left, right, dxfattribs={"layer": "SubLayer"})
    msp.add_line(top, bottom, dxfattribs={"layer": "SubLayer"})


def find_intersection(curve, point, alpha):
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

    return intersections


def add_distanttiltline(msp, dat, center, distance, alpha):
    dist_center = center + np.array(
        [
            distance * np.cos(np.radians(alpha)),
            distance * np.sin(np.radians(alpha)),
        ]
    )
    intersections = find_intersection(dat, dist_center, alpha)
    start = intersections[0]
    end = intersections[1]
    # top = dist_center + np.array(
    #     [
    #         -upperlength * np.sin(np.radians(alpha)),
    #         upperlength * np.cos(np.radians(alpha)),
    #     ]
    # )
    # bottom = dist_center - np.array(
    #     [
    #         -lowerlength * np.sin(np.radians(alpha)),
    #         lowerlength * np.cos(np.radians(alpha)),
    #     ]
    # )
    msp.add_line(start, end, dxfattribs={"layer": "SubLayer"})


# def add_TEarc(msp, chord, point_ref, radius):
#     TE_center = np.array([chord, 0]) + point_ref
#     msp.add_arc(
#         center=TE_center,
#         radius=radius,
#         start_angle=162,
#         end_angle=175.5,
#         dxfattribs={"layer": "SubLayer"},
#     )


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


def add_TEarc(msp, geo, point_ref, radius):
    TE_center = np.array([geo.chord_ref, 0]) + point_ref
    intersections = find_circle_intersection(geo.dat_out + point_ref, TE_center, radius)
    start = intersections[0]
    start_angle = np.rad2deg(
        np.arctan2(start[1] - TE_center[1], start[0] - TE_center[0])
    )
    end = intersections[1]
    end_angle = np.rad2deg(np.arctan2(end[1] - TE_center[1], end[0] - TE_center[0]))
    # start_start = start + np.array(
    #     [
    #         0.5 * -np.sin(np.radians(start_angle)),
    #         0.5 * np.cos(np.radians(start_angle)),
    #     ]
    # )
    # start_end = start - np.array(
    #     [
    #         0.5 * -np.sin(np.radians(start_angle)),
    #         0.5 * np.cos(np.radians(start_angle)),
    #     ]
    # )
    # end_start = end + np.array(
    #     [
    #         0.5 * -np.sin(np.radians(end_angle)),
    #         0.5 * np.cos(np.radians(end_angle)),
    #     ]
    # )
    # end_end = end - np.array(
    #     [
    #         0.5 * -np.sin(np.radians(end_angle)),
    #         0.5 * np.cos(np.radians(end_angle)),
    #     ]
    # )
    # msp.add_line(start_start, start_end, dxfattribs={"layer": "SubLayer"})
    # msp.add_line(end_start, end_end, dxfattribs={"layer": "SubLayer"})

    msp.add_arc(
        center=TE_center,
        radius=radius,
        start_angle=start_angle,
        end_angle=end_angle,
        dxfattribs={"layer": "SubLayer"},
    )


if __name__ == "__main__":
    file_path = cf.Settings.BOOK_PATH
    xw.Book(file_path).set_mock_caller()
    main()
