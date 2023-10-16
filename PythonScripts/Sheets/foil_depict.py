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

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf
from geometry.Airfoil import GeometricalAirfoil


def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sht = wb.sheets[sn.wing]

    # シートから値を読み取る
    foil1name_arr = sht.range(ca.foil1name).expand("down").value
    foil1rate_arr = sht.range(ca.foil1rate).expand("down").value
    foil2name_arr = sht.range(ca.foil2name).expand("down").value
    foil2rate_arr = sht.range(ca.foil2rate).expand("down").value
    chordlen_arr = sht.range(ca.chordlen).expand("down").value
    taper_arr = sht.range(ca.taper).expand("down").value
    spar_arr = sht.range(ca.spar).expand("down").value
    ishalf_arr = [item == "half" for item in sht.range(ca.ishalf).expand("down").value]
    diam_z_arr = sht.range(ca.diam_z).expand("down").value
    diam_x_arr = sht.range(ca.diam_x).expand("down").value
    spar_position_arr = sht.range(ca.spar_position).expand("down").value
    alpha_rib_arr = sht.range(ca.alpha_rib).expand("down").value

    # 必要なdatファイルを呼び出して格納
    dat_dict = {}

    for foilname in ["rev_root_140"]:
        # フォルダのパスを取得
        foilpath = os.path.join(os.path.dirname(__file__), cf.AIRFOIL_PATH, foilname)

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

    for id in range(45, 69 + 1):  # リブ番号の範囲を指定
        chord = chordlen_arr[id]
        taper = taper_arr[id]
        diam_z = diam_z_arr[id]
        diam_x = diam_x_arr[id]
        spar_position = spar_position_arr[id]
        alpha_rib = alpha_rib_arr[id]
        point_ref = np.array([-spar_position * chord, -point_ref_count * 150])
        spar = spar_arr[id]
        is_half = ishalf_arr[id]  # ((-1) ** id - 1) / 2
        point_ref_count += 1

        # 翼型のdatデータを取得
        dat_raw = dat_dict["rev_root_140"]
        # 幾何値参照用オブジェクトを作成
        geo = GeometricalAirfoil(dat_raw, chord_ref=chord)
        # 外形リブを描写
        msp.add_lwpolyline(
            geo.dat_ref + point_ref,
            format="xy",
            close=True,
            dxfattribs={"layer": "RefFoilLayer"},
        )

        offset_arr = np.array([[-0.75, 0.25, 2]])
        if is_half:
            offset_base = 0
            half_x_start = -0.75 * geo.chord_ref
            half_x_end = 0.4 * geo.chord_ref
            half_start = (
                np.array([abs(half_x_start), geo.y(half_x_start)])
                + geo.nvec(half_x_start) * 2
            )
            half_end = np.array([abs(half_x_end), geo.y(half_x_end)])
            msp.add_line(
                half_start + point_ref,
                half_end + point_ref,
                dxfattribs={"layer": "FoilLayer"},
            )
        else:
            # オフセット配列の定義 []
            offset_base = 0.45
        # リブオフセット
        dat_out = offset_foil(geo, offset_arr, offset_base)
        # リブ描写
        msp.add_lwpolyline(
            dat_out + point_ref,
            format="xy",
            close=True,
            dxfattribs={"layer": "FoilLayer"},
        )
        # ストリンガー用の長方形を作図
        add_tangedsquare(msp, geo, point_ref, 2, -0.70, 4, 4)
        add_tangedsquare(msp, geo, point_ref, 2, -0.25, 4, 4)
        add_tangedsquare(msp, geo, point_ref, 2, -0.1, 4, 4)
        add_tangedsquare(msp, geo, point_ref, 2, -0.01, 2, 5)
        add_tangedsquare(msp, geo, point_ref, 2, 0.005, 2, 5)
        add_tangedsquare(msp, geo, point_ref, 2, 0.2, 4, 4)
        # コードラインを追加
        msp.add_line(
            np.array([0, 0]) + point_ref,
            np.array([chord, 0]) + point_ref,
            dxfattribs={"layer": "SubLayer", "linetype": "CENTER"},
        )
        # 桁の十字線追加
        spar_x = spar_position * chord
        spar_center = np.array([abs(spar_x), geo.camber(spar_x)]) + point_ref
        add_cross(msp, spar_center, diam_x, diam_z, alpha_rib)
        # ダミーラインを作図
        dummy_center = spar_center - np.array([0, geo.thickness(spar_x) * 0.2])
        add_distanttiltline(
            msp, geo.dat_ref + point_ref, dummy_center, 0, 90 + alpha_rib
        )
        # dummy_start = dummy_center + np.array([1, np.tan(np.radians(alpha_rib))]) * (
        #     chord * 0.9 - spar_x
        # )
        # dummy_end = dummy_center - np.array([1, np.tan(np.radians(alpha_rib))]) * spar_x
        # msp.add_line(dummy_start, dummy_end, dxfattribs={"layer": "SubLayer"})

        # 円弧を追加
        add_TEarc(msp, chord, point_ref, 20)
        add_TEarc(msp, chord, point_ref, 40)
        add_TEarc(msp, chord, point_ref, 80)

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
        if spar == "2番":
            setline1_offset = -150
            setline2_offset = 200
            add_distanttiltline(
                msp, geo.dat_ref + point_ref, spar_center, setline1_offset, alpha_rib
            )
            add_distanttiltline(
                msp, geo.dat_ref + point_ref, spar_center, setline2_offset, alpha_rib
            )
        elif spar == "3番":
            setline1_offset = -100
            setline2_offset = 150
            add_distanttiltline(
                msp, geo.dat_ref + point_ref, spar_center, setline1_offset, alpha_rib
            )
            add_distanttiltline(
                msp, geo.dat_ref + point_ref, spar_center, setline2_offset, alpha_rib
            )

        # テキストを作成
        label_location = np.array([0.1 * chord, geo.camber(-0.1 * chord)]) + point_ref
        if is_half:
            label_text = str(id) + " half"
        else:
            label_text = str(id) + " full"
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
    file_name = cf.get_outputs_path() + "/master/rib_master_spar3_230930" + ".dxf"
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


def add_TEarc(msp, chord, point_ref, radius):
    TE_center = np.array([chord, 0]) + point_ref
    msp.add_arc(
        center=TE_center,
        radius=radius,
        start_angle=162,
        end_angle=175.5,
        dxfattribs={"layer": "SubLayer"},
    )


if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
