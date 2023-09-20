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
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from scipy import interpolate

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
    # DXFエンティティのレイヤーを設定
    # プランクやリブ材段差加工後のリブを作図するためのレイヤー
    doc.layers.new(name="FoilLayer", dxfattribs={"color": 0})
    # 翼型の外形を作図するためのレイヤー
    doc.layers.new(name="RefFoilLayer", dxfattribs={"color": 1})
    # ストリンガーを作図するためのレイヤー
    doc.layers.new(name="StringerLayer", dxfattribs={"color": 2})
    # コードラインやダミーライン、円弧など補助図形を作図するためのレイヤー
    doc.layers.new(name="SubLayer", dxfattribs={"color": 4})
    # 文字を入力するためのレイヤー
    doc.layers.new(name="LetterLayer", dxfattribs={"color": 5})
    point_ref_count = 0

    for id in range(40, 60 + 1):  # リブ番号の範囲を指定
        chord = chordlen_arr[id]
        taper = taper_arr[id]
        diam_z = diam_z_arr[id]
        diam_x = diam_x_arr[id]
        spar_position = spar_position_arr[id]
        alpha_rib = alpha_rib_arr[id]
        point_ref = np.array([-spar_position * chord, -point_ref_count * 150])
        point_ref_count += 1
        is_half = ((-1) ** id - 1) / 2

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

        if is_half:
            # オフセット配列の定義 []
            offset_arr = np.array([-0.75, 0.25, 2, 0])
            # ハーフリブラインを追加
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
            offset_arr = np.array([-0.75, 0.25, 2, 0.45])
        # リブオフセット
        dat_out = offset_foil(geo, offset_arr)
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
            dxfattribs={"layer": "SubLayer"},
        )
        # 桁の十字線追加
        spar_x = spar_position * chord
        spar_center = np.array([abs(spar_x), geo.camber(spar_x)]) + point_ref
        add_cross(msp, spar_center, diam_x, diam_z)
        # ダミーラインを作図
        dummy_center = spar_center - np.array([0, geo.thickness(spar_x) * 0.2])
        dummy_start = dummy_center + np.array([1, np.tan(np.radians(alpha_rib))]) * (
            chord * 0.9 - spar_x
        )
        dummy_end = dummy_center - np.array([1, np.tan(np.radians(alpha_rib))]) * spar_x
        msp.add_line(dummy_start, dummy_end, dxfattribs={"layer": "SubLayer"})

        # 80mmオフセット線を追加
        if taper == "基準":
            refline_offset = 80
            refline_start = np.array(
                [
                    -refline_offset,
                    geo.y(-spar_position * chord + refline_offset),
                ]
            )
            refline_end = np.array(
                [-refline_offset, geo.y(spar_position * chord - refline_offset)]
            )
            msp.add_line(refline_start, refline_end, dxfattribs={"layer": "SubLayer"})

        # 円弧を追加
        add_TEarc(msp, chord, point_ref, 20)
        add_TEarc(msp, chord, point_ref, 40)
        add_TEarc(msp, chord, point_ref, 80)

        # テキストを作成
        label_location = np.array([0.1 * chord, geo.y(-0.1 * chord) - 30]) + point_ref
        if is_half:
            label_text = str(id) + "番" + "ハーフ"
        else:
            label_text = str(id) + "番"
        label_height = 15
        msp.add_text(
            label_text,
            dxfattribs={
                "layer": "TextLayer",
                "height": label_height,
                "insert": label_location,
            },
        )
        info_location = np.array([0.8 * chord, geo.y(0.8 * chord)]) + point_ref
        info_text = str(np.round(chord * 1e3) / 1e3) + "mm"
        info_height = 10
        msp.add_text(
            info_text,
            dxfattribs={
                "layer": "TextLayer",
                "height": info_height,
                "insert": info_location,
            },
        )

    # リブ付け用基準線追加
    setline1_start = np.array([-100, 100])
    setline1_end = np.array([-100, -point_ref_count * 150])
    setline2_start = np.array([150, 100])
    setline2_end = np.array([150, -point_ref_count * 150])
    msp.add_line(setline1_start, setline1_end, dxfattribs={"layer": "SubLayer"})
    msp.add_line(setline2_start, setline2_end, dxfattribs={"layer": "SubLayer"})

    # dxfファイルに保存
    file_name = "rib_master_spar3" + ".dxf"
    doc.saveas(file_name)

    # メッセージを表示して終了
    print("ファイルが保存されました。")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=True)
    plt.show()


def offset_foil(geo, offset_arr):
    # dat_old = dat.copy()
    # chord = np.amax(dat_old[:, 0]) - np.amin(dat_old[:, 0])
    # # 新しい形状のゼロ行列を作成
    # dat_new = np.zeros((dat_old.shape[0] + 4, dat_old.shape[1]))
    # # startの座標を取得
    # coords_start = np.array([abs(start), geo.y(start)]) * chord
    # # endの座標を取得
    # coords_end = np.array([abs(end), geo.y(end)]) * chord
    # # オフセット開始/終了インデックスを取得
    # index_start = nearest_index(dat_old, start)
    # index_end = nearest_index(dat_old, end)
    # # オフセットする配列を取得
    # coords_offset = np.concatenate(
    #     [[coords_start], dat_old[index_start:index_end], [coords_end]]
    # )
    # # 配列に対して、depthだけオフセットした座標を取得
    # coords_offset = coords_offset + geo.nvec(coords_offset[:, 0]) * depth
    # # 座標を格納
    # dat_new = np.concatenate(
    #     [
    #         dat_old[:index_start],
    #         [coords_start],
    #         coords_offset,
    #         [coords_end],
    #         dat_old[index_end:],
    #     ]
    # )
    x = np.linspace(-1, 1, 10000) * geo.chord_ref
    depth_arr = np.ones(10000) * offset_arr[3]
    depth_arr[
        int(0.5 * (offset_arr[0] + 1) * 10000) : int(0.5 * (offset_arr[1] + 1) * 10000)
    ] = offset_arr[2]
    arr = np.array([np.abs(x), geo.y(x)]).T
    arr_new = arr + geo.nvec(x) * depth_arr[:, np.newaxis]
    # print(arr_new)
    return arr_new


def nearest_index(dat, x):
    # datを0~1に正規化
    xmin = np.amin(dat[:, 0])
    xmax = np.amax(dat[:, 0])
    dat_norm = (dat - xmin) / (xmax - xmin)
    # y座標が初めて負になるインデックスを取得
    first_negative_y_index = np.where(dat_norm[:, 1] < 0)[0][0]
    # 上側の点データを取得
    upper_side_data = dat_norm[:first_negative_y_index].copy()
    # x座標を左右反転
    upper_side_data[:, 0] = -upper_side_data[:, 0]
    # 結合
    _x = np.concatenate(
        [upper_side_data[:, 0], dat_norm[first_negative_y_index:][:, 0]]
    )
    # xより右側でxに最も近いインデックスを取得
    if x == 1:
        index = len(dat) + 1
    else:
        index = np.where(_x > x)[0][0]

    return index


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


def add_cross(msp, center, width, height):
    left = center - np.array([width / 2, 0])
    right = center + np.array([width / 2, 0])
    bottom = center - np.array([0, height / 2])
    top = center + np.array([0, height / 2])
    msp.add_line(left, right, dxfattribs={"layer": "SubLayer"})
    msp.add_line(top, bottom, dxfattribs={"layer": "SubLayer"})


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
