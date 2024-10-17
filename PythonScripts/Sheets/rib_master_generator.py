"""
翼型のデータを読み込み、型紙を出力するプログラム
mode = "print" : 印刷用(DXF)
       "lasercut" : レーザーカット用(DXF)
       "jig" : ジグレーザーカット用(SVG)
       "plot": matplotlibでプロット
"""

import os
import xlwings as xw
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.enums import TextEntityAlignment

import classes.Config as cf
from classes.Geometry import GeometricalAirfoil

mode = "jig"  # "print", "lasercut", "jig"
# "print":紙で印刷する用 番号あり, "lasercut":レーザーカッター用 線の両端のみ描画, "jig":リブ付のジグ用
preview = True  # matplotlibでプレビューを表示するか
all_at_once = False  # 一つの図面、ファイルにまとめるか

protrude_length = 1  # 線引き線の飛び出し長さ

# "lasercut"の場合のパラメータ
peephole_length = 10  # 線引き用に開ける四角穴の一辺長さ（peephole: のぞき穴）

# "jig"の場合のパラメータ
channel_width = 60  # チャンネル材の幅
channel_height = 30  # チャンネル材の高さ
torelance = 0.1  # チャンネル材とのはめあい交差
jig_width = 100  # ジグ全体の幅
jig_height = 45  # ジグの四角部分の高さ
spar_height = 140  # リブ付時の桁中心とチャンネル材下部との高さ差

peephole_length = 5  # lasercutで線引き用に開ける四角穴の一辺長さ（peephole: のぞき穴）

# 出力ファイル名 例：rib_master_print_20210901_123456.dxf
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
if all_at_once:
    file_name = os.path.join(
        cf.Settings.OUTPUTS_PATH, "master", f"rib_master_{mode}_{current_time}.dxf"
    )
else:
    output_dir = os.path.join(
        cf.Settings.OUTPUTS_PATH, "master", f"rib_master_{mode}_{current_time}"
    )
    os.makedirs(output_dir, exist_ok=True)


def main():
    global file_name
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
    balsatip_length = sht.range(cf.Wing.balsatip_length).value
    carbontip_length = sht.range(cf.Wing.carbontip_length).value
    koenzai_length = sht.range(cf.Wing.koenzai_length).value
    ribset_line = np.array(sht.range(cf.Wing.ribset_line).value)
    channel_distance = np.array(sht.range(cf.Wing.channel_distance).value)
    hole_margin = sht.range(cf.Wing.hole_margin).value
    refline_offset = sht.range(cf.Wing.refline_offset).value

    total_rib_num = len(foil1name_arr)

    # 必要な翼型のdatファイルを呼び出して辞書型配列に格納
    dat_dict = {}
    foilnames = np.unique(np.concatenate((foil1name_arr, foil2name_arr)))
    for foilname in foilnames:
        foilpath = os.path.join(cf.Settings.AIRFOIL_PATH, foilname)
        dat = np.loadtxt(
            fname=os.path.join(foilpath, foilname + ".dat"),
            dtype="float",
            skiprows=1,
        )
        dat_dict[foilname] = dat

    if all_at_once:
        # 型紙の作図用空間を作成
        doc = ezdxf.new("R2007", setup=True)
        msp = doc.modelspace()
        doc.layers.new(
            name="Layer",
            dxfattribs={
                "color": 0,
                "lineweight": 50,
            },
        )
    for i, id in enumerate(range(total_rib_num)):  # リブ番号の範囲を指定total_rib
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

        do_ribset = False
        if spar == "0番" or spar == "1番" or spar == "2番" or spar == "3番":
            do_ribset = True
            if spar == "0番":
                ribset_line_num = 0
            elif spar == "1番":
                ribset_line_num = 1
            elif spar == "2番":
                ribset_line_num = 2
            elif spar == "3番":
                ribset_line_num = 3
            ribset_line_offsets = np.ravel(ribset_line[:, ribset_line_num])
            channel_distances = np.ravel(channel_distance[:, ribset_line_num])

        # 描画の基準点
        point_ref = np.array([spar_position * chord, -(100 + i * 200)])

        if not all_at_once:
            # 型紙の作図空間を作成
            doc = ezdxf.new("R2007", setup=True)
            msp = doc.modelspace()
            doc.layers.new(
                name="Layer",
                dxfattribs={
                    "color": 1,
                    "lineweight": 50,
                },
            )
            if mode == "lasercut":
                point_ref = np.array([spar_position * chord + 10, 200])
            elif mode == "jig":
                point_ref = np.array([jig_width / 2, 0])

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
            start_index = nearest_next_index(geo.dat_extended, half_x_start)
            end_index = nearest_next_index(geo.dat_extended, half_x_end)
            dat_out = np.vstack(
                [
                    [half_start],
                    dat_offset[
                        start_index + 1 : end_index + 3
                    ],  # プランクの段差分dat_offsetはdat_extendedより点数が多いのを調整
                    [half_end],
                    [half_start],
                ]
            )
        else:
            dat_out = dat_offset.copy()
        # 桁中心を原点に移動＆迎角だけ回転
        dat_out = rotate_points(dat_out - spar_center, (0, 0), alpha_rib)

        # 作図開始

        if mode == "print" or mode == "lasercut":
            # リブ
            msp.add_lwpolyline(
                dat_out + point_ref,
                format="xy",
                close=True,
                dxfattribs={"layer": "Layer"},
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

            # 後縁円弧
            if not is_half:
                add_TEarc(msp, geo, point_ref, koenzai_length, spar_center, alpha_rib)
                add_TEarc(msp, geo, point_ref, carbontip_length, spar_center, alpha_rib)
                add_TEarc(msp, geo, point_ref, balsatip_length, spar_center, alpha_rib)

            # 桁
            theta = np.linspace(0, 2 * np.pi, 300)
            if diam_x != 0:
                x = (diam_x + hole_margin) / 2 * np.cos(theta)
                y = (diam_z + hole_margin) / 2 * np.sin(theta)
                spar_hole = np.vstack([x, y]).T
                msp.add_lwpolyline(
                    spar_hole + point_ref,
                    format="xy",
                    close=True,
                    dxfattribs={"layer": "Layer"},
                )

        if mode == "print":
            # ダミーライン
            add_line_inside_foil(
                msp,
                dat_out,
                (0, 0),
                90,
                point_ref,
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
                    dxfattribs={"layer": "Layer"},
                )

                # コードライン
                rotated_chordline = rotate_points(
                    np.array([[0, 0], [chord, 0]]) - spar_center, (0, 0), alpha_rib
                )
                msp.add_line(
                    rotated_chordline[0] + point_ref,
                    rotated_chordline[1] + point_ref,
                    dxfattribs={"layer": "Layer"},
                )

            # 桁線
            intersections_center = find_line_intersection(dat_out, (0, 0), 0)
            msp.add_line(
                intersections_center[0] + point_ref,
                intersections_center[1] + point_ref,
                dxfattribs={"layer": "Layer", "linetype": "CENTER"},
            )

            # オフセット線
            if taper == "基準":
                intersections = find_line_intersection(dat_out, (refline_offset, 0), 0)
                msp.add_line(
                    intersections[0] + point_ref,
                    intersections[1] + point_ref,
                    dxfattribs={"layer": "Layer", "linetype": "CENTER"},
                )
            if do_ribset:
                for offset in ribset_line_offsets:
                    add_line_inside_foil(
                        msp,
                        dat_out,
                        (offset, 0),
                        0,
                        point_ref,
                    )

            # テキスト
            label_location = np.array([0.1 * chord - spar_x, 0])
            label_text = str(id)
            if taper == "基準":
                label_text = label_text + " ref"
            if spar == "端リブ":
                label_text = label_text + " end"
            label_height = 15
            info_text = str(np.round(chord * 1e3) / 1e3) + "mm"
            info_height = 10
            msp.add_text(
                label_text,
                height=label_height,
                dxfattribs={
                    "layer": "Layer",
                },
            ).set_placement(
                label_location + point_ref, align=TextEntityAlignment.BOTTOM_LEFT
            )
            msp.add_text(
                info_text,
                height=info_height,
                dxfattribs={
                    "layer": "Layer",
                },
            ).set_placement(
                (label_location[0], label_location[1] - 5) + point_ref,
                align=TextEntityAlignment.TOP_LEFT,
            )

        if mode == "lasercut":
            # ダミーライン
            add_line_inside_foil(
                msp, dat_out, (0, 0), 90, point_ref, 2, 2, peephole=True
            )

            # コードライン
            add_line_inside_foil(
                msp,
                dat_out,
                rotate_points(np.array([0, -spar_center[1]]), (0, 0), alpha_rib),
                90 - alpha_rib,
                point_ref,
                2,
                2,
            )

            # 桁線
            msp.add_line(
                np.array([0, (diam_z + hole_margin) / 2]) + point_ref,
                np.array([0, (diam_z + hole_margin) / 2 + protrude_length]) + point_ref,
                dxfattribs={"layer": "Layer"},
            )
            msp.add_line(
                np.array([0, -(diam_z + hole_margin) / 2]) + point_ref,
                np.array([0, -(diam_z + hole_margin) / 2 - protrude_length])
                + point_ref,
                dxfattribs={"layer": "Layer"},
            )
            msp.add_line(
                np.array([(diam_x + hole_margin) / 2, 0]) + point_ref,
                np.array([(diam_x + hole_margin) / 2 + protrude_length, 0]) + point_ref,
                dxfattribs={"layer": "Layer"},
            )
            msp.add_line(
                np.array([-(diam_x + hole_margin) / 2, 0]) + point_ref,
                np.array([-(diam_x + hole_margin) / 2 - protrude_length, 0])
                + point_ref,
                dxfattribs={"layer": "Layer"},
            )

            # オフセット線
            if do_ribset:
                for offset in ribset_line_offsets:
                    add_line_inside_foil(
                        msp, dat_out, (offset, 0), 0, point_ref, 2, 2, peephole=True
                    )
            # 文字入れ
            text_location = label_location = np.array([0.25 * chord - spar_x, 0])
            text_interval = 7.5
            text_height = 10

            for digit in str(id):
                write_num_with_lines(
                    msp,
                    digit,
                    text_location + point_ref,
                    text_height,
                    text_interval,
                )
                text_location += np.array([text_interval, 0])

        if mode == "jig":
            if all_at_once:
                # リブ
                msp.add_lwpolyline(
                    dat_out + point_ref,
                    format="xy",
                    close=True,
                    dxfattribs={"layer": "Layer"},
                )
            if do_ribset:
                for j, offset in enumerate(ribset_line_offsets):

                    intersection = find_line_intersection(dat_out, (offset, 0), 0)
                    if len(intersection) == 0:
                        continue

                    intersection = intersection[1]

                    dat_section = divide_dat(
                        dat_out,
                        offset - channel_width / 2,
                        offset + channel_width / 2,
                    )
                    if is_half and j == 1:
                        dat_section = np.array(
                            [
                                find_line_intersection(
                                    dat_out, (offset - channel_width / 2, 0), 0
                                )[1],
                                (
                                    find_line_intersection(
                                        dat_out, (offset + channel_width / 2, 0), 0
                                    )[1]
                                    if find_line_intersection(
                                        dat_out, (offset + channel_width / 2, 0), 0
                                    ).size
                                    != 0
                                    else dat_out[-1]
                                ),
                            ]
                        )
                    jig_points = np.vstack(
                        [
                            [
                                [-jig_width / 2, 0],
                                [-jig_width / 2, jig_height],
                            ],
                            dat_section
                            + np.array([-channel_distances[j], spar_height]),
                            [
                                [jig_width / 2, jig_height],
                                [jig_width / 2, 0],
                                [channel_width / 2 + torelance / 2, 0],
                                [channel_width / 2 + torelance / 2, channel_height],
                                [-channel_width / 2 - torelance / 2, channel_height],
                                [-channel_width / 2 - torelance / 2, 0],
                            ],
                        ]
                    )

                    peak_line = np.array(
                        [intersection, intersection + np.array([0, -protrude_length])]
                    ) + np.array([-channel_distances[j], spar_height])

                    space_between = np.array([jig_width * j, 0])

                    if all_at_once:
                        add_line_inside_foil(
                            msp,
                            dat_out,
                            (offset, 0),
                            0,
                            point_ref,
                        )
                        jig_points += np.array([channel_distances[j], -spar_height])
                        peak_line += np.array([channel_distances[j], -spar_height])

                        space_between = np.array([0, 0])

                    msp.add_lwpolyline(
                        jig_points + point_ref + space_between,
                        format="xy",
                        close=True,
                        dxfattribs={"layer": "Layer", "color": 1},
                    )
                    msp.add_line(
                        peak_line[0] + point_ref + space_between,
                        peak_line[1] + point_ref + space_between,
                        dxfattribs={"layer": "Layer", "color": 1},
                    )

                    text_location = np.array([15.0, 35.0])
                    text_interval = 7.5
                    text_height = 5
                    if all_at_once:
                        text_location += np.array([channel_distances[j], -spar_height])
                    if j == 0:
                        color = 1
                        write_num_with_lines(
                            msp,
                            "L",
                            text_location + point_ref + space_between,
                            text_height,
                            color,
                        )
                    else:
                        write_num_with_lines(
                            msp,
                            "T",
                            text_location + point_ref + space_between,
                            text_height,
                            color,
                        )

                    text_location[0] += text_interval

                    for digit in str(id):
                        write_num_with_lines(
                            msp,
                            digit,
                            text_location + point_ref + space_between,
                            text_height,
                            color,
                        )
                        text_location += np.array([text_interval, 0])

        if not all_at_once:
            file_name = os.path.join(output_dir, f"rib_{id}.dxf")
            doc.saveas(file_name)
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)
            fig.savefig(os.path.join(output_dir, f"rib_{id}.png"))
            plt.close()

    if all_at_once:
        doc.saveas(file_name)

    # メッセージを表示して終了
    print("ファイルが保存されました。")

    if preview:
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=True)
        plt.show()


def nearest_next_index(dat, x):
    index = next(
        (i for i, point in enumerate(dat) if point[0] > x),
        None,
    )
    return index


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


# 翼型下部のあるｘ位置からあるｘ位置までを分割して返す関数。もしｘ位置が翼型の範囲外の場合は、範囲内の最も近い点を返す
def divide_dat(dat, start, end):
    dat_lower = dat[np.argmin(dat[:, 0]) :].copy()
    start_index = np.argmin(np.abs(dat_lower[:, 0] - start))
    if dat_lower[start_index][0] < start:
        start_index += 1
    end_index = np.argmin(np.abs(dat_lower[:, 0] - end))
    if dat_lower[end_index][0] > end:
        end_index -= 1
    return dat_lower[start_index : end_index + 1]


def add_line_inside_foil(
    msp, dat, point, alpha, point_ref, inward_length=1, outward_length=1, peephole=False
):
    intersections = find_line_intersection(dat, point, alpha)
    if len(intersections) == 0:
        return
    if mode == "lasercut":
        if len(intersections) >= 2:
            vec = (intersections[1] - intersections[0]) / np.linalg.norm(
                intersections[1] - intersections[0]
            )
            switch = 1  # inwardとoutwardの切り替え
            for intersection in intersections:
                msp.add_line(
                    intersection + switch * inward_length * vec + point_ref,
                    intersection - switch * outward_length * vec + point_ref,
                    dxfattribs={"layer": "Layer"},
                )
                switch = -1

            if peephole:
                t = 0.2
                point_former = t * intersections[0] + (1 - t) * intersections[1]
                point_center = (intersections[0] + intersections[1]) / 2
                point_latter = t * intersections[1] + (1 - t) * intersections[0]

                for point in [point_former, point_center, point_latter]:
                    msp.add_line(
                        point
                        - (peephole_length / 2 + protrude_length) * vec
                        + point_ref,
                        point
                        + (peephole_length / 2 + protrude_length) * vec
                        + point_ref,
                        dxfattribs={"layer": "Layer"},
                    )
                    add_square(
                        msp, point + point_ref, vec, peephole_length, peephole_length
                    )
    else:
        msp.add_line(
            intersections[0] + point_ref,
            intersections[1] + point_ref,
            dxfattribs={"layer": "Layer"},
        )


def add_square(msp, square_center, vec, height, width):  # vecはheight方向の単位ベクトル
    half_height = height / 2
    half_width = width / 2
    top_left = (
        square_center + half_height * vec + half_width * np.array([-vec[1], vec[0]])
    )
    top_right = (
        square_center + half_height * vec - half_width * np.array([-vec[1], vec[0]])
    )
    bottom_left = (
        square_center - half_height * vec + half_width * np.array([-vec[1], vec[0]])
    )
    bottom_right = (
        square_center - half_height * vec - half_width * np.array([-vec[1], vec[0]])
    )

    points = np.array([top_left, top_right, bottom_right, bottom_left, top_left])
    msp.add_lwpolyline(
        points,
        dxfattribs={"layer": "Layer"},
    )


def add_tangedsquare(msp, geo, point_ref, gap, x, width, depth, spar_center, alpha_rib):
    x = x * geo.chord_ref
    nvec = geo.nvec(x)

    square_center = rotate_points(
        np.array([abs(x), geo.y(x)]) + nvec * (gap + depth / 2) - spar_center,
        (0, 0),
        alpha_rib,
    )
    vec = rotate_points(nvec, (0, 0), alpha_rib)

    add_square(msp, square_center + point_ref, vec, depth, width)


def add_TEarc(msp, geo, point_ref, radius, spar_center, alpha_rib):
    TE_center = np.array([geo.chord_ref, 0])
    intersections = find_circle_intersection(geo.dat_out, TE_center, radius)
    rotated_TE_center = rotate_points(TE_center - spar_center, (0, 0), alpha_rib)
    rotated_intersections = rotate_points(
        intersections - spar_center, (0, 0), alpha_rib
    )
    start = rotated_intersections[0]
    start_angle = np.arctan2(
        start[1] - rotated_TE_center[1], start[0] - rotated_TE_center[0]
    )

    end = rotated_intersections[1]
    end_angle = np.arctan2(end[1] - rotated_TE_center[1], end[0] - rotated_TE_center[0])

    if mode == "print":
        msp.add_arc(
            center=rotated_TE_center + point_ref,
            radius=radius,
            start_angle=np.rad2deg(start_angle),
            end_angle=np.rad2deg(end_angle),
            dxfattribs={"layer": "Layer"},
        )
    elif mode == "lasercut":
        start_end = start + protrude_length * np.array(
            [
                -np.sin(start_angle),
                np.cos(start_angle),
            ]
        )
        end_start = end - protrude_length * np.array(
            [
                -np.sin(end_angle),
                np.cos(end_angle),
            ]
        )
        msp.add_line(
            start + point_ref, start_end + point_ref, dxfattribs={"layer": "Layer"}
        )
        msp.add_line(
            end_start + point_ref, end + point_ref, dxfattribs={"layer": "Layer"}
        )


def write_num_with_lines(msp, num, text_location, text_height, color=0):
    # 与えられた数字の外形を座標点の配列で返す関数（数字は0~9）
    if num == "0":
        coords = np.array([[0.4, 2], [0, 2], [0, 0], [0.4, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.6, 0],
                    [1, 0],
                    [1, 2],
                    [0.6, 2],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "1":
        coords = np.array([[0.5, 2], [0.5, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "2":
        coords = np.array([[0, 2], [1, 2], [1, 1], [0, 1], [0, 0], [1, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "3":
        coords = np.array([[0, 2], [1, 2], [1, 1], [0, 1], [1, 1], [1, 0], [0, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "4":
        coords = np.array([[0, 2], [0, 1], [1, 1], [1, 2], [1, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "5":
        coords = np.array([[1, 2], [0, 2], [0, 1], [1, 1], [1, 0], [0, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "6":
        coords = np.array([[1, 2], [0.6, 2]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.4, 2],
                    [0, 2],
                    [0, 0],
                    [0.4, 0],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.6, 0],
                    [1, 0],
                    [1, 1],
                    [0.6, 1],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.4, 1],
                    [0, 1],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "7":
        coords = np.array([[0, 1], [0, 2], [1, 2], [1, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "8":
        coords = np.array([[0.4, 2], [0, 2], [0, 0], [0.4, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.6, 0],
                    [1, 0],
                    [1, 2],
                    [0.6, 2],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )

        coords = (
            np.array(
                [
                    [0, 1],
                    [0.4, 1],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.6, 1],
                    [1, 1],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "9":
        coords = (
            np.array(
                [
                    [1, 1],
                    [0.6, 1],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.4, 1],
                    [0, 1],
                    [0, 2],
                    [0.4, 2],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.6, 2],
                    [1, 2],
                    [1, 0],
                    [0.6, 0],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
        coords = (
            np.array(
                [
                    [0.4, 0],
                    [0, 0],
                ]
            )
            / 2
        )
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "L":
        coords = np.array([[0, 2], [0, 0], [1, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    elif num == "T":
        coords = np.array([[0, 2], [1, 2], [0.5, 2], [0.5, 0]]) / 2
        msp.add_lwpolyline(
            text_height * coords + text_location,
            format="xy",
            close=False,
            dxfattribs={"layer": "Layer", "color": color},
        )
    else:
        raise ValueError("Invalid number")


if __name__ == "__main__":
    file_path = cf.Settings.BOOK_PATH
    xw.Book(file_path).set_mock_caller()
    main()
