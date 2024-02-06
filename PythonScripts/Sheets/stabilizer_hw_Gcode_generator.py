"""
DBox安定板のNC熱線用Gコードを出力するプログラム
"""

import numpy as np
from scipy.interpolate import interp1d
import tkinter as tk
import tkinter.filedialog as tkfd
from datetime import datetime


foilname = "NACA 0009"
chord_r = 950  # 翼根コード長[mm]
chord_t = 650  # 翼端コード長[mm]
rate_r = 0.725  # 翼根舵面割合
rate_t = 0.65  # 翼端舵面割合
shear_r = 84  # 翼根剪断中心[mm]
shear_t = 84  # 翼端剪断中心[mm]
span = 800  # スパン[mm]
gap = 35  # 安定板-舵面間隙間[mm]
c_length = 5  # 面取り長さ[mm]

R_xy = 0.7  # 太側片側溶けしろ[mm](XY)
R_uz = 0.7  # 細側片側溶けしろ[mm](UZ)
subject_yheight = 120  # 発泡高さ[mm]　発泡高さの1/2に中心が来る
subject_xorigin = 50  # 熱線X原点-発泡X原点距離[mm]
subject_xmargin = 20  # 発泡X原点-加工開始点X距離[mm]　発泡のガワの厚さ
wait_margin = 10  # 発泡表面から待機点までの距離[mm]
F = 100  # 送り速度[mm/min]
F_run = 500  # 加工開始点までの移動速度[mm/min]
F_notch = 500  # 切り込み送り速度[mm/min]
notch_time = 0  # 線引きでの停止時間[sec]
notch_depth = 1  # 切り込み深さ[mm]
notch_height = 1.5  # 切り込み戻り高さ[mm]
backlash_x = 0.15  # バックラッシュ分オフセット量[mm]
distance_xy2uz = 1070  # 駆動面距離[mm]

distance_xy = (distance_xy2uz - span) / 2  # xy駆動面から翼根設計断面までの距離
distance_uz = (distance_xy2uz - span) / 2  # uz駆動面から翼端設計断面までの距離

subject_xstart = subject_xorigin + subject_xmargin  # 加工開始点X座標
point_wait = np.array(
    [subject_xorigin - wait_margin, subject_yheight / 2]
)  # 出発&到着時待機ポイント
point_start = np.array([subject_xstart, subject_yheight / 2])  # 加工開始ポイント

# 駆動面の大きさに変換
chord_xy = chord_r + (chord_r - chord_t) * (distance_xy / distance_xy2uz)
chord_uz = chord_t + (chord_t - chord_r) * (distance_uz / distance_xy2uz)
rate_xy = rate_r + (rate_r - rate_t) * (distance_xy / distance_xy2uz)
rate_uz = rate_t + (rate_t - rate_r) * (distance_uz / distance_xy2uz)
shear_xy = shear_r + (shear_r - shear_t) * (distance_xy / distance_xy2uz)
shear_uz = shear_t + (shear_t - shear_r) * (distance_uz / distance_xy2uz)

# datデータを取得
dat = np.loadtxt(
    fname=foilname + ".dat",
    dtype="float",
    skiprows=1,
)


# 翼型幾何値参照用クラス
class GeometricalAirfoil:
    def __init__(self, dat, chord_ref=1):
        self.dat = dat.copy()
        self.chord_ref = chord_ref
        # datを0~1に正規化
        xmin = np.amin(self.dat[:, 0])
        xmax = np.amax(self.dat[:, 0])
        self.chord_act = xmax - xmin
        self.dat_norm = (self.dat - xmin) / self.chord_act
        # 規格コード長に合わせて拡大
        self.dat_ref = self.dat_norm * self.chord_ref
        # y座標が初めて負になるインデックスを取得
        first_negative_y_index = np.where(self.dat_norm[:, 1] < 0)[0][0]
        # 上側の点データを取得
        upper_side_data = self.dat_ref[:first_negative_y_index].copy()
        # x座標を左右反転
        upper_side_data[:, 0] = -upper_side_data[:, 0]
        # 結合
        _x = np.concatenate(
            [upper_side_data[:, 0], self.dat_ref[first_negative_y_index:][:, 0]]
        )
        _y = np.concatenate(
            [upper_side_data[:, 1], self.dat_ref[first_negative_y_index:][:, 1]]
        )
        self.dat_extended = np.array([_x, _y]).T
        # モデル作成
        self.interp = interp1d(
            self.dat_extended[:, 0],
            self.dat_extended[:, 1],
            kind="linear",
            fill_value="extrapolate",
        )

    def y(self, x):
        # 任意xにおける翼型のy座標を返す
        return self.interp([x])[0]

    def thickness(self, x):
        # 任意xにおける翼厚を返す
        return self.y(-x) - self.y(x)

    def nvec(self, x):
        # 任意xにおける翼型の法線ベクトルを返す
        delta = 0.000001
        x_elem = self.interp(x) - self.interp(x + delta)
        y_elem = np.sign(x) * delta
        size = np.sqrt(x_elem**2 + y_elem**2)
        return np.array([x_elem, y_elem] / size).T

    def offset_foil(self, offset_base, offset_arr=[]):
        # offset_arr = [[start,end,depth], [start,end,depth], [start,end,depth...
        dat = self.dat_extended.copy()
        depth_arr = np.ones(len(dat)) * offset_base
        if len(offset_arr) != 0:
            for i in range(len(offset_arr)):
                start = np.array(
                    [
                        offset_arr[i, 0] * self.chord_ref,
                        self.y(offset_arr[i, 0] * self.chord_ref),
                    ]
                )
                end = np.array(
                    [
                        offset_arr[i, 1] * self.chord_ref,
                        self.y(offset_arr[i, 1] * self.chord_ref),
                    ]
                )
                idx_start = np.searchsorted(dat[:, 0], start[0])
                idx_end = np.searchsorted(dat[:, 0], end[0])
                # datに挿入
                dat = np.insert(dat, [idx_start, idx_start], [start, start], axis=0)
                dat = np.insert(dat, [idx_end + 2, idx_end + 2], [end, end], axis=0)
                # depth行列を更新
                depth_arr = np.insert(
                    depth_arr, [idx_start, idx_start, idx_end, idx_end], 0
                )
                depth_arr[idx_start] = depth_arr[idx_start - 1]
                depth_arr[idx_start + 1 : idx_end + 3] = offset_arr[i, 2]
                depth_arr[idx_end + 3] = depth_arr[idx_end + 4]

        # オフセット
        move = self.nvec(dat[:, 0]) * depth_arr[:, np.newaxis]
        dat[:, 0] = np.abs(dat[:, 0])
        self.dat_out = dat + move
        return self.dat_out


# 幾何値参照用オブジェクトを作成
foil_xy = GeometricalAirfoil(dat, chord_ref=chord_xy)
foil_uz = GeometricalAirfoil(dat, chord_ref=chord_uz)

# 溶け代分オフセット
dat_xy = foil_xy.offset_foil(-R_xy)
dat_uz = foil_uz.offset_foil(-R_uz)

# 向きを調整（x逆にして舵面分割位置を原点に）
dat_xy[:, 0] = -dat_xy[:, 0] + chord_xy * (1 - rate_xy)
dat_uz[:, 0] = -dat_uz[:, 0] + chord_uz * (1 - rate_uz)

# 安定板の角までおよび角からの軌跡
to_stab_edge_xy = np.array(
    [
        [gap / 2 - R_xy, 0],
        [
            gap / 2 - R_xy,
            foil_xy.thickness(chord_xy * (1 - rate_xy) - gap / 2) / 2 - c_length,
        ],
        [
            gap / 2 + c_length,
            foil_xy.thickness(chord_xy * (1 - rate_xy) - gap / 2 - c_length) / 2 + R_xy,
        ],
    ]
)
to_stab_edge_uz = np.array(
    [
        [gap / 2 - R_uz, 0],
        [
            gap / 2 - R_uz,
            foil_uz.thickness(chord_uz * (1 - rate_uz) - gap / 2) / 2 - c_length,
        ],
        [
            gap / 2 + c_length,
            foil_uz.thickness(chord_uz * (1 - rate_uz) - gap / 2 - c_length) / 2 + R_uz,
        ],
    ]
)

from_stab_edge_xy = to_stab_edge_xy.copy()
from_stab_edge_xy[:, 1] = -from_stab_edge_xy[:, 1]
from_stab_edge_xy = from_stab_edge_xy[::-1]
from_stab_edge_uz = to_stab_edge_uz.copy()
from_stab_edge_uz[:, 1] = -from_stab_edge_uz[:, 1]
from_stab_edge_uz = from_stab_edge_uz[::-1]


# 翼型上部のあるｘ位置からあるｘ位置までを分割して返す関数
def divide_dat(dat, start, end):
    dat_upper = dat[: np.where(dat[:, 1] < 0)[0][0]].copy()
    start_index = np.argmin(np.abs(dat_upper[:, 0] - start))
    if dat_upper[start_index][0] < start:
        start_index += 1
    end_index = np.argmin(np.abs(dat_upper[:, 0] - end))
    if dat_upper[end_index][0] > end:
        end_index -= 1
    return dat_upper[start_index : end_index + 1]


# 翼型を区間ごと分割
# dat_1は安定板の角からせん断中心まで
# dat_2はせん断中心から前縁まで
# dat_3は前縁からせん断中心まで(dat_2を逆さにして逆順にする)
# dat_4はせん断中心から安定板の角まで(dat_1を逆さにして逆順にする)
dat_1_xy = divide_dat(
    dat_xy,
    gap / 2 + c_length,
    shear_xy,
)
dat_1_uz = divide_dat(
    dat_uz,
    gap / 2 + c_length,
    shear_uz,
)
dat_2_xy = divide_dat(
    dat_xy,
    shear_xy,
    chord_xy * (1 - rate_xy) + R_xy,
)
dat_2_uz = divide_dat(
    dat_uz,
    shear_uz,
    chord_uz * (1 - rate_uz) + R_uz,
)
dat_3_xy = dat_2_xy.copy()
dat_3_xy[:, 1] = -dat_3_xy[:, 1]
dat_3_xy = dat_3_xy[::-1]
dat_3_uz = dat_2_uz.copy()
dat_3_uz[:, 1] = -dat_3_uz[:, 1]
dat_3_uz = dat_3_uz[::-1]
dat_4_xy = dat_1_xy.copy()
dat_4_xy[:, 1] = -dat_4_xy[:, 1]
dat_4_xy = dat_4_xy[::-1]
dat_4_uz = dat_1_uz.copy()
dat_4_uz[:, 1] = -dat_4_uz[:, 1]
dat_4_uz = dat_4_uz[::-1]


def resample_curve(points, num_points):
    # pointsは曲線を形成する座標点のリストで、num_pointsは目的の点の数です。
    old_xs, old_ys = zip(*points)  # XとYを分離
    old_xs, old_ys = np.array(old_xs), np.array(old_ys)

    # 線形補間関数の生成
    linear_interp_x = interp1d(np.linspace(0, 1, len(old_xs)), old_xs)
    linear_interp_y = interp1d(np.linspace(0, 1, len(old_ys)), old_ys)

    # 新しいサンプル点を生成
    new_xs = linear_interp_x(np.linspace(0, 1, num_points))
    new_ys = linear_interp_y(np.linspace(0, 1, num_points))

    # 新しい座標点の配列を返す
    return list(zip(new_xs, new_ys))


dat_1_uz = resample_curve(dat_1_uz, dat_1_xy.shape[0])
dat_2_uz = resample_curve(dat_2_uz, dat_2_xy.shape[0])
dat_3_uz = resample_curve(dat_3_uz, dat_3_xy.shape[0])
dat_4_uz = resample_curve(dat_4_uz, dat_4_xy.shape[0])

point_offset = point_start - np.array([gap / 2, 0])  # 翼型座標系のオフセット量

namelist = [
    "to_stab_edge_xy",
    "to_stab_edge_uz",
    "from_stab_edge_xy",
    "from_stab_edge_uz",
    "dat_1_xy",
    "dat_1_uz",
    "dat_2_xy",
    "dat_2_uz",
    "dat_3_xy",
    "dat_3_uz",
    "dat_4_xy",
    "dat_4_uz",
]
for name in namelist:
    # 翼型座標系の座標を加工開始点からの相対座標に変換
    globals()[name] = globals()[name] + point_offset
    # 下半身はバックラッシュ分オフセット
    if name in [
        "from_stab_edge_xy",
        "from_stab_edge_uz",
        "dat_3_xy",
        "dat_3_uz",
        "dat_4_xy",
        "dat_4_uz",
    ]:
        globals()[name] = globals()[name] - np.array([backlash_x, 0])
    # 小数点以下3桁にまるめる
    globals()[name] = [
        [round(globals()[name][i][0], 3), round(globals()[name][i][1], 3)]
        for i in range(len(globals()[name]))
    ]

# 線引きする点の座標
chop_0_xy = [round(subject_xstart - R_xy, 3), round(subject_yheight / 2, 3)]
chop_0_uz = [round(subject_xstart - R_uz, 3), round(subject_yheight / 2, 3)]
chop_1_xy = [
    round(subject_xstart - gap / 2 + shear_xy, 3),
    round(
        subject_yheight / 2
        + foil_xy.thickness(chord_xy * (1 - rate_xy) - shear_xy) / 2
        + R_xy,
        3,
    ),
]
chop_1_uz = [
    round(subject_xstart - gap / 2 + shear_uz, 3),
    round(
        subject_yheight / 2
        + foil_uz.thickness(chord_uz * (1 - rate_uz) - shear_uz) / 2
        + R_uz,
        3,
    ),
]
chop_2_xy = [
    round(subject_xstart - gap / 2 + chord_xy * (1 - rate_xy) + R_xy, 3),
    round(subject_yheight / 2, 3),
]
chop_2_uz = [
    round(subject_xstart - gap / 2 + chord_uz * (1 - rate_uz) + R_uz, 3),
    round(subject_yheight / 2, 3),
]
chop_3_xy = [
    round(subject_xstart - gap / 2 + shear_xy - backlash_x, 3),
    round(
        subject_yheight / 2
        - foil_xy.thickness(chord_xy * (1 - rate_xy) - shear_xy) / 2
        - R_xy,
        3,
    ),
]
chop_3_uz = [
    round(subject_xstart - gap / 2 + shear_uz - backlash_x, 3),
    round(
        subject_yheight / 2
        - foil_uz.thickness(chord_uz * (1 - rate_uz) - shear_uz) / 2
        - R_uz,
        3,
    ),
]


# [[x,y]]の配列をなぞるようにGcodeを返す
def trace(name, path_xy, path_uz, F):
    g = "\n\n(trace " + name + ")"
    if isinstance(path_xy[0], list):
        for i in range(len(path_xy)):
            g += "\nG01 F " + str(F)
            g += (
                " X "
                + str(path_xy[i][0])
                + " Y "
                + str(path_xy[i][1])
                + " U "
                + str(path_uz[i][0])
                + " Z "
                + str(path_uz[i][1])
            )
    else:
        g += "\nG01 F " + str(F)
        g += (
            " X "
            + str(path_xy[0])
            + " Y "
            + str(path_xy[1])
            + " U "
            + str(path_uz[0])
            + " Z "
            + str(path_uz[1])
        )
    return str(g)


# pointを基準にdirection方向に線引きをする direction = "right" or "left" or "up" or "down"
def chop(direction, point_xy, point_uz):
    g = "\n\n(chop " + str(direction) + ")"

    if direction == "right":
        point_xy_bottom = [point_xy[0] + notch_depth, point_xy[1]]
        point_xy_top = [point_xy_bottom[0] - notch_height, point_xy_bottom[1]]
        point_uz_bottom = [point_uz[0] + notch_depth, point_uz[1]]
        point_uz_top = [point_uz_bottom[0] - notch_height, point_uz_bottom[1]]
    elif direction == "left":
        point_xy_bottom = [point_xy[0] - notch_depth, point_xy[1]]
        point_xy_top = [point_xy_bottom[0] + notch_height, point_xy_bottom[1]]
        point_uz_bottom = [point_uz[0] - notch_depth, point_uz[1]]
        point_uz_top = [point_uz_bottom[0] + notch_height, point_uz_bottom[1]]
    elif direction == "up":
        point_xy_bottom = [point_xy[0], point_xy[1] + notch_depth]
        point_xy_top = [point_xy_bottom[0], point_xy_bottom[1] - notch_height]
        point_uz_bottom = [point_uz[0], point_uz[1] + notch_depth]
        point_uz_top = [point_uz_bottom[0], point_uz_bottom[1] - notch_height]
    elif direction == "down":
        point_xy_bottom = [point_xy[0], point_xy[1] - notch_depth]
        point_xy_top = [point_xy_bottom[0], point_xy_bottom[1] + notch_height]
        point_uz_bottom = [point_uz[0], point_uz[1] - notch_depth]
        point_uz_top = [point_uz_bottom[0], point_uz_bottom[1] + notch_height]
    else:
        point_xy_bottom = [point_xy[0], point_xy[1]]
        point_xy_top = [point_xy_bottom[0], point_xy_bottom[1]]
        point_uz_bottom = [point_uz[0], point_uz[1]]
        point_uz_top = [point_uz_bottom[0], point_uz_bottom[1]]

    g += (
        "\nG01"
        + " F "
        + str(F)
        + " X "
        + str(point_xy[0])
        + " Y "
        + str(point_xy[1])
        + " U "
        + str(point_uz[0])
        + " Z "
        + str(point_uz[1])
    )
    g += (
        "\nG01"
        + " F "
        + str(F_notch)
        + " X "
        + str(point_xy_bottom[0])
        + " Y "
        + str(point_xy_bottom[1])
        + " U "
        + str(point_uz_bottom[0])
        + " Z "
        + str(point_uz_bottom[1])
    )
    g += (
        "\nG01"
        + " F "
        + str(F_notch)
        + " X "
        + str(point_xy_top[0])
        + " Y "
        + str(point_xy_top[1])
        + " U "
        + str(point_uz_top[0])
        + " Z "
        + str(point_uz_top[1])
    )
    g += (
        "\nG01"
        + " F "
        + str(F)
        + " X "
        + str(point_xy[0])
        + " Y "
        + str(point_xy[1])
        + " U "
        + str(point_uz[0])
        + " Z "
        + str(point_uz[1])
    )
    return str(g)


"""Gcodeの出力"""
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
file = f"stabilizer_hw_{current_time}.txt"
# file = tkfd.asksaveasfilename(
#     title="stabilizer_hw", filetypes=[("txt", ".txt")], defaultextension="txt"
# )  # 保存先のダイアログを表示
# if file:  # パスが選ばれた時に保存する
with open(file, "w") as f:
    g = "G90 G21 "  # 設定
    g += trace("to_wait_point", point_wait, point_wait, F_run)  # 待機点までの移動
    g += trace("to_start_point", point_start, point_start, F)  # 加工開始点までの移動
    g += chop("right", chop_0_xy, chop_0_uz)  # 右方向に線引き
    g += trace("to_stab_edge", to_stab_edge_xy, to_stab_edge_uz, F)
    g += trace("section1", dat_1_xy, dat_1_uz, F)
    g += chop("down", chop_1_xy, chop_1_uz)  # 下方向に線引き
    g += trace("section2", dat_2_xy, dat_2_uz, F)
    g += chop("left", chop_2_xy, chop_2_uz)  # 左方向に線引き
    g += trace("section3", dat_3_xy, dat_3_uz, F)
    g += chop("up", chop_3_xy, chop_3_uz)  # 上方向に線引き
    g += trace("section4", dat_4_xy, dat_4_uz, F)
    g += trace("from_stab_edge", from_stab_edge_xy, from_stab_edge_uz, F)
    g += trace("to_wait_point", point_wait, point_wait, F)  # 待機点までの移動
    g += trace("to_origin", [0, 0], [0, 0], F_run)  # 原点までの移動
    f.write(g)  # 書き込み
    f.close()  # ファイルを閉じる
