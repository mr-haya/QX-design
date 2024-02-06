"""
幾何クラス

- GeometricalAirfoil
    

"""
import xlwings as xw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


class GeometricalAirfoil:
    """
    翼型の幾何特性を表すクラス

    Parameters
    ----------
    dat : numpy.ndarray
        翼型の座標データ
    chord_ref : float
        規格コード長

    Attributes
    ----------
    dat : numpy.ndarray
        翼型の座標データ（正規化されてなくてもよい）
    chord_ref : float
        規格コード長
    dat_norm : numpy.ndarray
        翼型の座標データを0~1に正規化したもの
    dat_ref : numpy.ndarray
        翼型の座標データを規格コード長に合わせて拡大したもの
    dat_extended : numpy.ndarray
        翼型の上半分を左右反転させたdatデータ
    interp : scipy.interpolate.interpolate.interp1d
        dat_extendedをスプライン曲線で補間した関数。
        翼型の任意xでのy座標を、定義域を-chord_ref~chord_refとして.interp([x])で取得できる。
    """

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
        self.interp = interpolate.interp1d(
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

    def camber(self, x):
        # 任意xにおけるキャンバーを返す
        return (self.y(-x) + self.y(x)) / 2

    def perimeter(self, start, end):
        # startからendまでの周囲長を返す
        start_index = np.argmin(np.abs(self.normalized_dat[:, 0] - start))
        end_index = np.argmin(np.abs(self.normalized_dat[:, 0] - end))
        perimeter = 0
        for i in range(start_index, end_index):
            perimeter += np.sqrt(
                (self.dat[i + 1][1] - self.dat[i][1]) ** 2
                + (self.dat[i + 1][0] - self.dat[i][0]) ** 2
            )
        return perimeter

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

    @property
    def tmax(self):
        # 最大翼厚
        self.tmax_at, self.thickness_max = max(
            [(x, self.thickness(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return self

    @property
    def tmax_at(self):
        # 最大翼厚位置
        return self

    @property
    def cmax(self):
        # 最大キャンバー
        self.cmax_at, self.camber_max = max(
            [(x, self.camber(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return self

    @property
    def cmax_at(self):
        # 最大キャンバー位置
        return self

    @property
    def curvature(
        self,
        dx_dt,
        dy_dt,
        d2x_dt2,
        d2y_dt2,
    ):
        # 任意xにおける翼型の曲率を返す
        return np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / np.power(
            dx_dt**2 + dy_dt**2, 3 / 2
        )

    @property
    def area(self):
        # 面積
        area = 0
        for i in range(int(len(self.dat) / 2) - 1):
            area += (
                (
                    (self.dat[i + 1][1] - self.dat[-(i + 1)][1])
                    + (self.dat[i][1] - self.dat[-i][1])
                )
                * (self.dat[i][0] - self.dat[i + 1][0])
                / 2
            )
        return area

    @property
    def leading_edge_radius(self):
        # 前縁半径（曲率の逆数）
        leading_edge_radius = 1 / self.curvature
        return 1  # leading_edge_radius / self.max_thickness()

    @property
    def trailing_edge_angle(self):
        # 後縁角
        return 0.1

    def outline(self):
        # 翼型の輪郭をfigで返す
        dpi = 72  # 画像の解像度
        figsize = (10, 2)  # 画像のサイズ
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, aspect="equal")
        ax.plot([r[0] for r in self.dat], [r[1] for r in self.dat], label="original")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        return fig
