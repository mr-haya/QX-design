"""
翼型クラス

Attributes:
    name (str): 翼型名
    dat (np.array): 翼型座標datデータ
    geometry (function): 翼型の上半分を左右反転させたdatデータをスプライン曲線で補間した関数。これで翼型の任意xでのy座標を、定義域を-1~1として.geometry([x])で取得できる。
    
    
Methods:
    thickness(x): 任意xにおける翼厚を返す
    camber(x):任意xにおけるキャンバーを返す
    

"""
import xlwings as xw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate


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
        self.interp = interpolate.interp1d(
            self.dat_extended[:, 0],
            self.dat_extended[:, 1],
            kind="linear",
            fill_value="extrapolate",
        )

    def y(self, x):
        return self.interp([x])[0]

    def normalvec(self, x):
        delta = 0.001
        x_elem = self.y(x) - self.y(x + delta)
        y_elem = delta
        size = np.sqrt(x_elem ^ 2 + y_elem ^ 2)
        return np.array([-y_elem / size, x_elem / size])

    def thickness(self, x):
        return self.y(-x) - self.y(x)

    def camber(self, x):
        return (self.y(-x) + self.y(x)) / 2

    def perimeter(self, start, end):
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
        delta = 0.000001
        x_elem = self.interp(x) - self.interp(x + delta)
        y_elem = np.sign(x) * delta
        size = np.sqrt(x_elem**2 + y_elem**2)
        return np.array([x_elem, y_elem] / size).T

    @property
    def tmax(self):
        self.tmax_at, self.thickness_max = max(
            [(x, self.thickness(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return self

    @property
    def tmax_at(self):
        return self

    @property
    def cmax(self):
        self.cmax_at, self.camber_max = max(
            [(x, self.camber(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return self

    @property
    def cmax_at(self):
        return self

    @property
    def curvature(
        self,
        dx_dt,
        dy_dt,
        d2x_dt2,
        d2y_dt2,
    ):
        return np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / np.power(
            dx_dt**2 + dy_dt**2, 3 / 2
        )

    @property
    def area(self):
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

    # 前縁半径を求める（曲率の逆数）
    @property
    def leading_edge_radius(self):
        leading_edge_radius = 1 / self.curvature
        return 1  # leading_edge_radius / self.max_thickness()

    @property
    def trailing_edge_angle(self):
        return 0.1

    def outline(self):
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

    def plot(self):
        dpi = 72
        figsize = (10, 2)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, aspect="equal")
        ax.plot(
            [r[0] for r in self.dat],
            [r[1] for r in self.dat],
            label="original",
            color="black",
        )
        ax.plot(
            [r[0] for r in self.dat_ref],
            [r[1] for r in self.dat_ref],
            label="reference",
            color="red",
        )
        ax.plot(
            [r[0] for r in self.dat_extended],
            [r[1] for r in self.dat_extended],
            label="extended",
            color="blue",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.5, 0.5)
        ax.legend()
        return fig
