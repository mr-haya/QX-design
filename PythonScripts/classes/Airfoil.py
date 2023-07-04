import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from config import settings


class Airfoil:
    def __init__(
        self,
        foilname,
        Re_min=None,
        Re_max=None,
        Re_step=None,
        alpha_min=None,
        alpha_max=None,
        alpha_step=None,
    ):
        self.name = foilname
        self.dat = fetch_dat(foilname)
        # 翼型座標を[-1, 1]に正規化
        self.normalized_dat = normalize_dat(self.dat)
        # 翼型座標補間関数を作成
        self.geometry = interpolate.interp1d(
            self.normalized_dat[:, 0],
            self.normalized_dat[:, 1],
            kind="cubic",
            fill_value="extrapolate",
        )
        # 空力係数を補間
        if Re_min is not None:
            self.coefs = coefs_model(
                foilname, Re_min, Re_max, Re_step, alpha_min, alpha_max, alpha_step
            )

    def thickness(self, x):
        return self.geometry([-x]) - self.geometry([x])

    def max_thickness(self):
        at, max_thickness = max(
            [(x, self.thickness(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return max_thickness[0], at

    def camber(self, x):
        return (self.geometry(-x) + self.geometry(x)) / 2

    def max_camber(self):
        at, max_camber = max(
            [(x, self.camber(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        return max_camber, at

    def leading_edge_radius(self):
        return 0.1

    def trairing_edge_angle(self):
        return 0.1

    def point(self, x):
        return self.geometry(x)

    def CL_max(self, Re):
        max_val = -np.inf
        max_alpha = None
        for alpha in np.arange(-5, 20, 0.01):
            val = self.coefs["CL"]([alpha, Re])

            # If this point has a higher value than the current max, update the max
            if val > max_val:
                max_val = val
                max_alpha = alpha

        return max_val[0]

    def L_D_max(self, Re):
        max_val = -np.inf
        max_alpha = None
        for alpha in np.arange(-5, 20, 0.01):
            val = self.coefs["CL"]([alpha, Re]) / self.coefs["CD"]([alpha, Re])

            # If this point has a higher value than the current max, update the max
            if val > max_val:
                max_val = val
                max_alpha = alpha

        return max_val[0], max_alpha

    def CL(self, alpha, Re):
        return self.coefs["CL"]([alpha, Re])[0]

    def L_D(self, alpha, Re):
        return self.coefs["CL"]([alpha, Re])[0] / self.coefs["CD"]([alpha, Re])[0]

    def Cm(self, alpha, Re):
        return self.coefs["Cm"]([alpha, Re])[0]

    def XCp(self, alpha, Re):
        return self.coefs["XCp"]([alpha, Re])[0]

    def Top_Xtr(self, alpha, Re):
        return self.coefs["Top_Xtr"]([alpha, Re])[0]

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

    def length(self, start, end):
        start_index = np.argmin(np.abs(self.normalized_dat[:, 0] - start))
        end_index = np.argmin(np.abs(self.normalized_dat[:, 0] - end))
        length = 0
        for i in range(start_index, end_index):
            length += np.sqrt(
                (self.dat[i + 1][1] - self.dat[i][1]) ** 2
                + (self.dat[i + 1][0] - self.dat[i][0]) ** 2
            )
        return length

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

    def show_l_d(self):
        l_d = self.coefs("CL") / self.coefs("CD")
        X1, Y1 = np.meshgrid(l_d.index, l_d.columns, indexing="ij")
        show_data((X1, Y1, l_d.values))
        plt.show()


# 翼型名からフォルダのパスを作成
def path(foil_name):
    return os.path.join(os.path.dirname(__file__), settings.airfoil_path, foil_name)


# datファイルを取得
def fetch_dat(foil_name):
    file_name = os.path.join(path(foil_name), foil_name + ".dat")
    return np.loadtxt(fname=file_name, dtype="float", skiprows=1)


# datデータをxが-1~1になるよう正規化
def normalize_dat(dat):
    # y座標が初めて負になるインデックスを取得
    first_negative_y_indices = np.where(dat[:, 1] < 0)[0][0]

    # 上側の点データを取得
    upper_side_data = dat[:first_negative_y_indices].copy()

    # x座標を左右反転
    upper_side_data[:, 0] = -upper_side_data[:, 0]

    x = np.concatenate([upper_side_data[:, 0], dat[first_negative_y_indices:][:, 0]])
    y = np.concatenate([upper_side_data[:, 1], dat[first_negative_y_indices:][:, 1]])
    return np.array([x, y]).T


def coefs_model(foilname, alpha_min, alpha_max, alpha_step, Re_min, Re_max, Re_step):
    # レイノルズ数と迎角のリストを生成
    alpha_list = np.around(
        np.arange(alpha_min, alpha_max + alpha_step, alpha_step), decimals=1
    )
    Re_list = np.arange(Re_min, Re_max + Re_step, Re_step)

    # XFLR5の解析結果を取得
    coef_array = xflr5_output(foilname, alpha_list, Re_list)

    # 別の格納用配列
    coefs_model = {}

    i = 0
    for coef_name in settings.coef_index.keys():
        # スライスを取り出し、それをDataFrameに変換
        df = pd.DataFrame(
            coef_array[i], index=alpha_list, columns=Re_list, dtype="float"
        )

        # 欠損値を線形補完
        df = df.interpolate(method="linear", limit_direction="both")

        # 補間関数を作成
        df_interp = interpolate.RegularGridInterpolator(
            (df.index, df.columns),
            df.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # 辞書に追加
        coefs_model[coef_name] = df_interp
        i += 1

    return coefs_model


# xflr5の解析結果を取得
# coef_arrayは3次元行列で、[係数, α, Re]
def xflr5_output(foil_name, alpha_list, Re_list):
    alpha_min = alpha_list[0]
    alpha_step = alpha_list[1] - alpha_list[0]
    alpha_num = len(alpha_list)

    Re_min = Re_list[0]
    Re_step = Re_list[1] - Re_list[0]
    Re_num = len(Re_list)

    # データを格納するためのリスト[係数, α, Re]
    coef_array = np.nan * np.ones((len(settings.coef_index), alpha_num, Re_num))

    # データを読み込む
    for i in range(Re_num):
        file_name = (
            foil_name
            + "_T1_Re"
            + "{:.3f}".format((Re_min + Re_step * i) / 1000000)
            + "_M0.00_N9.0.txt"
        )
        file_path = os.path.join(path(foil_name), file_name)

        # テキストファイルの読み込みとデータの抽出
        with open(file_path, "r") as file:
            lines = file.readlines()
            start_index = 11
            for line in lines[start_index:]:
                values = line.strip().split()

                # 空白の場合はスキップ
                if values == []:
                    continue

                # 値が存在する場合はリストに格納
                alpha_in_line = float(values[0])
                if alpha_in_line in alpha_list:
                    j = 0
                    for coef_index in settings.coef_index.values():
                        extracted_value = float(values[coef_index])
                        alpha_index = round((alpha_in_line - alpha_min) / alpha_step)
                        coef_array[j, alpha_index, i] = extracted_value
                        j += 1
    return coef_array


def show_data(type="wireframe", *args):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for arg in args:
        ax.plot_wireframe(
            arg[0], arg[1], arg[2]
        ) if type == "wireframe" else ax.plot_surface(
            arg[0], arg[1], arg[2]
        ) if type == "surface" else ax.plot(
            arg[0], arg[1], arg[2]
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
