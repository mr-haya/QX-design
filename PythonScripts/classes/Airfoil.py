"""
翼型クラス

Attributes:
    name (str): 翼型名
    dat (np.array): 翼型座標datデータ
    geometry (function): 翼型の上半分を左右反転させたdatデータをスプライン曲線で補間した関数。これで翼型の任意xでのy座標を、定義域を-1~1として.geometry([x])で取得できる。
    coefs (dict): XFLR5で出力したtxtデータから補間した翼型の空力係数モデル。あるRe、alphaにおけるCLは.coefs["CL"]([alpha, Re])[0]で取得できる。配列で取得するには、
                        alpha = np.linspace(-10, 20, 100)  # 迎角
                        Re = np.linspace(1e6, 1e5, 100)  # レイノルズ数
                        X1, Y1 = np.meshgrid(alpha, Re, indexing="ij")
                        CL = np.array(foil.coefs["CL"]((X1, Y1)))
                    のようにする。
    
    
Methods:
    thickness(x): 任意xにおける翼厚を返す
    camber(x):任意xにおけるキャンバーを返す
    

"""

import os
import xlwings as xw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf


class Airfoil:
    def __init__(self, foilname):
        self.name = foilname

        # フォルダのパスを取得
        self.path = os.path.join(os.path.dirname(__file__), cf.AIRFOIL_PATH, self.name)

        # datデータを取得
        self.dat = np.loadtxt(
            fname=os.path.join(self.path, self.name + ".dat"),
            dtype="float",
            skiprows=1,
        )
        # datデータをxが-1~1になるよう正規化
        self.normalized_dat = self._normalize_dat()

        # エクセルシートを取得
        wb = xw.Book.caller()
        sheet = wb.sheets[sn.foil]

        # 解析範囲を読み込む
        self.alpha_min = sheet.range(ca.alpha_min_cell).value
        self.alpha_max = sheet.range(ca.alpha_max_cell).value
        self.alpha_step = sheet.range(ca.alpha_step_cell).value
        self.Re_min = sheet.range(ca.Re_min_cell).value
        self.Re_max = sheet.range(ca.Re_max_cell).value
        self.Re_step = sheet.range(ca.Re_step_cell).value
        self.xflr5 = self._xflr5()
        self.coefs = self._coefs()
        self.geometry = interpolate.interp1d(
            self.normalized_dat[:, 0],
            self.normalized_dat[:, 1],
            kind="cubic",
            fill_value="extrapolate",
        )
        self.tmax_at, self.thickness_max = max(
            [(x, self.thickness(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        self.cmax_at, self.camber_max = max(
            [(x, self.camber(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        # # 曲率を計算
        # curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / np.power(
        #     dx_dt**2 + dy_dt**2, 3 / 2
        # )

        # # 最前部の曲率を求める
        # leading_edge_curvature = np.min(curvature)

        # # 前縁半径を求める（曲率の逆数）
        # leading_edge_radius = 1 / leading_edge_curvature
        self.leading_edge_radius = 1  # leading_edge_radius / self.max_thickness()
        self.trailing_edge_angle = 0.1
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
        self.area = area

    def y(self, x):
        return self.geometry([x])[0]

    def thickness(self, x):
        return self.y(-x) - self.y(x)

    def camber(self, x):
        return (self.y(-x) + self.y(x)) / 2

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

    def CD(self, alpha, Re):
        return self.coefs["CD"]([alpha, Re])[0]

    def CDp(self, alpha, Re):
        return self.coefs["CDp"]([alpha, Re])[0]

    def L_D(self, alpha, Re):
        return self.coefs["CL"]([alpha, Re])[0] / self.coefs["CD"]([alpha, Re])[0]

    def Cm(self, alpha, Re):
        return self.coefs["Cm"]([alpha, Re])[0]

    def XCp(self, alpha, Re):
        return self.coefs["XCp"]([alpha, Re])[0]

    def Top_Xtr(self, alpha, Re):
        return self.coefs["Top_Xtr"]([alpha, Re])[0]

    def Bot_Xtr(self, alpha, Re):
        return self.coefs["Bot_Xtr"]([alpha, Re])[0]

    def Cpmin(self, alpha, Re):
        return self.coefs["Cpmin"]([alpha, Re])[0]

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

    # datデータをxが-1~1になるよう正規化
    def _normalize_dat(self):
        # y座標が初めて負になるインデックスを取得
        first_negative_y_indices = np.where(self.dat[:, 1] < 0)[0][0]

        # 上側の点データを取得
        upper_side_data = self.dat[:first_negative_y_indices].copy()

        # x座標を左右反転
        upper_side_data[:, 0] = -upper_side_data[:, 0]

        x = np.concatenate(
            [upper_side_data[:, 0], self.dat[first_negative_y_indices:][:, 0]]
        )
        y = np.concatenate(
            [upper_side_data[:, 1], self.dat[first_negative_y_indices:][:, 1]]
        )
        return np.array([x, y]).T

    def _coefs(self):
        coefs_model = {}
        for i, coef_name in enumerate(cf.COEF_INDEX.keys()):
            # スライスを取り出し、DataFrameに変換
            df = pd.DataFrame(
                self.xflr5[i],
                index=self.alpha_list,
                columns=self.Re_list,
                dtype="float",
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

        return coefs_model

    def alpha_index(self, alpha):
        return round((alpha - self.alpha_min) / self.alpha_step)

    def Re_index(self, Re):
        return round((Re - self.Re_min) / self.Re_step)

    # xflr5の解析結果を取得[cf.COEF_INDEX["係数"], α, Re]
    def _xflr5(self):
        foil_name = str(self.name)

        # レイノルズ数と迎角のリストを生成
        self.alpha_list = np.around(
            np.arange(
                self.alpha_min, self.alpha_max + self.alpha_step, self.alpha_step
            ),
            decimals=1,
        )
        self.Re_list = np.arange(self.Re_min, self.Re_max + self.Re_step, self.Re_step)
        alpha_num = len(self.alpha_list)
        Re_num = len(self.Re_list)

        # データを格納するためのリスト[係数, α, Re]
        output = np.nan * np.ones((len(cf.COEF_INDEX), alpha_num, Re_num))

        # データを読み込む
        for i in range(Re_num):
            file_name = (
                foil_name
                + "_T1_Re"
                + "{:.3f}".format((self.Re_min + self.Re_step * i) / 1000000)
                + "_M0.00_N9.0.txt"
            )
            file_path = os.path.join(self.path, file_name)

            # テキストファイルの読み込みとデータの抽出
            with open(file_path, "r") as file:
                lines = file.readlines()
                for line in lines[cf.START_INDEX :]:
                    values = line.strip().split()

                    # 空白の場合はスキップ
                    if values == []:
                        continue

                    # 値が存在する場合はリストに格納
                    alpha_in_line = float(values[0])
                    if alpha_in_line in self.alpha_list:
                        for j, coef_index in enumerate(cf.COEF_INDEX.values()):
                            extracted_value = float(values[coef_index])
                            output[
                                j, self.alpha_index(alpha_in_line), i
                            ] = extracted_value
        return output


class MixedAirfoil(Airfoil):  # とりあえず幾何形状とCL,CD,Cmのみを混ぜる
    def __init__(self, airfoil1, airfoil2, ratio):  # ratioはairfoil2の割合
        self.airfoil1 = airfoil1
        self.airfoil2 = airfoil2
        self.ratio = ratio
        self.dat = np.array(
            [
                [
                    self.airfoil1.dat[i][0] * (1 - ratio)
                    + self.airfoil2.dat[i][0] * ratio,
                    self.airfoil1.dat[i][1] * (1 - ratio)
                    + self.airfoil2.dat[i][1] * ratio,
                ]
                for i in range(len(airfoil1.dat))
            ]
        )
        self.normalized_dat = np.array(
            [
                [
                    self.airfoil1.normalized_dat[i][0] * (1 - ratio)
                    + self.airfoil2.normalized_dat[i][0] * ratio,
                    self.airfoil1.normalized_dat[i][1] * (1 - ratio)
                    + self.airfoil2.normalized_dat[i][1] * ratio,
                ]
                for i in range(len(self.airfoil1.normalized_dat))
            ]
        )
        self.geometry = interpolate.interp1d(
            self.normalized_dat[:, 0],
            self.normalized_dat[:, 1],
            kind="cubic",
            fill_value="extrapolate",
        )

        self.tmax_at, self.thickness_max = max(
            [(x, self.thickness(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        self.cmax_at, self.camber_max = max(
            [(x, self.camber(x)) for x in np.arange(0, 1, 0.001)], key=lambda x: x[1]
        )
        self.leading_edge_radius = (
            self.airfoil1.leading_edge_radius * (1 - ratio)
            + self.airfoil2.leading_edge_radius * ratio
        )
        self.trailing_edge_angle = (
            self.airfoil1.trailing_edge_angle * (1 - ratio)
            + self.airfoil2.trailing_edge_angle * ratio
        )
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
        self.area = area

    def CL(self, alpha, Re):
        return (
            self.airfoil1.CL(alpha, Re) * (1 - self.ratio)
            + self.airfoil2.CL(alpha, Re) * self.ratio
        )

    def CD(self, alpha, Re):
        return (
            self.airfoil1.CD(alpha, Re) * (1 - self.ratio)
            + self.airfoil2.CD(alpha, Re) * self.ratio
        )

    def Cm(self, alpha, Re):
        return (
            self.airfoil1.Cm(alpha, Re) * (1 - self.ratio)
            + self.airfoil2.Cm(alpha, Re) * self.ratio
        )
