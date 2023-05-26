"""
翼型の計算にかかわる関数をまとめたモジュール

前提条件
coef_name : CL, CD, CDp, Cm, Top_Xtr, Bot_Xtr, Cpmin, Chinge, XCpを指定可能
coef_df : alphaを一軸目、Reを二軸目としてXFLR5の解析結果を格納し、欠損を線形補完したもの
"""
import os
import xlwings as xw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator

from tools import show_data

# 値読み込み先セル一覧
Re_min_cell = "C4"
Re_max_cell = "D4"
Re_step_cell = "E4"
alpha_min_cell = "C3"
alpha_max_cell = "D3"
alpha_step_cell = "E3"


# 翼型名、Re、alphaからCoefを補間し出力する関数
def coef_interp(foil_name, coef_name, Re, alpha):
    coef_df = make_coef_df(foil_name, coef_name)
    coef_interp = RegularGridInterpolator(
        (coef_df.index, coef_df.columns),
        coef_df.values,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    # Re数をずらして補間データを生成
    # Re_list = Cl_df.columns + 10000
    # X1, Y1 = np.meshgrid(Cl_df.index, Cl_df.columns, indexing="ij")
    # X2, Y2 = np.meshgrid(Cl_df.index, Re_list, indexing="ij")
    # Cl_test = Cl_interp((X2, Y2))
    # show_data((X1, Y1, Cl_df.values), (X2, Y2, Cl_test))
    # plt.show()
    return coef_interp((alpha, Re))


def make_coef_df(foil_name, coef_name):
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sheet = wb.sheets["翼型"]

    # 解析範囲を読み込む
    Re_min = sheet.range(Re_min_cell).value
    Re_max = sheet.range(Re_max_cell).value
    Re_step = sheet.range(Re_step_cell).value
    alpha_min = sheet.range(alpha_min_cell).value
    alpha_max = sheet.range(alpha_max_cell).value
    alpha_step = sheet.range(alpha_step_cell).value

    # 翼型名からフォルダのパスを作成
    foil_path = os.path.join(os.path.dirname(__file__), "../翼型", foil_name)

    # レイノルズ数と迎角のリストを生成
    Re_list = np.arange(Re_min, Re_max + Re_step, Re_step)
    alpha_list = np.around(
        np.arange(alpha_min, alpha_max + alpha_step, alpha_step), decimals=1
    )
    Re_num = len(Re_list)
    alpha_num = len(alpha_list)

    # データを格納するためのリスト
    coef_array = np.nan * np.ones((alpha_num, Re_num))

    # 係数に対応する列番号を取得
    if coef_name == "CL":
        coef_index = 1
    elif coef_name == "CD":
        coef_index = 2
    elif coef_name == "CDp":
        coef_index = 3
    elif coef_name == "Cm":
        coef_index = 4
    elif coef_name == "Top_Xtr":
        coef_index = 5
    elif coef_name == "Bot_Xtr":
        coef_index = 6
    elif coef_name == "Cpmin":
        coef_index = 7
    elif coef_name == "Chinge":
        coef_index = 8
    elif coef_name == "XCp":
        coef_index = 11
    else:
        print("Error: Invalid coef_name")
        return

    # データを読み込む
    for i in range(Re_num):
        file_name = (
            foil_name
            + "_T1_Re"
            + "{:.3f}".format((Re_min + Re_step * i) / 1000000)
            + "_M0.00_N9.0.txt"
        )
        file_path = os.path.join(foil_path, file_name)

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
                    extracted_value = float(values[coef_index])
                    alpha_index = round((alpha_in_line - alpha_min) / alpha_step)
                    coef_array[alpha_index, i] = extracted_value

    # データフレームに変換
    coef_array = pd.DataFrame(
        coef_array, index=alpha_list, columns=Re_list, dtype="float"
    )

    # 欠損値を補間
    coef_array = coef_array.interpolate(method="linear", limit_direction="both")

    return coef_array.copy()


def calcClmax(foil_name):
    cl = make_coef_df(foil_name, "CL")
    max_Re = cl.max().idxmax()
    max_alpha = cl[max_Re].idxmax()
    clmax = cl[max_Re][max_alpha]

    return clmax # , max_Re, max_alpha


def calcClCdmax(foil_name):
    cl = make_coef_df(foil_name, "CL")
    cd = make_coef_df(foil_name, "CD")
    clcd = cl / cd
    max_Re = clcd.max().idxmax()
    max_alpha = clcd[max_Re].idxmax()
    clcdmax = clcd[max_Re][max_alpha]

    return clcdmax # , max_Re, max_alpha


def read_foil(foil_name):
    foil_path = os.path.join(os.path.dirname(__file__), "..", "翼型", foil_name)
    foil_coordinateData_path = os.path.join(foil_path, foil_name + ".dat")
    return np.loadtxt(fname=foil_coordinateData_path, dtype="float", skiprows=1)


def export_outline(foil_name):
    dpi = 72  # 画像の解像度
    figsize = (10, 2)  # 画像のサイズ

    foil_data = read_foil(foil_name)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot([r[0] for r in foil_data], [r[1] for r in foil_data], label="original")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig


def calcThickness(foil_data):
    return


def calcCamber(foil_data):
    return


def calcWholeLength(foil_data):
    return


def calcArea(foil_data):
    return


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "QX-23_Eclipse_v1.xlsm"
    file_path = os.path.join(script_dir, "..", file_name)

    foil_name = "QX0023"

    xw.Book(file_path).set_mock_caller()
    cl = make_coef_df(foil_name, "CL")
    cd = make_coef_df(foil_name, "CD")
    clcd = cl / cd
    X1, Y1 = np.meshgrid(clcd.index, clcd.columns, indexing="ij")
    # X2, Y2 = np.meshgrid(Cl_df.index, Re_list, indexing="ij")
    # Cl_test = Cl_interp((X2, Y2))
    show_data((X1, Y1, clcd.values))
    plt.show()
