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

from scripts import foil_fetch
from config import settings


def coefs(foilname, alpha_min, alpha_max, alpha_step, Re_min, Re_max, Re_step):
    # レイノルズ数と迎角のリストを生成
    alpha_list = np.around(
        np.arange(alpha_min, alpha_max + alpha_step, alpha_step), decimals=1
    )
    Re_list = np.arange(Re_min, Re_max + Re_step, Re_step)

    # XFLR5の解析結果を取得
    # coef_arrayは3次元行列で、[係数, α, Re]
    coef_array = foil_fetch.xflr5_output(foilname, alpha_list, Re_list)
    # 別の格納用配列
    coefs = {}

    i = 0
    for coef_name in settings.coef_index.keys():
        # スライスを取り出し、それをDataFrameに変換
        df = pd.DataFrame(
            coef_array[i], index=alpha_list, columns=Re_list, dtype="float"
        )
        # 欠損値を線形補完
        df = df.interpolate(method="linear", limit_direction="both")

        # 補間関数を作成
        df_interp = RegularGridInterpolator(
            (df.index, df.columns),
            df.values,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # 辞書に追加
        coefs[coef_name] = df_interp
        i += 1

    return coefs


def search_max(coef, Re):
    max_alpha = coef[Re].idxmax()
    max_coef = coef[Re][max_alpha]

    return max_coef, max_alpha


def calcThickness(foil_data):
    return


def calcCamber(foil_data):
    return


def calcWholeLength(foil_data):
    return


def calcArea(foil_data):
    return


if __name__ == "__main__":
    None
