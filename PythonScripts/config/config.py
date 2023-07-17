"""
雑多な設定を記述する
"""
import os

# ファイル名
BOOK_NAME = "QX-XX.xlsm"

# 翼型フォルダの相対パス
AIRFOIL_PATH = os.path.join("..", "..", "Airfoils", "source")

# Xflr5の出力txtファイル内での各係数の列番号
COEF_INDEX = {
    "CL": 1,
    "CD": 2,
    "CDp": 3,
    "Cm": 4,
    "Top_Xtr": 5,
    "Bot_Xtr": 6,
    "Cpmin": 7,
    "Chinge": 8,
    "XCp": 11,
}
# 読み込み開始行
START_INDEX = 11

# LLTの変数
LLT_SPAN_DIV = 120  # 翼弦長の分割数（偶数）
LLT_DAMPING_FACTOR = 0.1  # 循環の更新に使う謎係数．収束は遅くなるが数学的に安定するらしい．
LLT_ITERATION_MAX = 32767 - 1
LLT_ERROR = 10 ^ (-5)  # 誤差
LLT_RE_MAX = 1000000
LLT_RE_MIN = 100000
LLT_ALPHA_MAX = 20
LLT_ALPHA_MIN = -10

import xlwings as xw
import matplotlib.pyplot as plt


def get_file_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "..", BOOK_NAME)


def wb():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return xw.Book(os.path.join(script_dir, "..", "..", BOOK_NAME))


# show_data((X, Y, values))
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
