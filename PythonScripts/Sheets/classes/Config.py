"""
設定クラス
    - Config
"""

import os
import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt


class Settings:
    # ファイルパス
    script_dir = os.path.dirname(os.path.abspath(__file__))
    BOOK_PATH = os.path.join(
        script_dir, "..", "..", "..", "QX-XX.xlsm"
    )  # 直接実行する場合に必要
    AIRFOIL_PATH = os.path.join(script_dir, "..", "..", "..", "Airfoils", "source")
    OUTPUTS_PATH = os.path.join(
        script_dir, "..", "..", "..", "Outputs"
    )  # 図面などの出力先

    # Xflr5
    XFLR5_COEF_INDEX = {  # 出力txtファイル内での各係数の列番号
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
    XFLR5_START_INDEX = 11  # 読み込み開始行

    # LLTの変数
    LLT_SPAN_DIV = 120  # 翼弦長の分割数（偶数）
    LLT_DAMPING_FACTOR = (
        0.1  # 循環の更新に使う謎係数．収束は遅くなるが数学的に安定するらしい．
    )
    LLT_ITERATION_MAX = 32767 - 1
    LLT_ERROR = 10 ^ (-5)  # 誤差
    LLT_RE_MAX = 1000000
    LLT_RE_MIN = 100000
    LLT_ALPHA_MAX = 20
    LLT_ALPHA_MIN = -10


class Overview:
    # ------------------------ 全機諸元 ------------------------ ##
    SHEET = "全機諸元"

    V = "N25"
    ALPHA = "N26"
    BETA = "N27"
    P = "N28"
    Q = "N29"
    R = "N30"
    dh = "N31"
    de = "N32"
    dr = "N33"
    hE = "N34"
    BALLAST = "N35"

    HAC = "X36"


class weight:
    # ------------------------ 機体重量 ------------------------ ##
    sheet = "重量分布"
    pass


class Wing:
    # -------------------------- 主翼 ------------------------- ##
    name = "主翼（空力）"
    planform = "H6:M13"
    ribzai_thickness = "AG5"
    plank_thickness = "AG7"
    plank_start = "AG11"
    plank_end = "AH11"
    halfline_start = "AG12"
    halfline_end = "AH12"
    hole_margin = "AK11"
    refline_offset = "AK12"
    weight = "BH40"
    ribset_line = "AO4:AQ5"

    # .expand("down")
    stringer = "AB4:AD4"
    spec_rib_cell = "E44:G44"
    margin_rib_cell = "K44"
    export_aero_cell = "L44:AB44"
    export_geometry_cell = "AM44:AW44"

    chordlen = "D51"
    taper = "F51"
    spar = "G51"
    ishalf = "H51"
    diam_z = "I51"
    diam_x = "J51"
    spar_position = "K51"
    foil1name = "O51"
    foil1rate = "P51"
    foil2name = "Q51"
    foil2rate = "R51"
    alpha_rib = "W51"


class Tail:
    # ------------------------ 水平尾翼 ------------------------ ##
    sheet_name = "水平尾翼"
    pass


class Fin:
    # ------------------------ 垂直尾翼 ------------------------ ##
    sheet_name = "垂直尾翼"
    pass


class Cowl:
    # ------------------------- カウル ------------------------- ##
    sheet_name = "カウル"
    pass


class Frame:
    # ------------------------ 胴接構造 ------------------------ ##
    sheet_name = "胴接構造"
    pass


class Spar:
    # -------------------------- 主桁 -------------------------- ##
    sheet_name = "主桁"
    spar_cell = "AH3:AT65"
    spar_yn_cell = "AA6"
    spar_export_cell = "AB6"
    length_0_cell = "F20"
    length_1_cell = "F22"
    length_2_cell = "F24"
    length_3_cell = "F26"
    laminate_0_cell = "B20"
    laminate_1_cell = "B22"
    laminate_2_cell = "B24"
    laminate_3_cell = "B26"
    ellipticity_0_cell = "C20"
    ellipticity_1_cell = "C22"
    ellipticity_2_cell = "C24"
    ellipticity_3_cell = "C26"
    taper_ratio_0_cell = "G20"
    taper_ratio_1_cell = "G22"
    taper_ratio_2_cell = "G24"
    taper_ratio_3_cell = "G26"
    zi_0_cell = "F6"
    zi_1_cell = "J7"
    zi_2_cell = "N9"
    zi_3_cell = "R11"
    spar1_start_cell = "C7"
    spar2_start_cell = "C9"
    spar3_start_cell = "C11"


class Laminate:
    # ------------------------ 積層構成 ------------------------ ##
    sheet_name = "積層構成"
    laminate_cell = "B4:L21"


class Foil:
    # -------------------------- 翼型 -------------------------- ##
    sheet_name = "翼型"
    alpha_min_cell = "C3"
    alpha_max_cell = "D3"
    alpha_step_cell = "E3"
    Re_min_cell = "C4"
    Re_max_cell = "D4"
    Re_step_cell = "E4"
    foil_detail_cell = "B14"  # .expand("table")
    foil_outline_cell = [14, 19]


class Params:
    # ------------------------ パラメータ ----------------------- ##
    sheet_name = "パラメータ"
    rho_cell = "D4"
    g_cell = "D5"
    mu_cell = "D6"
    nu_cell = "D7"
    T_cell = "D8"  # celcius
    P_cell = "D10"  # hPa
    R_cell = "D12"  # 気体定数
    kappa_cell = "D13"  # 比熱比
    a_cell = "D14"  # 音速
    k_cell = "D15"  # カルマン定数
    z0_cell = "D16"  # 粗度長さ
    prepreg_cell = "B54:X57"


class History:
    # ----------------------- 歴代主要諸元 ---------------------- ##
    sheet_name = "歴代主要諸元"
    pass
