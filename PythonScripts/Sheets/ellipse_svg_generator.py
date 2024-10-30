import pandas as pd
import svgwrite
import numpy as np

import os
import xlwings as xw
from datetime import datetime
import matplotlib.pyplot as plt

import classes.Config as cf
from classes.Geometry import GeometricalAirfoil

# DPIの設定
DPI = 96
mm = DPI / 25.4


# 変数
osamu_thickness = 15 * mm
osamu_width = 50 * mm
text_height = 10 * mm

# 出力フォルダ名 例：Outputs/master/ketaana_master_20210901_123456/
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(
    cf.Settings.OUTPUTS_PATH, "master", f"ketaana_master_{current_time}"
)
os.makedirs(output_dir, exist_ok=True)

def main():
    # エクセルのシートを取得
    wb = xw.Book.caller()
    sht = wb.sheets[cf.Wing.name]
    diam_z_arr = sht.range(cf.Wing.diam_z).expand("down").value
    diam_x_arr = sht.range(cf.Wing.diam_x).expand("down").value
    hole_margin = sht.range(cf.Wing.hole_margin).value
    
    total_rib_num = len(diam_z_arr)
    for i, id in enumerate(range(total_rib_num)):  # リブ番号の範囲を指定total_rib
        
        height = (diam_z_arr[id]+hole_margin) * mm
        width = (diam_x_arr[id]+hole_margin) * mm

        frame_height = height + text_height
        frame_width = width + text_height

        # SVGファイルの設定
        dwg = svgwrite.Drawing("output.svg")

        # スタイルの設定
        dwg.add(
            dwg.style(
                ".red { fill:none; stroke:red; stroke-width:3px; } .black { fill:none; stroke:black; stroke-width:3px; }"
            )
        )

        # 楕円を作成
        x = frame_width / 2
        y = frame_height / 2
        center = (x, y)
        dwg.add(dwg.ellipse(center, r=(width, height), class_="red"))
        
        # オサムくんの厚さを作成
        dwg.add(dwg.line((x+width+osamu_thickness, y+osamu_width/2),(x+width+osamu_thickness, y-osamu_width/2),class_="red"))
        dwg.add(dwg.line((x+width+osamu_thickness, y+osamu_width/2),(x+width*np.sqrt(1-(osamu_width/2/height)**2), y+osamu_width/2),class_="red"))
        dwg.add(dwg.line((x+width+osamu_thickness, y-osamu_width/2),(x+width*np.sqrt(1-(osamu_width/2/height)**2), y-osamu_width/2),class_="red"))
        

        # 2つの軸を作成
        axis_x_1_start = (x - frame_width / 2, y)
        axis_x_1_end = (x - width, y)
        axis_x_2_start = (x + width, y)
        axis_x_2_end = (x + frame_width / 2 + osamu_thickness, y)
        axis_y_1_start = (x, y - frame_height / 2)
        axis_y_1_end = (x, y - height)
        axis_y_2_start = (x, y + height)
        axis_y_2_end = (x, y + frame_height / 2)
        dwg.add(dwg.line(axis_x_1_start, axis_x_1_end, class_="black"))
        dwg.add(dwg.line(axis_x_2_start, axis_x_2_end, class_="black"))
        dwg.add(dwg.line(axis_y_1_start, axis_y_1_end, class_="black"))
        dwg.add(dwg.line(axis_y_2_start, axis_y_2_end, class_="black"))

        # 楕円を囲む枠を作成
        dwg.add(
            dwg.line(
                (x - frame_width / 2, y - frame_height / 2),
                (x - frame_width / 2, y + frame_height / 2),
                class_="red",
            )
        )
        dwg.add(
            dwg.line(
                (x - frame_width / 2, y + frame_height / 2),
                (x + frame_width / 2 + osamu_thickness, y + frame_height / 2),
                class_="red",
            )
        )
        dwg.add(
            dwg.line(
                (x + frame_width / 2 + osamu_thickness, y + frame_height / 2),
                (x + frame_width / 2 + osamu_thickness, y - frame_height / 2),
                class_="red",
            )
        )
        dwg.add(
            dwg.line(
                (x + frame_width / 2 + osamu_thickness, y - frame_height / 2),
                (x - frame_width / 2, y - frame_height / 2),
                class_="red",
            )
        )

        # テキストを作成
        text_location = (x - frame_width / 2 + 1, y - frame_height / 2 + text_height + 1)
        dwg.add(dwg.text(str(id), insert=text_location, class_="black", font_size=text_height))

        # SVGファイルに保存
        file_name = os.path.join(output_dir, f"ketaana_{id}.svg")
        dwg.saveas(file_name)

    print("ファイルが保存されました。")
    
if __name__ == "__main__":
    file_path = cf.Settings.BOOK_PATH
    xw.Book(file_path).set_mock_caller()
    main()
