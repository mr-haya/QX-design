import pandas as pd
import svgwrite

# DPIの設定
DPI = 96
mm = DPI / 25.4

# Excelファイルからデータを読み取り
df = pd.read_excel("input.xlsx", header=0, index_col=0)
print(df)

# 変数
text_height = 10 * mm
rib_nums = df.index
heights = df["major_axis"]
widths = df["minor_axis"]

print("rib_nums: ", rib_nums)
print("heights: ", heights)
print("widths: ", widths)


for rib_num in rib_nums:
    width = widths[rib_num] * mm
    height = heights[rib_num] * mm
    text = str(rib_num)

    frame_height = height * 2 + text_height
    frame_width = width * 2 + text_height

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

    # 2つの軸を作成
    axis_x_1_start = (x - frame_width / 2, y)
    axis_x_1_end = (x - width, y)
    axis_x_2_start = (x + width, y)
    axis_x_2_end = (x + frame_width / 2, y)
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
            (x + frame_width / 2, y + frame_height / 2),
            class_="red",
        )
    )
    dwg.add(
        dwg.line(
            (x + frame_width / 2, y + frame_height / 2),
            (x + frame_width / 2, y - frame_height / 2),
            class_="red",
        )
    )
    dwg.add(
        dwg.line(
            (x + frame_width / 2, y - frame_height / 2),
            (x - frame_width / 2, y - frame_height / 2),
            class_="red",
        )
    )

    # テキストを作成
    text_location = (x - frame_width / 2 + 1, y - frame_height / 2 + text_height + 1)
    dwg.add(dwg.text(text, insert=text_location, class_="black", font_size=text_height))

    # SVGファイルに保存
    file_name = text + ".svg"
    dwg.saveas(file_name)

print("ファイルが保存されました。")
