"""
便利な関数をまとめたモジュール

show_data(type="wireframe",*args)
    引数に与えられたデータを3Dプロットする
    type="wireframe"でワイヤーフレーム、type="surface"でサーフェス、type指定なしでプロット
    *argsには、(X, Y, Z)の形式のタプルを与える
    例: show_data(type="surface", (X1, Y1, Z1), (X2, Y2, Z2))

get_os_type()
    OSの種類を返す
    返り値は"windows"か"mac"

get_onedrive_path()
    ワークシートが保存されているOneDriveのパスを返す
    ワークシートがOneDriveでない場合はローカルパスとしてそのまま返す
    返り値は辞書型で、"commercial"と"consumer"のキーを持つ
    例: onedrive_path = get_onedrive_path()
        commercial_path = onedrive_path["commercial"]
        consumer_path = onedrive_path["consumer"]

onedrive_url_to_local_path(url):
    OneDriveのパスをローカルパスに変換する
    OneDriveでない場合はそのまま返す
    返り値はローカルパス
    例: onedrive_path = onedrive_url_to_local_path("https://onedrive.live.com/xxx")
        
"""
import xlwings as xw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import config


# %%
import os
import platform
import urllib.parse


def get_os_type():
    system = platform.system().lower()
    if "windows" in system:
        return "windows"
    if "darwin" in system:
        return "mac"


def get_onedrive_path():
    if get_os_type() == "windows":
        return {
            "commercial": os.getenv("OneDriveCommercial", os.getenv("OneDrive")),
            "consumer": os.getenv("OneDriveConsumer", os.getenv("OneDrive")),
        }
    elif get_os_type() == "mac":
        return {
            "commercial": config.ONEDRIVE_PATH,
            "consumer": config.ONEDRIVE_PATH,
        }


def onedrive_url_to_local_path(url):
    OneDriveCommercialUrlPattern = "*my.sharepoint.com*"
    onedrive_path = get_onedrive_path()

    if not url.startswith("https://"):
        return url

    if OneDriveCommercialUrlPattern in url:
        FilePathPos = url.index("/Documents") + 10
        return os.path.join(
            onedrive_path["commercial"], url[FilePathPos:].replace("/", os.path.sep)
        )
    else:
        FilePathPos = url[9:].index("/") + 10
        FilePathPos = url[FilePathPos + 1 :].index("/") + FilePathPos + 2

        if FilePathPos == -1:
            return onedrive_path["consumer"]
        else:
            return os.path.join(
                onedrive_path["consumer"], url[FilePathPos:].replace("/", os.path.sep)
            )


# %%
def get_file_path():
    wb = xw.Book.caller()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", config.BOOK_NAME)
    xw.Book(file_path).set_mock_caller()
