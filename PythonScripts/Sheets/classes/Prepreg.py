import xlwings as xw
import pandas as pd

from config import sheet_name as sn
from config import cell_adress as ca


class Prepreg:
    def __init__(self, name):
        wb = xw.Book.caller()
        sheet = wb.sheets[sn.params]
        df = sheet[ca.prepreg_cell].options(pd.DataFrame).value

        self.FAW = df["FAW"][name]
        self.resin_rate = df["Resin Content"][name]
        self.FLt = df["FLt"][name]
        self.ELt = df["ELt"][name]
        self.nuLt = df["νLt"][name]
        self.FTt = df["FTt"][name]
        self.ETt = df["ETt"][name]
        self.nuTt = df["νTt"][name]
        self.FLc = df["FLc"][name]
        self.ELc = df["ELc"][name]
        self.nuLc = df["νLc"][name]
        self.FTc = df["FTc"][name]
        self.ETc = df["ETc"][name]
        self.nuTc = df["νTc"][name]
        self.FLT = df["FLT"][name]
        self.GLT = df["GLT"][name]
        self.F1 = df["F1"][name]
        self.t = df["t"][name]
