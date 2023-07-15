import xlwings as xw
import pandas as pd
import math

from config import sheet_name as sn
from config import cell_adress as ca

from .Laminate import Laminate


class Spar:  # 任意位置での断面と各種剛性、強度を返す
    def __init__(self, name, length, laminate_name, ellipticity, taper_ratio, zi_0):
        self.name = name
        self.length = length
        self.ellipticity = ellipticity
        self.taper_ratio = taper_ratio
        self.zi_0 = zi_0

        # 積層板のインスタンスを生成
        self.laminate = Laminate(laminate_name)
        self.E = self.laminate.E_equiv
        self.G = self.laminate.G_equiv
        self.nu = self.laminate.nu_equiv

        # 初期化
        self._zi = 0
        self._xi = 0
        self._zo = 0
        self._xo = 0
        self._Ix = 0
        self._Iy = 0
        self._J = 0
        self._area = 0
        self._am = 0
        self._perimeter = 0
        self._EI = 0
        self._GJ = 0

    def slice_at(self, x):
        if x >= 0 and x <= self.length:
            self._zi = (self.zi_0 - self.taper_ratio * x) / 2
            self._xi = zi * self.ellipticity
            self._zo = zi + self.laminate.thickness
            self._zo = xi + self.laminate.thickness_zenshu
            (
                self._Ix,
                self._Iy,
                self._J,
                self._area,
                self._am,
                self._perimeter,
            ) = calc_section_modulus(self._zi, self._xi, self._zo, self._xo)

    @property
    def zi(self):
        return self._zi

    @property
    def xi(self):
        return self._xi

    @property
    def zo(self):
        return self._zo

    @property
    def xo(self):
        return self._xo

    @property
    def Ix(self):
        return self._Ix

    @property
    def Iy(self):
        return self._Iy

    @property
    def J(self):
        return self._J

    @property
    def area(self):
        return self._area

    @property
    def perimeter(self):
        return self._perimeter

    @property
    def am(self):
        return self._am

    @property
    def EI(self):
        return self._Ix * self.E * 1e-3

    @property
    def GJ(self):
        return self._J * self.G * 1e-3


class WingSpar(Spar):  # spar0~3で構成される主翼桁
    def __init__(self):
        # エクセルのシートを取得
        wb = xw.Book.caller()
        sheet = wb.sheets[sn.spar]

        # 各桁のインスタンスを生成
        self.spar0 = Spar(
            "0番",
            sheet.range(ca.length_0_cell).value,
            sheet.range(ca.laminate_0_cell).value,
            sheet.range(ca.ellipticity_0_cell).value,
            sheet.range(ca.taper_ratio_0_cell).value,
            sheet.range(ca.zi_0_cell).value,
        )
        self.spar1 = Spar(
            "1番",
            sheet.range(ca.length_1_cell).value,
            sheet.range(ca.laminate_1_cell).value,
            sheet.range(ca.ellipticity_1_cell).value,
            sheet.range(ca.taper_ratio_1_cell).value,
            sheet.range(ca.zi_1_cell).value,
        )
        self.spar2 = Spar(
            "2番",
            sheet.range(ca.length_2_cell).value,
            sheet.range(ca.laminate_2_cell).value,
            sheet.range(ca.ellipticity_2_cell).value,
            sheet.range(ca.taper_ratio_2_cell).value,
            sheet.range(ca.zi_2_cell).value,
        )
        self.spar3 = Spar(
            "3番",
            sheet.range(ca.length_3_cell).value,
            sheet.range(ca.laminate_3_cell).value,
            sheet.range(ca.ellipticity_3_cell).value,
            sheet.range(ca.taper_ratio_3_cell).value,
            sheet.range(ca.zi_3_cell).value,
        )
        self.spar1_start = sheet.range(ca.spar1_start_cell).value
        self.spar2_start = sheet.range(ca.spar2_start_cell).value
        self.spar3_start = sheet.range(ca.spar3_start_cell).value

        # 初期化
        self._zi = 0
        self._xi = 0
        self._zo = 0
        self._xo = 0
        self._Ix = 0
        self._Iy = 0
        self._J = 0
        self._area = 0
        self._am = 0
        self._perimeter = 0
        self._EI = 0
        self._GJ = 0

    def slice_at(self, x):
        if x < self.spar1_start:
            x_0 = x
            self.spar0.slice_at(x_0)
            self._zi = self.spar0.zi
            self._xi = self.spar0.xi
            self._zo = self.spar0.zo
            self._xo = self.spar0.xo
            self._E = self.spar0.E
            self._G = self.spar0.G
            self._nu = self.spar0.nu
        elif x < self.spar0.length / 2:
            x_0 = x
            x_1 = x - self.spar1_start
            self.spar0.slice_at(x_0)
            self.spar1.slice_at(x_1)
            self._zi = self.spar0.zi
            self._xi = self.spar0.xi
            self._zo = self.spar1.zo
            self._xo = self.spar1.xo
            self._E = (self.spar0.E + self.spar1.E) / 2
            self._G = (self.spar0.G + self.spar1.G) / 2
            self._nu = (self.spar0.nu + self.spar1.nu) / 2
        elif x < self.spar2_start:
            x_1 = x - self.spar1_start
            self.spar1.slice_at(x_1)
            self._zi = self.spar1.zi
            self._xi = self.spar1.xi
            self._zo = self.spar1.zo
            self._xo = self.spar1.xo
            self._E = self.spar1.E
            self._G = self.spar1.G
            self._nu = self.spar1.nu
        elif x < self.spar1_start + self.spar1.length:
            x_1 = x - self.spar1_start
            x_2 = x - self.spar2_start
            self.spar1.slice_at(x_1)
            self.spar2.slice_at(x_2)
            self._zi = self.spar1.zi
            self._xi = self.spar1.xi
            self._zo = self.spar2.zo
            self._xo = self.spar2.xo
            self._E = (self.spar1.E + self.spar2.E) / 2
            self._G = (self.spar1.G + self.spar2.G) / 2
            self._nu = (self.spar1.nu + self.spar2.nu) / 2
        elif x < self.spar3_start:
            x_2 = x - self.spar2_start
            self.spar2.slice_at(x_2)
            self._zi = self.spar2.zi
            self._xi = self.spar2.xi
            self._zo = self.spar2.zo
            self._xo = self.spar2.xo
            self._E = self.spar2.E
            self._G = self.spar2.G
            self._nu = self.spar2.nu
        elif x < self.spar2_start + self.spar2.length:
            x_2 = x - self.spar2_start
            x_3 = x - self.spar3_start
            self.spar2.slice_at(x_2)
            self.spar3.slice_at(x_3)
            self._zi = self.spar2.zi
            self._xi = self.spar2.xi
            self._zo = self.spar3.zo
            self._xo = self.spar3.xo
            self._E = (self.spar2.E + self.spar3.E) / 2
            self._G = (self.spar2.G + self.spar3.G) / 2
            self._nu = (self.spar2.nu + self.spar3.nu) / 2
        else:
            x_3 = x - self.spar3_start
            self.spar3.slice_at(x_3)
            self._zi = self.spar3.zi
            self._xi = self.spar3.xi
            self._zo = self.spar3.zo
            self._xo = self.spar3.xo
            self._E = self.spar3.E
            self._G = self.spar3.G
            self._nu = self.spar3.nu

        (
            self._Ix,
            self._Iy,
            self._J,
            self._area,
            self._am,
            self._perimeter,
        ) = calc_section_modulus(self._zi, self._xi, self._zo, self._xo)

    @property
    def E(self):
        return self._E

    @property
    def G(self):
        return self._G

    @property
    def nu(self, x):
        return self._nu


def calc_section_modulus(z_in, x_in, z_out, x_out, num=32766):
    pi = math.pi
    dtheta = (pi / 2) / num
    a_in = x_in
    b_in = z_in
    a_out = x_out
    b_out = z_out
    a_cl = (a_in + a_out) / 2
    b_cl = (b_in + b_out) / 2
    Am = pi * a_cl * b_cl

    Ix, Iy, j, Area, Length = 0, 0, 0, 0, 0

    for i in range(num + 1):
        theta = dtheta * i

        # Calculate x, y, r for inner, outer, and center line
        x_in, y_in, r_in = calculate_coordinates(a_in, b_in, theta)
        x_out, y_out, r_out = calculate_coordinates(a_out, b_out, theta)
        x_cl, y_cl, r_cl = calculate_coordinates(a_cl, b_cl, theta)

        # Calculate phi, ds, dr, dA, t
        phi = pi / 2 if theta == 0 else math.atan(-((b_cl / a_cl) ** 2) * (x_cl / y_cl))
        ds = (
            r_cl * dtheta
            if theta == 0
            else r_cl * dtheta / math.cos(theta - pi / 2 - phi)
        )
        dr = r_out - r_in
        if dr == 0:
            dr = 1
        da = r_cl * dtheta * dr
        thickness = dr * math.cos(theta - pi / 2 - phi)

        # Calculate Ix, Iy, J, L, A
        Ix += y_cl * y_cl * da
        Iy += x_cl * x_cl * da
        j += ds / thickness
        Length += ds
        Area += da

    # Calculate J
    j = (4 * Am * Am) / (j * 4)

    return Ix * 4, Iy * 4, j, Area * 4, Am, Length * 4


def calculate_coordinates(a, b, theta):
    x = a * math.cos(theta)
    y = b * math.sin(theta)
    r = math.sqrt(x**2 + y**2)
    return x, y, r
