import xlwings as xw
import numpy as np
import scipy as sp

from .Airfoil import Airfoil, MixedAirfoil
from .State import state

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf


class Rib:
    def __init__(self, y):
        self.y = y
        self.y_abs = abs(y)

        # シートを取得
        wb = xw.Book.caller()
        sht_wing = wb.sheets[sn.wing]
        sht_ov = wb.sheets[sn.overview]

        # 平面形を読み込む
        self.planform = sht_wing.range(ca.planform_cell).value
        self.vertex_arr = [r[0] for r in self.planform]
        if self.y_abs > self.vertex_arr[-1]:
            raise ValueError("y is out of range")
        self.taper_section = self._taper_section()
        self.chord = self._chord()
        self.setting_angle0 = self._setting_angle0()
        self.dihedral_angle0 = self._dihedral_angle0()
        self.center = self._center()
        self.airfoil = self._airfoil()

        self.hac = sht_ov.range(ca.hac_cell).value  # 翼型の空力中心位置

        # 初期化
        self.y_def = 0  # リブの変形後のy座標
        self.z_def = 0  # リブの変形後のz座標
        self.phi = 0  # リブのねじれ角
        self.theta = 0  # リブのたわみ角
        self.setting_angle = self.setting_angle0
        self.dihedral_angle = self.dihedral_angle0
        self.alpha_induced = 0
        self.alpha_effective = 0
        self.wi = 0
        self.Re = 0

        self._a0 = 0
        self._a1 = 0
        self._Cda = 0
        self._dL = 0
        self._dDp = 0
        self._dW = 0
        self._dN = 0
        self._dT = 0
        self._dM_ac = 0
        self._dM_cg = 0
        self._B_M = 0
        self._B_T = 0
        self._Torque = 0
        self._Shear_Force = 0

    @property
    def CL(self):
        return self.airfoil.CL(self.alpha_effective, self.Re)

    @property
    def Cdp(self):
        return self.airfoil.CD(self.alpha_effective, self.Re)

    @property
    def Cm_ac(self):
        return self.airfoil.Cm(self.alpha_effective, self.Re)

    @property
    def Cm_cg(self):
        return self.Cm_ac + self.CL * (self.center - self.hac)

    def dy(self, dy):  # 翼素投影面積
        self.cdy = self.chord * self.dy * np.cos(np.rad * self.dihedral_angle)

    @property
    def dynamic_pressure(self):
        return 0.5 * state.rho * (state.Vair - np.rad * state.r * self.y_def) ** 2

    @property
    def dL(self):
        return self.dynamic_pressure * self.CL * self.cdy

    @property
    def dDp(self):
        return self.dynamic_pressure * self.Cdp * self.cdy

    @property
    def dM_cg(self):
        return self.dynamic_pressure * self.Cm_cg * self.cdy * self.chord

    @property
    def dM_ac(self):
        return self.dynamic_pressure * self.Cm_ac * self.cdy * self.chord

    @property
    def alpha_x(self):
        return self.alpha_effective - self.setting_angle

    @property
    def dN(self):  # 機体上方の力
        return (
            self.dynamic_pressure
            * self.cdy
            * (
                self.CL * np.cos(np.rad * self.alpha_x)
                - self.Cdp * np.sin(np.rad * self.alpha_x)
            )
        )

    @property
    def dT(self):  # 機体後方の力
        return (
            self.dynamic_pressure
            * self.cdy
            * (
                -self.CL * np.sin(np.rad * self.alpha_x)
                + self.Cdp * np.cos(np.rad * self.alpha_x)
            )
        )

    def _chord(self):
        chord_arr = [r[1] for r in self.planform]
        y = self.y_abs
        chord_interp = sp.interpolate.interp1d(
            self.vertex_arr,
            chord_arr,
            kind="linear",
            fill_value="extrapolate",
        )
        return chord_interp([y])[0]

    def _setting_angle0(self):
        setting_angle_arr = [r[2] for r in self.planform]
        y = self.y_abs
        setting_angle_interp = sp.interpolate.interp1d(
            self.vertex_arr,
            setting_angle_arr,
            kind="linear",
            fill_value="extrapolate",
        )
        return setting_angle_interp([y])[0]

    def _dihedral_angle0(self):
        dihedral_angle_arr = [r[3] for r in self.planform]
        return dihedral_angle_arr[self.taper_section]

    def _center(self):
        center_arr = [r[4] for r in self.planform]
        y = self.y_abs
        center_interp = sp.interpolate.interp1d(
            self.vertex_arr,
            center_arr,
            kind="linear",
            fill_value="extrapolate",
        )
        return center_interp([y])[0]

    def _airfoil(self):
        foil_arr = [r[5] for r in self.planform]
        if foil_arr[self.taper_section] == foil_arr[self.taper_section + 1]:
            return Airfoil(foil_arr[self.taper_section])
        else:
            mixture = (self.y_abs - self.vertex_arr[self.taper_section]) / (
                self.vertex_arr[self.taper_section + 1]
                - self.vertex_arr[self.taper_section]
            )
            return MixedAirfoil(
                Airfoil(foil_arr[self.taper_section]),
                Airfoil(foil_arr[self.taper_section + 1]),
                mixture,
            )

    def _taper_section(self):
        y = self.y_abs
        arr = self.vertex_arr
        for i in range(1, len(arr)):
            if y < arr[i]:
                return i - 1
        return len(arr) - 1
