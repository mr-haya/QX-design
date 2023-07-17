import os
import xlwings as xw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf

from .State import state
from .Rib import Rib
from .Spar import WingSpar

rad = np.pi / 180  # conversion factor from degrees to radians
deg = 180 / np.pi  # conversion factor from radians to degrees


class Wing:
    def __init__(self):
        # シートを取得
        wb = xw.Book.caller()
        sht_wing = wb.sheets[sn.wing]
        # スパー生成
        self.spar = WingSpar()
        # 値読み込み
        self.b = sht_wing.range(ca.planform_cell).value[7][0] * 2
        self.weight = sht_wing.range(ca.weight_cell).value * 2  # 主翼重量
        self.planform = sht_wing.range(ca.planform_cell).value
        self.vertex_arr = [r[0] for r in self.planform]
        self.chord_arr = [r[1] for r in self.planform]
        self.S_planform = 0
        for i in range(7):
            self.S_planform += (
                (self.vertex_arr[i + 1] - self.vertex_arr[i])
                * (self.chord_arr[i + 1] + self.chord_arr[i])
                / 2
            )

        # 二次構造を読み込む

    def llt(self):
        # 揚力線理論
        # 参考文献：揚力線理論を拡張した地面効果域における翼の空力設計法 西出憲司 やや改変
        # 参考文献：Numerical nonliner lifting-line method @onedrive
        # 参考文献：グライダーの製作(1)　有限会社オリンポス

        # シートを取得
        wb = xw.Book.caller()
        sht_pms = wb.sheets[sn.params]
        sht_wing = wb.sheets[sn.wing]
        sht_def = wb.sheets[sn.deflection]
        sht_ov = wb.sheets[sn.overview]

        # 分割された仮想リブを用いて揚力線理論を解く
        # その後、実際のリブの位置に合わせて補間する
        self.span_div = cf.LLT_SPAN_DIV
        self.half_div = self.span_div // 2
        self.dy = self.b / self.span_div  # パネル幅
        Rib.dy = self.dy
        self.ds = self.dy / 2  # パネルの半幅

        # パネル生成
        self.panels = [
            Rib(y)
            for y in np.linspace(
                -self.b / 2 + self.ds, self.b / 2 - self.ds, self.span_div
            )
        ]
        self.dW = [
            self.weight * (self.panels[i].chord * self.dy / self.S_planform)
            for i in range(self.span_div)
        ]

        # # 境界生成

        # 初期化
        self.yp = np.zeros((self.span_div, self.span_div))
        self.zp = np.zeros((self.span_div, self.span_div))
        self.ymp = np.zeros((self.span_div, self.span_div))
        self.zmp = np.zeros((self.span_div, self.span_div))
        self.Rpij = np.zeros((self.span_div, self.span_div))
        self.Rmij = np.zeros((self.span_div, self.span_div))
        self.Rpijm = np.zeros((self.span_div, self.span_div))
        self.Rmijm = np.zeros((self.span_div, self.span_div))
        self.Qij = np.zeros((self.span_div, self.span_div))
        self.bending_moment = np.zeros(self.span_div)
        self.bending_moment_T = np.zeros(self.span_div)
        self.shear_force = np.zeros(self.span_div)
        self.torque = np.zeros(self.span_div)

        self._chord_mac = 0
        self._y_ = 0
        self._S_def = 0
        self._b_def = 0
        self._AR = 0
        self._Cla = 0
        self._Dp = 0
        self._L_roll = 0
        self._M_pitch = 0
        self._N_yaw = 0
        self._Drag_induced = 0
        self._Drag = 0
        self._CL = 0
        self._CDp = 0
        self._CDi = 0
        self._Cm_ac = 0
        self._e = 0
        self._aw = 0
        self._M_pitch = 0
        self._Cm_cg = 0

        for i in range(2):  # for i in range(cf.LLT_ITERATION_MAX):
            self.iteration = i
            self.epsilon = 0  # 平均吹き下ろし角
            self.Lift = 0  # 揚力
            self.Induced_drag = 0  # 誘導抗力

            self._calc_yz()
            self._calc_ypzp()
            self._calc_Rij()
            self._calc_Qij()
            self._update_circulation()
            self._calc_downwash()
            self._calc_alpha_effective()
            self._calc_Re()
            self._calc_force()
            self._calc_moment()
            self._calc_deflection()
            if self.iteration > 0:
                error = abs((self.Lift - self.Lift_old) / self.Lift_old)
                print(self.Lift, error)
                if cf.LLT_ERROR > error:
                    break
            self.Lift_old = self.Lift
            # xw.App().status_bar = "反復回数" & iteration & "回" & String(iteration, "■") 'ステータスバーに反復回数を表示
        self._calc_overview()
        print("Done!")
        # xw.App().status_bar = False

    def TR797(self):
        pass

    def _calc_yz(self):
        # 変形後取り付け角、上反角の更新
        for panel in self.panels:
            panel.setting_angle = panel.setting_angle0 + panel.phi
            panel.dihedral_angle = (
                panel.dihedral_angle0 + np.sign(panel.y) * panel.theta
            )

        # 変形後y,z座標の更新
        for i in range(self.half_div - 2, 0, -1):  # 左翼
            panel = self.panels[i]
            prev_panel = self.panels[i + 1]
            panel.y_def = prev_panel.y_def - self.dy * np.cos(
                -np.radians(prev_panel.dihedral_angle)
            )
            panel.z_def = prev_panel.z_def + self.dy * np.sin(
                -np.radians(prev_panel.dihedral_angle)
            )

        for i in range(self.half_div + 1, self.span_div):  # 右翼
            panel = self.panels[i]
            prev_panel = self.panels[i - 1]
            panel.y_def = prev_panel.y_def + self.dy * np.cos(
                np.radians(prev_panel.dihedral_angle)
            )
            panel.z_def = prev_panel.z_def + self.dy * np.sin(
                np.radians(prev_panel.dihedral_angle)
            )

    def _calc_ypzp(self):
        for j in range(self.span_div):
            for i in range(self.span_div):
                self.yp[i, j] = (self.panels[i].y_def - self.panels[j].y_def) * np.cos(
                    rad * self.panels[j].dihedral_angle
                ) + (self.panels[i].z_def - self.panels[j].z_def) * np.sin(
                    rad * self.panels[j].dihedral_angle
                )
                self.zp[i, j] = -(self.panels[i].y_def - self.panels[j].y_def) * np.sin(
                    rad * self.panels[j].dihedral_angle
                ) + (self.panels[i].z_def - self.panels[j].z_def) * np.cos(
                    rad * self.panels[j].dihedral_angle
                )
                self.ymp[i, j] = (self.panels[i].y_def - self.panels[j].y_def) * np.cos(
                    -rad * self.panels[j].dihedral_angle
                ) + (
                    self.panels[i].z_def - (-self.panels[j].z_def - 2 * state.hE)
                ) * np.sin(
                    -rad * self.panels[j].dihedral_angle
                )
                self.zmp[i, j] = -(
                    self.panels[i].y_def - self.panels[j].y_def
                ) * np.sin(-rad * self.panels[j].dihedral_angle) + (
                    self.panels[i].z_def - (-self.panels[j].z_def - 2 * state.hE)
                ) * np.cos(
                    -rad * self.panels[j].dihedral_angle
                )

    def _calc_Rij(self):
        for j in range(self.span_div):
            for i in range(self.span_div):
                self.Rpij[i, j] = np.sqrt(
                    (self.yp[i, j] - self.ds) ** 2 + self.zp[i, j] ** 2
                )
                self.Rmij[i, j] = np.sqrt(
                    (self.yp[i, j] + self.ds) ** 2 + self.zp[i, j] ** 2
                )
                self.Rpijm[i, j] = np.sqrt(
                    (self.ymp[i, j] - self.ds) ** 2 + self.zmp[i, j] ** 2
                )
                self.Rmijm[i, j] = np.sqrt(
                    (self.ymp[i, j] + self.ds) ** 2 + self.zmp[i, j] ** 2
                )

    def _calc_Qij(self):
        for j in range(self.span_div):
            for i in range(self.span_div):
                self.Qij[i, j] = (
                    (
                        -(self.yp[i, j] - self.ds) / (self.Rpij[i, j] ** 2)
                        + (self.yp[i, j] + self.ds) / (self.Rmij[i, j] ** 2)
                    )
                    * np.cos(
                        rad
                        * (
                            self.panels[i].dihedral_angle
                            - self.panels[j].dihedral_angle
                        )
                    )
                    + (
                        -(self.zp[i, j] / (self.Rpij[i, j] ** 2))
                        + (self.zp[i, j] / (self.Rmij[i, j] ** 2))
                    )
                    * np.sin(
                        rad
                        * (
                            self.panels[i].dihedral_angle
                            - self.panels[j].dihedral_angle
                        )
                    )
                    + (
                        (self.ymp[i, j] - self.ds) / (self.Rpijm[i, j] ** 2)
                        - (self.ymp[i, j] + self.ds) / (self.Rmijm[i, j] ** 2)
                    )
                    * np.cos(
                        rad
                        * (
                            self.panels[i].dihedral_angle
                            + self.panels[j].dihedral_angle
                        )
                    )
                    + (
                        self.zmp[i, j] / (self.Rpijm[i, j] ** 2)
                        - self.zmp[i, j] / (self.Rmijm[i, j] ** 2)
                    )
                    * np.sin(
                        rad
                        * (
                            self.panels[i].dihedral_angle
                            + self.panels[j].dihedral_angle
                        )
                    )
                )

    def _update_circulation(self):
        for i in range(self.span_div):
            if self.iteration > 1:
                self.panels[i].circulation = self.panels[i].circulation_old
                +cf.LLT_DAMPING_FACTOR * (
                    self.panels[i].circulation - self.panels[i].circulation_old
                )
            self.panels[i].circulation_old = self.panels[i].circulation

    def _calc_downwash(self):
        for i in range(self.span_div):
            self.panels[i].wi = 0
            for j in range(self.span_div):
                self.panels[i].wi += (
                    self.Qij[i, j] * self.panels[j].circulation * 1 / (4 * np.pi)
                )
            self.panels[i].alpha_induced = deg * np.arctan(
                self.panels[i].wi / state.Vair - rad * state.r * self.panels[i].y_def
            )
            self.epsilon += self.panels[i].alpha_induced
        self.epsilon /= self.span_div

    def _calc_alpha_effective(self):
        # 局所迎角 [deg]＝全機迎角+取り付け角+誘導迎角+ロール角速度pによる影響を追加+横滑りによる影響を追加
        for i in range(self.half_div - 2, 0, -1):  # 左翼
            panel = self.panels[i]
            panel.alpha_effective = (
                state.alpha
                + panel.setting_angle
                - panel.alpha_induced
                + deg
                * np.arctan(
                    (rad * state.p * panel.y_def)
                    / (state.Vair - rad * state.r * panel.y_def)
                )
                + deg
                * np.arctan(
                    state.Vair
                    * np.sin(rad * state.beta)
                    * np.sin(rad * panel.dihedral_angle)
                    / (state.Vair - rad * state.r * panel.y_def)
                )
            )

        for i in range(self.half_div + 1, self.span_div):  # 右翼
            panel = self.panels[i]
            panel.alpha_effective = (
                state.alpha
                + panel.setting_angle
                - panel.alpha_induced
                + deg
                * np.arctan(
                    (rad * state.p * panel.y_def)
                    / (state.Vair - rad * state.r * panel.y_def)
                )
                + deg
                * np.arctan(
                    state.Vair
                    * np.sin(rad * state.beta)
                    * np.sin(rad * panel.dihedral_angle)
                    / (state.Vair - rad * state.r * panel.y_def)
                )
            )

        for panel in self.panels:
            if panel.alpha_effective > cf.LLT_ALPHA_MAX:
                panel.alpha_effective = cf.LLT_ALPHA_MAX
            elif panel.alpha_effective < cf.LLT_ALPHA_MIN:
                panel.alpha_effective = cf.LLT_ALPHA_MIN

    def _calc_Re(self):
        for panel in self.panels:
            panel.Re = (
                state.Vair - rad * state.r * panel.y_def * panel.chord
            ) / state.mu
            if panel.Re > cf.LLT_RE_MAX:
                panel.Re = cf.LLT_RE_MAX
            elif panel.Re < cf.LLT_RE_MIN:
                panel.Re = cf.LLT_RE_MIN

    def _calc_force(self):
        for i in range(self.span_div):
            self.panels[i].circulation = (
                0.5
                * self.panels[i].chord
                * (state.Vair - rad * state.r * self.panels[i].y_def)
                * self.panels[i].CL
            )
            self.Lift += (
                state.rho
                * (state.Vair - rad * state.r * self.panels[i].y_def)
                * self.panels[i].circulation
                * self.dy
                * np.cos(rad * self.panels[i].dihedral_angle)
            )
            self.Induced_drag += (
                state.rho * self.panels[i].wi * self.panels[i].circulation * self.dy
            )

    def _calc_moment(self):
        for i in range(1, self.half_div):
            for j in range(1, i + 1):
                num1 = self.span_div - i
                num2 = self.span_div - j

                self.bending_moment[i] += (
                    self.panels[j - 1].dN
                    * np.cos(np.radians(self.panels[j - 1].dihedral_angle))
                    - self.dW[j - 1]
                ) * abs(self.panels[j - 1].y_def - self.panels[i].y_def) + self.panels[
                    j - 1
                ].dN * np.sin(
                    abs(np.radians(self.panels[j - 1].dihedral_angle))
                ) * abs(
                    self.panels[j - 1].z_def - self.panels[i].z_def
                )
                self.bending_moment[num1] += (
                    self.panels[num2].dN
                    * np.cos(np.radians(self.panels[num2].dihedral_angle))
                    - self.dW[num2]
                ) * abs(
                    self.panels[num2].y_def - self.panels[num1].y_def
                ) + self.panels[
                    num2
                ].dN * np.sin(
                    abs(np.radians(self.panels[num2].dihedral_angle))
                ) * abs(
                    self.panels[num2].z_def - self.panels[num1].z_def
                )

                self.bending_moment_T[i] += self.panels[j - 1].dT * abs(
                    self.panels[j - 1].y_def - self.panels[i].y_def
                )
                self.bending_moment_T[num1] += self.panels[num2].dT * abs(
                    self.panels[num2].y_def - self.panels[num1].y_def
                )

                self.torque[i] += self.panels[j - 1].dM_cg + self.panels[j - 1].dT * (
                    self.panels[j - 1].z_def - self.panels[i].z_def
                )
                self.torque[num1] += self.panels[num2].dM_cg + self.panels[num2].dT * (
                    self.panels[num2].z_def - self.panels[num1].z_def
                )

                self.shear_force[i] += self.panels[j - 1].dN - self.dW[j - 1]
                self.shear_force[num1] += self.panels[num2].dN - self.dW[num2]

        self.bending_moment[self.half_div] *= 0.5
        self.bending_moment_T[self.half_div] *= 0.5
        self.torque[self.half_div] *= 0.5
        self.shear_force[self.half_div] *= 0.5

    def _calc_deflection(self):
        for panel in self.panels:
            panel.deflection = 0
            panel.theta = 0
            panel.phi = 0

        for i in range(self.half_div - 1):
            num1 = self.half_div - 1 - i
            num2 = self.half_div + i

            # 左翼
            self.spar.slice_at(self.panels[num1].y)
            self.panels[num1].theta = (
                self.panels[num1 + 1].theta
                + (self.bending_moment[num1] / self.spar.EI) * self.dy
            )  # たわみ角
            self.panels[num1].phi = (
                self.panels[num1 + 1].phi + (self.torque[num1] * self.dy) / self.spar.GJ
            )  # ねじれ角

            # 右翼
            self.spar.slice_at(self.panels[num2].y)
            self.panels[num2 + 1].theta = (
                self.panels[num2].theta
                + (self.bending_moment[num2 + 1] / self.spar.EI) * self.dy
            )  # たわみ角
            self.panels[num2 + 1].phi = (
                self.panels[num2].phi + (self.torque[num2 + 1] * self.dy) / self.spar.GJ
            )  # ねじれ角

            # たわみ
            self.panels[num1].deflection = (
                self.panels[num1 + 1].deflection + self.panels[num1 + 1].theta * self.dy
            )  # 左翼
            self.panels[num2 + 1].deflection = (
                self.panels[num2].deflection + self.panels[num2].theta * self.dy
            )  # 右翼

        # Φを[rad]から[deg]に変換
        for i in range(self.span_div):
            self.panels[i].theta = np.degrees(self.panels[i].theta)  # [deg]への変換
            self.panels[i].phi = np.degrees(self.panels[i].phi)  # [deg]への変換

    def _calc_overview(self):
        for panel in self.panels:
            self._S_def += panel.cdy
            self._Dp += panel.dDp

    @property
    def S_def(self):
        return self._S_def

    @property
    def Dp(self):
        for panel in self.panels:
            self._Dp += panel.airfoil.CD(panel.alpha_effective, panel.Re)

        return self._Dp

    @property
    def b_def(self):
        return self.panels[0].y_def - self.panels[-1].y_def

    @property
    def AR(self):
        return self.b_def**2 / self.S_def

    @property
    def Drag(self):
        return self.Dp + self.Induced_drag

    def _change_mode(self):
        pass
