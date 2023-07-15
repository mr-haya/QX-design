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

        # 平面形を読み込む
        planform = sht_wing.range(ca.planform_cell).value

        # リブを読み込む

        # スパーを読み込む
        self.spar = WingSpar()

        # 二次構造を読み込む

    def llt(self):
        # 揚力線理論
        # 参考文献：揚力線理論を拡張した地面効果域における翼の空力設計法 西出憲司 やや改変
        # 参考文献：Numerical nonliner lifting-line method @onedrive
        # 参考文献：グライダーの製作(1)　有限会社オリンポス
        # p=1ならエクセルシートに結果を出力

        # シートを取得
        wb = xw.Book.caller()
        sht_pms = wb.sheets[sn.params]
        sht_wing = wb.sheets[sn.wing]
        sht_def = wb.sheets[sn.deflection]
        sht_ov = wb.sheets[sn.overview]

        # 分割された仮想リブを用いて揚力線理論を解く
        # その後、実際のリブの位置に合わせて補間する
        self.b = sht_wing.range(ca.planform_cell).value[7][0] * 2
        self.dy = self.b / cf.LLT_SPAN_DIV  # パネル幅
        self.ds = self.dy / 2  # パネルの半幅

        self.span_div = cf.LLT_SPAN_DIV
        self.half_div = self.span_div // 2

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
        self.circulation = np.zeros(self.span_div)
        self.circulation_old = np.zeros(self.span_div)
        self.epsilon = 0  # 平均吹き下ろし角
        self.Lift = 0  # 揚力
        self.Induced_drag = 0  # 誘導抗力

        self.Bending_moment = np.zeros(self.span_div)
        self.Bending_moment_T = np.zeros(self.span_div)
        self.Shear_Force = np.zeros(self.span_div)
        self.Torque = np.zeros(self.span_div)

        self.panels = [
            Rib(y)
            for y in np.linspace(
                -self.b / 2 + self.ds, self.b / 2 - self.ds, self.span_div
            )
        ]

        for i in range(1):  # for i in range(cf.LLT_ITERATION_MAX):
            self.iteration = i
            self._calc_yz()
            self._calc_ypzp()
            self._calc_Rij()
            self._calc_Qij()
            self._update_circulation()
            self._calc_downwash()
            self._calc_alpha_effective()
            self._calc_Re()
            self._calc_Force()
            self.calc_Moment()
            if self.iteration > 0:
                if cf.LLT_ERROR > abs((self.Lift - self.Lift_old) / self.Lift_old):
                    break
            self.Lift_old = self.Lift
            # xw.App().status_bar = "反復回数" & iteration & "回" & String(iteration, "■") 'ステータスバーに反復回数を表示

        # xw.App().status_bar = False

    def TR797(self):
        pass

    def _calc_yz(self):
        # 取り付け角、上反角の更新
        for panel in self.panels:
            panel.setting_angle = panel.setting_angle0 + panel.phi
            if panel.y < 0:
                panel.dihedral_angle = panel.dihedral_angle0 - panel.theta
            else:
                panel.dihedral_angle = panel.dihedral_angle0 + panel.theta

        # y,z座標の更新
        for i in range(self.half_div - 2, 0, -1):  # 左翼
            panel = self.panels[i]
            prev_panel = self.panels[i + 1]
            panel.y_def = prev_panel.y_def - self.dy * np.cos(
                -rad * prev_panel.dihedral_angle
            )
            panel.z_def = prev_panel.z_def + self.dy * np.sin(
                -rad * prev_panel.dihedral_angle
            )

        for i in range(self.half_div + 1, self.span_div):  # 右翼
            panel = self.panels[i]
            prev_panel = self.panels[i - 1]
            panel.y_def = prev_panel.y_def + self.dy * np.cos(
                rad * prev_panel.dihedral_angle
            )
            panel.z_def = prev_panel.z_def + self.dy * np.sin(
                rad * prev_panel.dihedral_angle
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
                self.circulation[i] = self.circulation_old[
                    i
                ] + cf.LLT_DAMPING_FACTOR * (
                    self.circulation[i] - self.circulation_old[i]
                )
            self.circulation_old[i] = self.circulation[i]

    def _calc_downwash(self):
        self.epsilon = 0
        for i in range(self.span_div):
            self.panels[i].wi = 0
            for j in range(self.span_div):
                self.panels[i].wi += (
                    self.Qij[i, j] * self.circulation[j] * 1 / (4 * np.pi)
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
            self.panels[i].dy = self.dy
            self.circulation[i] = (
                0.5
                * self.panels[i].chord
                * (state.Vair - rad * state.r * self.panels[i].y_def)
                * self.panels[i].CL
            )
            self.Lift += (
                state.rho
                * (state.Vair - rad * state.r * self.panels[i].y_def)
                * self.circulation[i]
                * self.dy
                * np.cos(rad * self.panels[i].dihedral_angle)
            )
            self.Induced_drag += (
                state.rho * self.panels[i].wi * circulation[i] * self.dy
            )

    def _calc_moment(self):
        for i in range(1, self.half_div):
            for j in range(1, i + 1):
                num1 = self.span_div - i
                num2 = self.span_div - j

                self.Bending_Moment[i] += (
                    self.panels[j - 1].dN
                    * np.cos(np.radians(self.panels[j - 1].dihedral_angle))
                    - dW[j - 1]
                ) * abs(self.panels[j - 1].y_def - self.panels[i].y) + self.panels[
                    j - 1
                ].dN * np.sin(
                    abs(np.radians(self.panels[j - 1].dihedral_angle))
                ) * abs(
                    self.panels[j - 1].z_def - self.panels[i].z
                )
                self.Bending_Moment[num1] += (
                    self.panels[num2].dN
                    * np.cos(np.radians(self.panels[num2].dihedral_angle))
                    - dW[num2]
                ) * abs(self.panels[num2].y_def - self.panels[num1].y) + self.panels[
                    num2
                ].dN * np.sin(
                    abs(np.radians(self.panels[num2].dihedral_angle))
                ) * abs(
                    self.panels[num2].z_def - self.panels[num1].z
                )

                self.Bending_Moment_T[i] += self.panels[j - 1].dT * abs(
                    self.panels[j - 1].y_def - self.panels[i].y
                )
                self.Bending_Moment_T[num1] += self.panels[num2].dT * abs(
                    self.panels[num2].y_def - self.panels[num1].y
                )

                self.Torque[i] += self.panels[j - 1].dM_cg + self.panels[j - 1].dT * (
                    self.panels[j - 1].z_def - self.panels[i].z
                )
                self.Torque[num1] += self.panels[num2].dM_cg + self.panels[num2].dT * (
                    self.panels[num2].z_def - self.panels[num1].z
                )

                self.Shear_Force[i] += self.panels[j - 1].dN - dW[j - 1]
                self.Shear_Force[num1] += self.panels[num2].dN - dW[num2]

        self.Bending_Moment[self.half_div] *= 0.5
        self.Bending_Moment_T[self.half_div] *= 0.5
        self.Torque[self.half_div] *= 0.5
        self.Shear_Force[self.half_div] *= 0.5

    def _calc_deflection(self):
        pass

    def _change_mode(self):
        pass
