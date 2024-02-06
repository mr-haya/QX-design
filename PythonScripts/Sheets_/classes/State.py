"""
解析条件や物理量を読み込む
"""
import xlwings as xw

import config.cell_adress as ca
import config.sheet_name as sn
import config.config as cf


class State:
    def __init__(self):
        wb = cf.wb()
        # wb = xw.Book.caller()
        sht_pms = wb.sheets[sn.params]
        sht_ov = wb.sheets[sn.overview]

        self.alpha = sht_ov.range(ca.alpha_cell).value  # 全機迎角 [deg]
        self.beta = sht_ov.range(ca.beta_cell).value
        self.Vair = sht_ov.range(ca.Vair_cell).value  # 対気速度 [m/s]
        self.hE = sht_ov.range(ca.hE_cell).value  # 高度 [m]
        self.p = sht_ov.range(ca.p_cell).value  # ロール角速度 [deg/s]
        self.q = sht_ov.range(ca.q_cell).value  # ピッチ角速度 [deg/s]
        self.r = sht_ov.range(ca.r_cell).value  # ヨー加速度 [deg/s]
        self.dh = sht_ov.range(ca.dh_cell).value  # 頭下げのモーメントを生じるほうが正
        self.de = sht_ov.range(ca.de_cell).value  # 頭下げのモーメントを生じるほうが正
        self.dr = sht_ov.range(ca.dr_cell).value  # 左旋のモーメントを生じるほうが正
        self.rho = sht_pms.range(ca.rho_cell).value  # 空気密度 [kg/m^3]
        self.mu = sht_pms.range(ca.mu_cell).value  # 動粘性係数 [m^2/s]


state = State()

if __name__ == "__main__":
    file_path = cf.get_file_path()
    xw.Book(file_path).set_mock_caller()
    main()
