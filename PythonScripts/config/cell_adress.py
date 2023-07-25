# セルアドレスを定義する

# ------------------------ 全機諸元 ------------------------ ##
alpha_cell = "I21"
beta_cell = "I22"
Vair_cell = "I23"
hE_cell = "I24"
p_cell = "I25"
q_cell = "I26"
r_cell = "I27"
dh_cell = "I28"
de_cell = "I29"
dr_cell = "I30"

hac_cell = "AC15"

# ---------------------- フライトプラン---------------------- ##
# ------------------------ 重量分布 ------------------------ ##
# ------------------------ 荷重評価 ------------------------ ##
# -------------------------- 主翼 ------------------------- ##
planform_cell = "H6:M13"
n_cell = "C36"
first_y_cell = "C37"
default_dy_cell = "C38"
secondary_structure_cell = "Y4:Y13"
plank_start_cell = "AH4"
plank_end_cell = "AH5"
weight_cell = "BH40"

# .expand("down")
stringer_cell = "AB4:AE4"
spec_rib_cell = "E44:G44"
margin_rib_cell = "K44"
export_aero_cell = "L44:AB44"
export_geometry_cell = "AM44:AW44"

# ------------------------ 水平尾翼 ------------------------ ##
# ------------------------ 垂直尾翼 ------------------------ ##
# ------------------------- カウル ------------------------- ##
# ------------------------ 胴接構造 ------------------------ ##
# -------------------------- 主桁 -------------------------- ##
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

# ------------------------ 積層構成 ------------------------ ##
laminate_cell = "B4:L21"

# -------------------------- 翼型 -------------------------- ##
alpha_min_cell = "C3"
alpha_max_cell = "D3"
alpha_step_cell = "E3"
Re_min_cell = "C4"
Re_max_cell = "D4"
Re_step_cell = "E4"
foil_detail_cell = "B14"  # .expand("table")
foil_outline_cell = [14, 19]

# -------------------------- 物性値 -------------------------- ##
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
