'LLT'
def read_value_from_sheet(Wing, state, chord, chord_cp, dW(), setting_angle0(), foil_mixture, dihedral_angle0, Eix, GJ(), Cl_coef, Cdp_coef, Cm_coef)

'LLTの反復計算'
for iteration in range(1, 100):
    calculation_yz(Wing, state, y, z, setting_angle0, setting_angle, phi, dihedral_angle0, dihedral_angle, theta)
    calculation_cp(Wing, state, cp, y, z)
    calculation_ypzp(Wing, state, yp, zp, ymp, zmp, cp, dihedral_angle)
    calculation_Rij(Wing, state, ds, yp, zp, ymp, zmp, Rpij, Rmij, Rpijm, Rmijm)
    calculation_Qij(Wing, state, ds, yp, zp, ymp, zmp, Rpij, Rmij, Rpijm, Rmijm, dihedral_angle, Qij)
    update_circulation(Wing, state, iteration, coef, circulation, circulation_old)
    calculation_downwash(Wing, state, circulation, Qij, alpha_induced, cp, wi)
    calculation_alpha_effective(Wing, state, alpha_effective, alpha_induced, cp, dihedral_angle, setting_angle, alpha_max, alpha_min)
    calculation_Re(Wing, state, Re, cp, chord_cp, Re_max, Re_min)
    calculation_aerodynamic_coefficient(Wing, state, CL, Cdp, Cm_ac, a0, a1, Cda, Cm_cg, dN, dT, dL, dDp, dM_cg, dM_ac, cp, chord_cp, alpha_induced, alpha_effective, Re, dihedral_angle, foil_mixture, Cl_coef, Cdp_coef, Cm_coef)
    calculation_Force(Wing, state, iteration, Lift, Induced_Drag, cp, circulation, dihedral_angle, wi, CL, chord_cp)
    calculation_Moment(Wing, state, Bending_Moment, Bending_Moment_T, Shear_Force, Torque, dN, dT, dM_cg, dW, dihedral_angle, cp, y, z)
    calculation_deflection(Wing, state, deflection, theta, phi, Bending_Moment, Torque, Eix, GJ)
    change_mode(Wing, state, dihedral_angle, dihedral_angle0, deflection, theta, phi, dM_ac, dM_cg, Shear_Force, Bending_Moment, Bending_Moment_T, Torque)
    '収束判定'
    if iteration > 0:
        if error > Abs((Lift(iteration) - Lift(iteration - 1)) / Lift(iteration - 1)):
           break
    if p = 1:
       Application.StatusBar = "反復回数" & iteration & "回" & String(iteration, "■") 'ステータスバーに反復回数を表示'


Application.StatusBar = False
if iteration > iteration_max:
    iteration = iteration_max

input_data_to_type(Wing, state, iteration, dihedral_angle, chord_cp, y, cp, a0, a1, Cda, dDp, dN, dT, dM_ac, CL, Cdp, Lift, Induced_Drag)
VLM_wing = Wing

if p == 1:
    sht1.Range("AS2") = iteration
    output_data_to_sheet(Wing, alpha_effective, Re, CL, Cdp, Cm_ac, a0, Cda, Cm_cg, setting_angle,  dihedral_angle, circulation, wi, alpha_induced, y, z, deflection, theta, phi, Shear_Force, Bending_Moment,  Torque, Bending_Moment_T)

