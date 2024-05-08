# unlikely to use often in helicalc main package, so I am putting this as a
# standalone utility in the GUI directory.
import numpy as np
from helicalc.tools import calc_cable_lengths

# resistivities
rho_Cu = 1.68e-8 # Ohm m, 20 deg C
alpha_Cu = 0.0039 # deg C^-1 (or K^-1)
rho_Cu_77K = rho_Cu * (1 + alpha_Cu * (77.-293.))# Ohm m, LN2 temp
rho_Cu_77K_approx = rho_Cu / 10. # Ohm m, LN2 temp, suggested improvement from A. Hocker
# for superconductor
rho_SC = 0.

def calc_resistive_power_coils(geom_df, resistivity=rho_Cu, full_cable=False):
    # calculate resistances
    L_cable_coils, L_cable_list = calc_cable_lengths(geom_df, use_rho_0_a=False)
    I_list = []
    R_list = []
    R_cables = 0.
    for i in range(len(geom_df)):
        L = L_cable_list[i]
        geom_coil = geom_df.iloc[i]
        if full_cable:
            A = geom_coil.w_cable * geom_coil.h_cable
        else:
            A = geom_coil.w_sc * geom_coil.h_sc
        R_ = resistivity * L/A # Ohm
        R_cables += R_
        R_list.append(R_)
        I_list.append(geom_coil.I_turn)
    R_list = np.array(R_list)
    I_list = np.array(I_list)
    # calculate power
    power_list = I_list**2 * R_list
    power_tot = np.sum(power_list) # W
    power_tot_MW = power_tot * 1e-6
    power_list_MW = power_list * 1e-6
    return power_tot_MW, power_tot, power_list_MW, power_list, R_cables, R_list, I_list, L_cable_coils, L_cable_list
