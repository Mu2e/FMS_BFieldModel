import numpy as np
import pandas as pd
from scipy.constants import mu_0

def read_solenoid_geom_combined(params_dir, geom_name, sep=',', skiprows=1):
    # coils
    cols_coil = ["Coil_Num","Ri", "Ro", "L", "x", "y", "z" , "rot0", "rot1", "rot2",
                 "I_tot", "N_layers", "N_turns", "N_turns_tot", "I_turn", "helicity"]
    cols_coil_rescale = ["Ri", "Ro", "L", "x", "y", "z"]

    Coils = pd.read_csv(params_dir+geom_name+"_coil.txt", names=cols_coil, sep=sep, skiprows=skiprows)
    for col in cols_coil_rescale:
        Coils[col] = Coils[col] / 1e3
    # conductor
    cols_cond = ["h_cable", "w_cable", "h_sc", "w_sc", "t_gi", "t_ci", "t_il", "phi0", "phi1"]
    cols_cond_rescale = ["h_cable", "w_cable", "h_sc", "w_sc", "t_gi", "t_ci", "t_il"]

    Conductor = pd.read_csv(params_dir+geom_name+"_conductor.txt", names=cols_cond, sep=sep, skiprows=skiprows)
    for col in cols_cond_rescale:
        Conductor[col] = Conductor[col] / 1e3

    # integration limits zeta
    Conductor['zeta0'] = (Conductor.w_cable-Conductor.w_sc)/2. + Conductor.t_ci
    Conductor['zeta1'] = Conductor.zeta0 + Conductor.w_sc

    df = pd.concat([Coils, Conductor], axis=1)

    # integration limits rho for first layer
    # higher layers increased by offsets
    df['rho0_a'] = df.Ri + (df.h_cable - df.h_sc)/2. + df.t_gi + df.t_ci
    df['rho1_a'] = df.rho0_a + df.h_sc

    # current density
    # df['j'] = df.I_turn / ((df.rho1_a-df.rho0_a)*(df.zeta1-df.zeta0))
    df['j'] = df.I_turn / (df.w_sc*df.h_sc) # faster/more clear?

    df['mu_fac'] = mu_0 * df.j / (4*np.pi)

    df['pitch'] = df.w_cable + 2.*df.t_ci # BAD--Hank says good. Helical parameters. Would need adjustment if desired pitch is larger than laying cables on top of one another
    # df['pitch_2'] = (df.L - df.w_cable - 2.*df.t_ci) / df.N_turns
    df['pitch_2'] = (df.L - df.w_cable - 2.*df.t_gi) / df.N_turns
    # df['pitch'] = df.L / df.N_turns
    df['pitch_bar'] = df['pitch'] / (2*np.pi) # follows OPERA
    # df['pitch_bar'] = df['pitch_2'] / (2*np.pi)

    # integration limits
    # df['phi_i']
    # df['N_turns_true'] = df['L'] / df['pitch']
    # df['phi_i'] = 0.
    # df['phi_f'] = 2*np.pi*df.N_turns_true
    # df['phi_i'] = df['phi0']

    # phi0 and phi1 to radians
    df['phi0_deg'] = df['phi0']
    df['phi1_deg'] = df['phi1']
    df['phi0'] = np.radians(df['phi0_deg'])
    df['phi1'] = np.radians(df['phi1_deg'])

    # useful things for helicalc -- DS coils
    # Number of turns * Ri (integer) -- used to determine number of field points to calculate at once
    df['Nt_Ri'] = (df.N_turns * df.Ri).astype(int)
    # which dxyz to use (default to 1 for multilayer, 2 for single layer)
    # can overrride where necessary
    df['dxyz'] = np.where(df['N_layers'] > 1, 1, 2)

    #return Coils, Conductor
    return df
