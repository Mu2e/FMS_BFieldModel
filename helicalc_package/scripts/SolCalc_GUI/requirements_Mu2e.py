# parameters related to field requirements and physical requirements
# used for Mu2e.
import numpy as np
from scipy.interpolate import interp1d
from coordinate_funcs import *

# physical requirements
coil_min_rad = 1.670 / 2. # m
cryo_inner_rad = 1.5 / 2. # m, nominal value
cryo_max_outer_rad = 2.6 / 2. # m
PS_TS_cryo_overlap = 0.561 # m
total_cryo_length_max = 4.5 # m
cryo_z_f = -3.443 # m
cryo_z_i = cryo_z_f - total_cryo_length_max

# regions defined in requirements docs
PS1_s_range = [-10.58, -9.08] # m
PS2_s_range = [-9.08, -6.58] # m
TS1_s_range = [-6.58, -5.58] # m
# translate to Z
PS1_z_range = [z_PS(s) for s in PS1_s_range]
PS2_z_range = [z_PS(s) for s in PS2_s_range]
TS1_z_range = [z_PS(s) for s in TS1_s_range]

##### PS axis requirements
### Field values
# PS1
s_PS1_Bz_min = -9.4 # m
z_PS1_Bz_min = z_PS(s_PS1_Bz_min)
PS1_Bz_min = 4.50 # T
# PS2
s_PS2_Bz_nom = -6.58 # m
z_PS2_Bz_nom = z_PS(s_PS2_Bz_nom)
PS2_Bz_nom = 2.50 # T
tol_PS2_Bz = 0.05 # percent
# TS1
s_TS1_Bz_nom = -5.58 # m
z_TS1_Bz_nom = z_PS(s_TS1_Bz_nom)
TS1_Bz_nom = 2.40 # T
tol_TS1_Bz = 0.05 # percent
# PS1: no local minimum -- this applies to all points R<=0.5 m
# PS2: no local minimum -- this applies to all points R<=0.25 m
def check_PS_extrema(df, z_col, Bz_col, x0=3.904, y0=0., PS_z_range=PS1_z_range):
    df_ = df.query(f'{PS_z_range[0]} <= {z_col} <= {PS_z_range[1]}')
    df_ = df_[np.isclose(df_.X, x0) & np.isclose(df_.Y, y0)]
    dBdz = np.diff(df_[Bz_col].values)/np.diff(df_[z_col].values)
    ddBddz = np.diff(dBdz)/np.diff(df_[z_col].values[1:])
    N_min = 0
    N_max = 0
    inds_extrema = np.where(np.diff(np.sign(np.diff(df_[Bz_col]))))[0]
    for i in inds_extrema:
        if ddBddz[i] > 0:
            N_min += 1
        else:
            N_max += 1
    return N_min, N_max
# PS2: within 5% of a uniform negative axial gradient, i.e. dB / B (from B_nom) < 0.05
def check_PS_axial_values(df, z_col, Bz_col, x0=3.904, y0=0.,
                          PS1_z_range=PS1_z_range, PS2_z_range=PS2_z_range,
                          dZ=0.001, z_at_min=z_PS2_Bz_nom,
                          z_at_max=PS2_z_range[0], z_PS1_Bz_min=z_PS1_Bz_min,
                          PS1_Bz_min=PS1_Bz_min, z_PS2_Bz_nom=z_PS2_Bz_nom,
                          PS2_Bz_nom=PS2_Bz_nom, tol_PS2_Bz=tol_PS2_Bz,
                          z_TS1_Bz_nom=z_TS1_Bz_nom, TS1_Bz_nom=TS1_Bz_nom,
                          tol_TS1_Bz=tol_TS1_Bz):
    # store entire Z_line (no query for PS2 region)
    df_ = df[np.isclose(df.X, x0) & np.isclose(df.Y, y0)].copy()
    df_PS2 = df_.query(f'{PS2_z_range[0]} <= {z_col} <= {PS2_z_range[1]}').copy()
    # print(df)
    # print(df.info())
    # print(df_)
    # print(df_PS2)
    # set up interpolation
    interp_func = interp1d(df_[z_col].values, df_[Bz_col].values, fill_value="extrapolate")
    zs_interp = np.arange(df_[z_col].round(3).min(), df_[z_col].round(3).max()+dZ, dZ)
    zs_interp_PS2 = np.arange(df_PS2[z_col].round(3).min(), df_PS2[z_col].round(3).max()+dZ, dZ)
    Bz_interp = interp_func(zs_interp)
    # calculate nominal field line
    # starting values
    i_max = np.argmax(Bz_interp)
    Bmax = Bz_interp[i_max]
    if z_at_max is None:
        zmax = zs_interp[i_max]
    else:
        zmax = z_at_max
    # print(zmax, Bmax)
    # ending values
    i_min = np.argmin(np.abs(zs_interp - z_at_min))
    Bmin = Bz_interp[i_min]
    zmin = zs_interp[i_min]
    # print(zmin, Bmin)
    # nominal line
    slope = (Bmin - Bmax) / (zmin - zmax)
    Bnom_func = lambda z: slope * (z - zmin) + Bmin
    # calculate values for interp func (plotting)
    Bnom_interp = Bnom_func(zs_interp_PS2)
    Bnom_interp_up = Bnom_interp * 1.05
    Bnom_interp_down = Bnom_interp * 0.95
    # calculate values for dataframe (check req)
    Bnom = Bnom_func(df_PS2[z_col].values)
    Bnom_up = Bnom * 1.05
    Bnom_down = Bnom * 0.95
    # check actual values
    map_in_tol = (df_PS2[Bz_col].values >= Bnom_down) & (df_PS2[Bz_col].values <= Bnom_up)
    N_out_of_tol = np.sum(~map_in_tol)
    to_spec = N_out_of_tol == 0
    # check that Bz max is in PS1
    zmax_actual = zs_interp[i_max]
    Bzmax_val_to_spec = Bmax >= PS1_Bz_min
    Bz_PS1 = interp_func(z_PS1_Bz_min).item()
    Bz_at_PS1_loc_to_spec = Bz_PS1 >= PS1_Bz_min
    Bzmax_loc_to_spec = (zmax_actual <= PS1_z_range[1]) and (zmax_actual >= PS1_z_range[0])
    # check Bz at PS2-TS1 interface
    Bz_PS2_TS1 = interp_func(z_PS2_Bz_nom).item()
    Bz_PS2_TS1_to_spec = (Bz_PS2_TS1 >= (PS2_Bz_nom * (1-tol_PS2_Bz))) and (Bz_PS2_TS1 <= (PS2_Bz_nom * (1+tol_PS2_Bz)))
    # check Bz at TS1-TS2 interface
    Bz_TS1_TS2 = interp_func(z_TS1_Bz_nom).item()
    Bz_TS1_TS2_to_spec = (Bz_TS1_TS2 >= (TS1_Bz_nom * (1-tol_TS1_Bz))) and (Bz_TS1_TS2 <= (TS1_Bz_nom * (1+tol_TS1_Bz)))
    # return anything I may want to plot or report
    # line 1: uniform gradient in TS2
    # line 2: PS1 value checks
    # line 2: PS2 and TS1 value checks
    return to_spec, N_out_of_tol, map_in_tol, zs_interp_PS2, Bnom_interp, Bnom_interp_up, Bnom_interp_down, \
    Bzmax_val_to_spec, Bmax, Bzmax_loc_to_spec, zmax_actual, Bz_at_PS1_loc_to_spec, Bz_PS1, \
    Bz_PS2_TS1_to_spec, Bz_PS2_TS1, Bz_TS1_TS2_to_spec, Bz_TS1_TS2

### Field gradients
# PS2, dBz/dz strictly < 0 for R<=0.25
# TS1, dBz/dZ <= 0.2 * slope_nom = 0.2 * -0.1 T/m = -0.02 T/m
# COMPLETE THIS!
# def check_axial_gradinet(df, )
