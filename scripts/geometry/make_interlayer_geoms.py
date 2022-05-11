from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import lmfit as lm

from helicalc import helicalc_dir
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.coil import CoilIntegrator

# downweight midpoint
weights_default = np.array([[0.1, 1],[0.1, 1],[0.1, 1]])

def prep_df_il(geom_df, coil_query='(Coil_Num >= 56)'):
    # note: only works for coils with 1 and 2 layers
    # FIXME! Add 3 layer coil support.
    # coil_query to check if any coils not in DS
    # and only want coils with multiple layers
    geom_il = geom_df.query(f'(N_layers > 1) & {coil_query}')
    # calculate radius values (start=R0, end=R1, mean=RC)
    geom_il.eval('R0 = (rho0_a+rho1_a)/2', inplace=True)
    geom_il.eval('R1 = R0 + (h_cable + 2*t_ci + t_il)', inplace=True)
    geom_il.eval('RC = (R0 + R1)/2', inplace=True)
    return geom_il

def get_endpoint_params(df_il):
    # assumes 2 layer coil
    # assumes end of inner layer is 36deg before phi0 (last of 10 OPERA bricks)
    # input point
    R0 = df_il.R0
    # set up coil integrator in each layer to get each end point
    # in, layer = 1; out, layer = 2
    dxyz = np.array([df_il.h_sc / 2, df_il.w_sc / 2, df_il.N_turns*np.pi])
    myCoil1 = CoilIntegrator(df_il, dxyz=dxyz, layer=1, lib=np,
                             int_func=np.trapz, interlayer_connect=True)
    myCoil2 = CoilIntegrator(df_il, dxyz=dxyz, layer=2, lib=np,
                             int_func=np.trapz, interlayer_connect=True)
    # integrate at point in the center of each coil, to set RX, RY, RZ
    myCoil1.integrate(df_il.x, df_il.y, df_il.z)
    myCoil2.integrate(df_il.x, df_il.y, df_il.z)
    # grab relevant point from each layer
    point_in0 = np.array([-myCoil1.RX[1, 1, 0], -myCoil1.RY[1, 1, 0],
                         -myCoil1.RZ[1, 1, 0]])
    point_out0 = np.array([-myCoil2.RX[1, 1, 0], -myCoil2.RY[1, 1, 0],
                          -myCoil2.RZ[1, 1, 0]])
    point_in = deepcopy(point_in0)
    point_out = deepcopy(point_out0)
    # adjust based on coil center -- no y center for Mu2e DS
    point_in[0] = point_in[0] + df_il.x
    point_out[0] = point_out[0] + df_il.x
    point_in[1] = point_in[1] + df_il.y
    point_out[1] = point_out[1] + df_il.y
    point_in[2] = point_in[2] + df_il.z
    point_out[2] = point_out[2] + df_il.z
    # calculate angles
    phi0 = np.arctan2(point_in0[1], point_in0[0])
    phi1 = np.arctan2(point_out0[1], point_out0[0])
    phi0_deg = np.degrees(phi0)
    phi1_deg = np.degrees(phi1)
    return point_in, point_out, point_in0, point_out0,\
           phi0, phi1, phi0_deg, phi1_deg

def get_midpoint(point_in0, point_out0, phi0, phi1, RC, xcoil, zcoil):
    zC = (point_in0[2]+point_out0[2])/2
    # get center phi
    phiC = (phi0 + phi1)/2
    xC = RC * np.cos(phiC)
    yC = RC * np.sin(phiC)
    point_mid0 = np.array([xC, yC, zC])
    point_mid = np.array([xC+xcoil, yC, zC+zcoil])
    return point_mid, point_mid0

def get_dphi(R, point_in, point_out):
    L = np.linalg.norm(point_in-point_out)
    dphi = 2*np.arcsin(L/(2*R))
    return dphi, np.degrees(dphi)

# 2nd point (middle) and 3rd point (end) are what should be used
# for chi2 distance
def generate_arc_points_local(dphi, R, N=3):
    # use N=3 to get two endpoints, and midpoint
    # larger N for plotting curve
    # dphi is in radians -- save in geometry file in degrees
    phis = np.linspace(0, dphi, N)
    xs = np.zeros(N)
    ys = -R * np.cos(phis) + R
    zs = R * np.sin(phis)
    return np.array([xs, ys, zs]).T

def rotate_to_global(pos, **params):
    # euler2, in degrees, in the following order: [Phi2, theta2, psi2]
    angles = [params['psi2'], params['theta2'], params['Phi2']]
    rot = Rotation.from_euler("zyz", angles=angles, degrees=True)
    return rot.apply(pos)

def get_arc_3points(R, dphi, pos0, **params):
    # generate endpoints and midpoint
    arc_local = generate_arc_points_local(dphi, R, N=3)
    # rotate to global frame
    arc_global = rotate_to_global(arc_local, **params)
    # translate to x0,y0,z0 endpoint
    arc_global += np.array([pos0])[-1,:]
    return arc_global.T

def get_arc_2points(R, dphi, pos0, **params):
    arc_global = get_arc_3points(R, dphi, pos0, **params)
    return arc_global[:, 1:]

def optimize_euler(pos_goal, R, dphi, pos0, weights=None):
    model = lm.Model(get_arc_2points, independent_vars=['R', 'dphi', 'pos0'])
    params = lm.Parameters()
    mi = -180.
    ma = 180.
    iv = 0.
#     mi = 0.
#     ma = 360.
#     iv = 180.
    params.add('Phi2', value=iv, min=mi, max=ma, vary=True)
    params.add('theta2', value=iv, min=mi, max=ma, vary=True)
    params.add('psi2', value=iv, min=mi, max=ma, vary=True)
    result = model.fit(pos_goal[:, 1:], R=R, dphi=dphi, pos0=pos0,
                       params=params, weights=weights, scale_covar=True)
    return result

def find_euler2_interlayer(df_il, weights=weights_default):
    # find coil layer end points that should be connected
    _ = get_endpoint_params(df_il)
    p_in, p_out, p_in0, p_out0, phi0, phi1, phi0_deg, phi1_deg = _
    # find midpoint (in polar coordinates)
    _ = get_midpoint(p_in0, p_out0, phi0, phi1, df_il.RC, df_il.x, df_il.z)
    p_mid, p_mid0 = _
    # find amount to wind
    dphi, dphi_deg = get_dphi(df_il.RC, p_in, p_out)
    # which positions to aim for
    # note p_in is correct by construction -- removed in optimize_euler
    pos_goal = np.array([p_in, p_mid, p_out]).T
    # run optimization
    result = optimize_euler(pos_goal, df_il.RC, dphi, p_in, weights)
    residuals = result.best_fit - pos_goal[:, 1:]
    euler2 = np.array([result.params['Phi2'].value,
                      result.params['theta2'].value,
                      result.params['psi2'].value])
    return euler2, p_in, p_mid, dphi_deg, result, residuals


if __name__=='__main__':
    # which version?
    # version = 13
    version = 14
    # load coils
    paramdir = helicalc_dir + 'dev/params/'
    paramname = f'Mu2e_V{version}'
    geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
    # prep for interlayer calculation
    geom_il = prep_df_il(geom_df)
    # loop through coils that need interlayer connector
    Phi2s = []
    theta2s = []
    psi2s = []
    x0s = []
    y0s = []
    z0s = []
    R0s = []
    dphis = []
    results = []
    residuals_list = []

    for i in range(len(geom_il)):
        df_il = geom_il.iloc[i]
        _ = find_euler2_interlayer(df_il)
        euler2, p_in, p_mid, dphi_deg, result, residuals = _
        # append to appropriate list
        Phi2s.append(euler2[0])
        theta2s.append(euler2[1])
        psi2s.append(euler2[2])
        x0s.append(p_in[0])
        y0s.append(p_in[1])
        z0s.append(p_in[2])
        R0s.append(df_il.RC)
        dphis.append(dphi_deg)
        results.append(result)
        residuals_list.append(residuals)
    # create dataframe with all parameters needed for arc bar integrator
    _ = pd.DataFrame({'cond N': geom_il.Coil_Num,
                     'W': geom_il.w_sc, 'T': geom_il.h_sc, 'I': geom_il.I_turn,
                     'R0': geom_il.RC, 'dphi': dphis,
                     'x0': x0s, 'y0': y0s, 'z0': z0s,
                     'Phi2': Phi2s, 'theta2': theta2s, 'psi2': psi2s})
    df_coil_il = _
    # save results
    df_coil_il.to_csv(helicalc_dir+'dev/params/'+paramname+
                      '_coil_interlayer.txt', index=False)
    # print results, and residuals
    print(df_coil_il)
    for i, result in enumerate(results):
        print(f'Coil {df_coil_il.iloc[i]["cond N"]}:'+
              f' chi2_red = {result.redchi:0.4E}')
