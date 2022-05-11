import os
import sys
import numpy as np
import pandas as pd
from helicalc import helicalc_dir, helicalc_data
from helicalc.coil import CoilIntegrator
from helicalc.busbar import ArcIntegrator3D
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.solenoid_geom_funcs import load_all_geoms
from helicalc.constants import dxyz_dict, dxyz_arc_bar_dict
from tqdm import tqdm

# interlayer connect
sys.path.append(os.path.join(helicalc_dir, 'scripts/geometry/'))
from make_interlayer_geoms import prep_df_il, find_euler2_interlayer

# deltaR hard code
deltaR=-2.313e-3 # using best result from DS-8
# deltaR=-3.472e-3 # test 2, scaling based on T parameter of conductor
# deltaR=-2.5e-3 # test 3, small bump to smaller R from DS-8
# deltaR=-2.1e-3 # test 4, small bump to larger R from DS-8
deltaR_mm = deltaR * 1e3

# output info
output_dir = helicalc_data+'Bmaps/helicalc_validation/optimize/'
# coil only map
# full
#save_name = output_dir+'Mau14.DS1_region.standard-helicalc.coil_56_full.pkl'
# y=0 plane
#save_name = output_dir+'Mau14.DS1_region_plane.standard-helicalc.coil_56_full.pkl'
#save_name = output_dir+f'Mau14.DS1_region_plane_sparse.coil_dR_{deltaR_mm:0.3f}mm-helicalc.coil_56_full.pkl'
save_name = output_dir+f'Mau14.DS1_region_plane.coil_dR_{deltaR_mm:0.3f}mm-helicalc.coil_56_full.pkl'

# load coil geometry
paramdir = helicalc_dir + 'dev/params/'
# paramname = 'Mu2e_V13'
paramname = 'Mu2e_V14' # correct

geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
# series
# df_DS1 = geom_df.query('Coil_Num == 56').copy().iloc[0]
# dataframe
df_DS1 = geom_df.query('Coil_Num == 56').copy()#.iloc[0]
# TEST! Round Z location
## df_DS1.z = round(df_DS1.z, )

# load chunk data
chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
df_chunks = pd.read_csv(chunk_file)
# if df_DS1 is a series
# N_chunk_coil = df_chunks.query(f'Nt_Ri == {df_DS1.Nt_Ri}').iloc[0].N_field_points
# if df_DS1 is a dataframe
N_chunk_coil = df_chunks.query(f'Nt_Ri == {df_DS1.Nt_Ri.iloc[0]}').iloc[0].N_field_points

# load interlayer geometry
df_dict = load_all_geoms(version=14, return_dict=True)
df_DS1_IL = df_dict['interlayers'].query('`cond N` == 56').copy().iloc[0]
# kludge for better column naming
df_DS1_IL['cond N'] = f'{int(df_DS1_IL["cond N"])}_il'
dxyz_interlayer = dxyz_arc_bar_dict[1]

N_chunk_inter = 10000

# load OPERA dataframe for grid to calculate on
opera_file = helicalc_data+'Bmaps/single_coil_Mau13/DSMap_V14_DS1only.pkl'
df_O = pd.read_pickle(opera_file)
# y=0 plane
df_O = df_O.query('(Y==0) & (-4.796 <= X <= -2.996)').copy()
#df_O = df_O[(np.isin(df_O.X, df_O.X.unique()[::4])) & (np.isin(df_O.Z, df_O.Z.unique()[::4]))]
df_O.reset_index(drop=True, inplace=True)
#print(df_O)

def DS1_calc(df=df_O, df_coil=df_DS1, df_interlayer=df_DS1_IL, outfile=save_name):
    df_ = df.copy()
    # loop over two layers
    for layer, dev in zip([1, 2], [1, 2]):
        # create coil
        myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz], layer=layer, dev=dev,)# lib=np, int_func=np.trapz)
        # integrate on grid and add to dataframe
        df_ = myCoil.integrate_grid(df_, N_batch=N_chunk_coil, tqdm=tqdm)
    # interlayer connect
    # TEST!!
    # comment out
    # create interlayer
    myArc = ArcIntegrator3D(df_interlayer, dxyz=dxyz_interlayer, dev=3,)# lib=np, int_func=np.trapz)
    # integrate on grid and add to dataframe
    df_ = myArc.integrate_grid(df_, N_batch=N_chunk_inter, tqdm=tqdm)
    #####
    # add coil components
    for i in ['x', 'y', 'z']:
        df_.eval(f'B{i}_helicalc = B{i}_helicalc_c56_l1 + B{i}_helicalc_c56_l2 + B{i}_bus_arc_cn_56_il', inplace=True)
        # TEST!!
        # df_.eval(f'B{i}_helicalc = B{i}_helicalc_c56_l1 + B{i}_helicalc_c56_l2', inplace=True)
        # Tesla to Gauss
        df_.eval(f'B{i} = B{i} * 1e4', inplace=True)
        df_.eval(f'B{i}_helicalc = B{i}_helicalc * 1e4', inplace=True)
        df_.eval(f'B{i}_delta = B{i}_helicalc - B{i}', inplace=True)
    # save
    df_.to_pickle(outfile)
    return df_

def interlayer_make_df(geom_il):
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
    return df_coil_il


def run_deltaR(df=df_O, df_coil=df_DS1, deltaR=deltaR, outfile=save_name):
    # note df_coil must be a dataframe! otherwise interlayer function prep doesn't work
    # change coil radius
    df_c = df_coil.copy()
    #df_c.Ri += deltaR # unnecessary
    df_c.rho0_a += deltaR
    # maker interlayer geom with altered coil radius
    df_il = prep_df_il(df_c)
    df_il_eul = interlayer_make_df(df_il)
    # now convert dataframes to series
    df_c = df_c.iloc[0]
    df_il_eul = df_il_eul.iloc[0]
    df_il_eul['cond N'] = f'{int(df_il_eul["cond N"])}_il'
    #print(df_c)
    #print(df_il_eul)
    df_result = DS1_calc(df, df_c, df_il_eul, outfile)
    # TEST! don't change IL
    # df_result = DS1_calc(df, df_c, df_DS1_IL, outfile)
    return df_result

if __name__ == '__main__':
    df_result = run_deltaR(df_O, df_DS1, deltaR, save_name)
    print(df_result.columns)
    print(df_result)
