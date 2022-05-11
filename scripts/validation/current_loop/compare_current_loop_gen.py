'''
There are several things to compare here. Most of the work in generation is bookkeeping.
1. OPERA GSOLENOID (should be correct, no parabolic approx.)
2. OPERA BR20 - 5 bricks
3. OPERA BR20 - 10 bricks
4. OPERA BR20 - 50 bricks
5. OPERA BR20 - 100 bricks
6. helicalc.coil (helical coil)
7. helicalc.solcalc (ideal solenoid) -- need geometry file
'''
import sys
import numpy as np
import pandas as pd
from helicalc import helicalc_dir, helicalc_data
from helicalc.coil import CoilIntegrator
from helicalc.solcalc import SolCalcIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.constants import dxyz_dict
from tqdm import tqdm

# output info
output_dir = helicalc_data+'Bmaps/helicalc_validation/current_loop/'
save_name = output_dir+'Mau14.loop_region.standard-helicalc-solcalc.current_loop_full.pkl'

# load coil geometry
paramdir = helicalc_dir + 'dev/params/'
paramname_helicalc = 'current_loop_v2'
paramname_solcalc = 'current_loop_solcalc'

# helicalc
geom_df_hel = read_solenoid_geom_combined(paramdir,paramname_helicalc)
df_hel = geom_df_hel.copy().iloc[0]
# kludge to remove pitch
df_hel.pitch_bar = 0.
# solcalc
geom_df_sol = read_solenoid_geom_combined(paramdir,paramname_solcalc)
df_sol = geom_df_sol.copy().iloc[0]
drz = np.array([2e-3,1e-3])

# load chunk data
# chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
# df_chunks = pd.read_csv(chunk_file)
# N_chunk_coil = df_chunks.query(f'Nt_Ri == {df_DS8.Nt_Ri}').iloc[0].N_field_points
N_chunk = 1000

# load OPERA dataframe for grid to calculate on
opera_file_gsol = helicalc_data+'Bmaps/current_loop/current_loop_map_gsolenoid.pkl'
opera_file_5br = helicalc_data+'Bmaps/current_loop/current_loop_map_5bricks.pkl'
opera_file_10br = helicalc_data+'Bmaps/current_loop/current_loop_map_10bricks.pkl'
opera_file_50br = helicalc_data+'Bmaps/current_loop/current_loop_map_50bricks.pkl'
opera_file_100br = helicalc_data+'Bmaps/current_loop/current_loop_map_100bricks.pkl'

opera_file_dict = {'gsol': opera_file_gsol,
                   '5br': opera_file_5br,
                   '10br': opera_file_10br,
                   '50br': opera_file_50br,
                   '100br': opera_file_100br,}

def load_OPERA_dfs(OPERA_file_dict, query_str='(Y==0.)'):
    columns_save = ['X', 'Y', 'Z']
    for i, item in enumerate(OPERA_file_dict.items()):
        suff, fname = item
        if i == 0:
            df = pd.read_pickle(fname)
            for B in ['Bx', 'By', 'Bz']:
                df.eval(f'{B}_{suff} = {B}', inplace=True)
                columns_save.append(f'{B}_{suff}')
        else:
            df_ = pd.read_pickle(fname)
            for B in ['Bx', 'By', 'Bz']:
                df.loc[:, f'{B}_{suff}'] = df_[B]#.values
                columns_save.append(f'{B}_{suff}')
    df = df[columns_save].copy()
    if query_str is not None:
        df = df.query(query_str).copy()
    return df

def helicalc_calc(df, df_helicalc):
    df_ = df.copy()
    # create coil
    myCoil = CoilIntegrator(df_helicalc, dxyz=dxyz_dict[df_helicalc.dxyz], layer=1, dev=1)
    # integrate on grid and add to dataframe
    df_ = myCoil.integrate_grid(df_, N_batch=N_chunk, tqdm=tqdm)
    # add coil components
    for i in ['x', 'y', 'z']:
        df_.eval(f'B{i}_helicalc = B{i}_helicalc_c63_l1', inplace=True)
        df_.drop(f'B{i}_helicalc_c63_l1', axis=1, inplace=True)
    return df_

def solcalc_calc(df, df_solcalc):
    df_ = df.copy()
    # create coil
    mySolCalc = SolCalcIntegrator(df_solcalc, drz=drz, use_basic_geom=True,)
    # integrate on grid and add to dataframe
    df_ = mySolCalc.integrate_grid(df_, tqdm=tqdm)
    # add coil components
    for i in ['x', 'y', 'z']:
        df_.eval(f'B{i}_solcalc = B{i}_solcalc_63', inplace=True)
        df_.drop(f'B{i}_solcalc_63', axis=1, inplace=True)
    return df_

def rescale_df(df):
    df = df.copy()
    for i in ['x', 'y', 'z']:
        for suff in ['gsol', '5br', '10br', '50br', '100br', 'helicalc', 'solcalc']:
            df.eval(f'B{i}_{suff} = B{i}_{suff} * 1e4', inplace=True)
            df.eval(f'B_{suff} = (Bx_{suff}**2+By_{suff}**2+Bz_{suff}**2)**(1/2)', inplace=True)
    return df

if __name__ == '__main__':
    df_O = load_OPERA_dfs(opera_file_dict, query_str=None)
    # print(df_O)
    # print(df_O.columns)
    df_O = helicalc_calc(df_O, df_hel)
    df_O = solcalc_calc(df_O, df_sol)
    df_O = rescale_df(df_O)
    df_O.to_pickle(save_name)
    print(df_O)
    print(df_O.columns)
    # print(df_result.columns)
    # print(df_result)
