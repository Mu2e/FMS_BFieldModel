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

# output info
output_dir = helicalc_data+'Bmaps/helicalc_validation/'
# coil only map
# full
#save_name = output_dir+'Mau14.DS1_region.standard-helicalc.coil_56_full.pkl'
# y=0 plane
save_name = output_dir+'Mau14.DS_region_plane.standard-helicalc.coil_56-66_full.pkl'

# load coil geometry
paramdir = helicalc_dir + 'dev/params/'
# paramname = 'Mu2e_V13'
paramname = 'Mu2e_V14' # correct

geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()

# load chunk data
chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
df_chunks = pd.read_csv(chunk_file)

# load interlayer geometry
df_dict = load_all_geoms(version=14, return_dict=True)
df_inter = df_dict['interlayers']
#N_chunk_inter = 10000
N_chunk_inter = 500
dxyz_interlayer = dxyz_arc_bar_dict[1]

# load OPERA dataframe for grid to calculate on
opera_file = helicalc_data+'Bmaps/single_coil_Mau13/DSMap_V14_nobus.pkl'
df_O = pd.read_pickle(opera_file)
# y=0 plane
df_O = df_O.query('(Y==0) & (-4.796 <= X <= -2.996)').copy()
# testing
#df_O = df_O[(np.isin(df_O.X, df_O.X.unique()[::16])) & (np.isin(df_O.Z, df_O.Z.unique()[::16]))]
df_O = df_O[(np.isin(df_O.X, df_O.X.unique()[::4])) & (np.isin(df_O.Z, df_O.Z.unique()[::4]))]

def DSi_calc(Coil_Num=56, df=df_O, df_coils=geom_df,
             df_interlayers=df_dict['interlayers'],
             override_chunk=True):
    # specific to coil
    df_ = df_coils.query(f'Coil_Num == {Coil_Num}').copy().iloc[0]
    # test
    # df_ = df_coils.query(f'Coil_Num == {Coil_Num}').copy()#.iloc[0]
    # print(df_)
    # df_ = df_.iloc[0]
    if override_chunk:
        N_chunk_coil = 50
    else:
        N_chunk_coil = df_chunks.query(f'Nt_Ri == {df_.Nt_Ri}').iloc[0].N_field_points
    N_layers = df_.N_layers
    if N_layers > 1:
        df_int = df_interlayers.query(f'`cond N` == {Coil_Num}').copy().iloc[0]
        # kludge for better column naming
        df_int['cond N'] = f'{int(df_int["cond N"])}_il'
        df = DS1_calc(df=df, df_coil=df_, df_interlayer=df_int)
    else:
        df = DS8_calc(df=df, df_coil=df_, N_chunk=N_chunk_coil)
    return df

def DS1_calc(df=df_O, df_coil=geom_df.iloc[0],
             df_interlayer=df_inter.query(f'`cond N` == 56').copy().iloc[0],
             N_chunk=100):
    df_ = df.copy()
    cn = df_coil.Coil_Num
    # loop over two layers
    for layer, dev in zip([1, 2], [1, 2]):
    # test on 1 GPU
    # for layer, dev in zip([1, 2], [1, 1]):
        # create coil
        myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz], layer=layer, dev=dev)
        # integrate on grid and add to dataframe
        df_ = myCoil.integrate_grid(df_, N_batch=N_chunk, tqdm=tqdm)
    # interlayer connect
    # create interlayer
    myArc = ArcIntegrator3D(df_interlayer, dxyz=dxyz_interlayer, dev=3)
    # integrate on grid and add to dataframe
    df_ = myArc.integrate_grid(df_, N_batch=N_chunk_inter, tqdm=tqdm)
    return df_

def DS8_calc(df=df_O, df_coil=geom_df.iloc[0], N_chunk=100):
    df_ = df.copy()
    cn = df_coil.Coil_Num
    # loop over layers (only 1 this time)
    for layer, dev in zip([1], [1]):
        # create coil
        myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz], layer=layer, dev=dev)
        # integrate on grid and add to dataframe
        df_ = myCoil.integrate_grid(df_, N_batch=N_chunk, tqdm=tqdm)
    return df_

def combine_columns(df):
    for i in ['x', 'y', 'z']:
        cols = []
        for col in df.columns:
            if (f'B{i}' in col) & (len(col) > 2):
                cols.append(col)
        eval_str = f'B{i}_helicalc = '+'+'.join(cols)
        df.eval(eval_str, inplace=True, engine='python')
        # Tesla to Gauss
        df.eval(f'B{i} = B{i} * 1e4', inplace=True)
        df.eval(f'B{i}_helicalc = B{i}_helicalc * 1e4', inplace=True)
        df.eval(f'B{i}_delta = B{i}_helicalc - B{i}', inplace=True)
        # delta
    return df


if __name__ == '__main__':
    coils = range(56, 67)
    for cn in coils:
        df_O = DSi_calc(Coil_Num=cn, df=df_O, df_coils=geom_df,
             df_interlayers=df_inter,
             override_chunk=True)
    df_O = combine_columns(df_O)
    df_O.to_pickle(save_name)
    print(df_O.columns)
    print(df_O)
