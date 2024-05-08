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
save_name = output_dir+'Mau14.DS1_region_plane.standard-helicalc.coil_56_full.pkl'

# load coil geometry
paramdir = helicalc_dir + 'dev/params/'
# paramname = 'Mu2e_V13'
paramname = 'Mu2e_V14' # correct

geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
df_DS1 = geom_df.query('Coil_Num == 56').copy().iloc[0]

# load chunk data
chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
df_chunks = pd.read_csv(chunk_file)
N_chunk_coil = df_chunks.query(f'Nt_Ri == {df_DS1.Nt_Ri}').iloc[0].N_field_points

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


def DS1_calc(df=df_O, df_coil=df_DS1, df_interlayer=df_DS1_IL, outfile=save_name):
    df_ = df.copy()
    # loop over two layers
    for layer, dev in zip([1, 2], [1, 2]):
        # create coil
        myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz], layer=layer, dev=dev)
        # integrate on grid and add to dataframe
        df_ = myCoil.integrate_grid(df_, N_batch=N_chunk_coil, tqdm=tqdm)
    # interlayer connect
    # create interlayer
    myArc = ArcIntegrator3D(df_interlayer, dxyz=dxyz_interlayer, dev=3)
    # integrate on grid and add to dataframe
    df_ = myArc.integrate_grid(df_, N_batch=N_chunk_inter, tqdm=tqdm)
    # add coil components
    for i in ['x', 'y', 'z']:
        df_.eval(f'B{i}_helicalc = B{i}_helicalc_c56_l1 + B{i}_helicalc_c56_l2 + B{i}_bus_arc_cn_56_il', inplace=True)
        # Tesla to Gauss
        df_.eval(f'B{i} = B{i} * 1e4', inplace=True)
        df_.eval(f'B{i}_helicalc = B{i}_helicalc * 1e4', inplace=True)
        df_.eval(f'B{i}_delta = B{i}_helicalc - B{i}', inplace=True)
    # save
    df_.to_pickle(outfile)
    return df_


if __name__ == '__main__':
    df_result = DS1_calc(df_O, df_DS1, df_DS1_IL, save_name)
    print(df_result.columns)
    print(df_result)
