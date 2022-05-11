import sys
import numpy as np
import pandas as pd
from helicalc import helicalc_dir, helicalc_data
from helicalc.coil import CoilIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.constants import dxyz_dict
from tqdm import tqdm

# output info
output_dir = helicalc_data+'Bmaps/helicalc_validation/'
# coil only map
# full
#save_name = output_dir+'Mau14.DS1_region.standard-helicalc.coil_56_full.pkl'
# y=0 plane
#save_name = output_dir+'Mau14.DS8_region_plane.standard-helicalc.coil_63_full.pkl'
save_name = output_dir+'Mau13.DS8_region_plane.standard-helicalc.coil_63_full.pkl'

# load coil geometry
paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13' # correct
# paramname = 'Mu2e_V14'

geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
df_DS8 = geom_df.query('Coil_Num == 63').copy().iloc[0]

# load chunk data
chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
df_chunks = pd.read_csv(chunk_file)
N_chunk_coil = df_chunks.query(f'Nt_Ri == {df_DS8.Nt_Ri}').iloc[0].N_field_points

# load OPERA dataframe for grid to calculate on
opera_file = helicalc_data+'Bmaps/single_coil_Mau13/DS-8-map-corr.pkl'
df_O = pd.read_pickle(opera_file)
# y=0 plane
df_O = df_O.query('(Y==0) & (-4.796 <= X <= -2.996)').copy() # +- 0.9
# df_O = df_O.query('(Y==0) & (-4.696 <= X <= -3.096)').copy() # +- 0.8


def DS8_calc(df=df_O, df_coil=df_DS8, outfile=save_name):
    df_ = df.copy()
    # loop over layers (only 1 this time)
    for layer, dev in zip([1], [1]):
        # create coil
        myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz], layer=layer, dev=dev)
        # integrate on grid and add to dataframe
        df_ = myCoil.integrate_grid(df_, N_batch=N_chunk_coil, tqdm=tqdm)
    # add coil components
    for i in ['x', 'y', 'z']:
        df_.eval(f'B{i}_helicalc = B{i}_helicalc_c63_l1', inplace=True)
        # Tesla to Gauss
        df_.eval(f'B{i} = B{i} * 1e4', inplace=True)
        df_.eval(f'B{i}_helicalc = B{i}_helicalc * 1e4', inplace=True)
        df_.eval(f'B{i}_delta = B{i}_helicalc - B{i}', inplace=True)
    # save
    df_.to_pickle(outfile)
    return df_


if __name__ == '__main__':
    df_result = DS8_calc(df_O, df_DS8, save_name)
    print(df_result.columns)
    print(df_result)
