import numpy as np
import pandas as pd
from tqdm import tqdm
from helicalc import helicalc_dir, helicalc_data
from helicalc.coil import CoilIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.tools import generate_cartesian_grid_df
from helicalc.constants import dxyz_dict, dxyz_dict_coarse, DS_grid

# which coil number?
# Coil_Num = 56 # type 1 (long double layer)
# Coil_Num = 60 # type 2 (short double layer)
Coil_Num = 63 # type 3 (single layer)
Layer = 1
# which GPU?
Dev = 0

# load coils
paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13'

geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
df_coil = geom_df.query(f'Coil_Num == {Coil_Num}').iloc[0]

# find correct chunk size
chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
df_chunks = pd.read_csv(chunk_file)
N_calc = df_chunks.query(f'Nt_Ri == {df_coil.Nt_Ri}').iloc[0].N_field_points
# OVERRIDE
#N_calc = 100
#N_calc = 300
# N_calc -= 20
print(f'N_calc = {N_calc}, of type: {type(N_calc)}')

# set up grid
df = generate_cartesian_grid_df(DS_grid)
df_ = df.iloc[:10000]

# coils
#myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz], layer=Layer, dev=Dev, interlayer_connect=True)
myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict_coarse[df_coil.dxyz], layer=Layer, dev=Dev, interlayer_connect=True)
# check not including the interlayer connect
# myCoil_nic = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz], layer=Layer, dev=Dev, interlayer_connect=False)

print('Everything is loaded...')
print('Integrating!')

myCoil.integrate_grid(df_, N_batch=N_calc, tqdm=tqdm)
myCoil.save_grid_calc(savetype='pkl', savename=f'Bmaps/helicalc_partial/'+
                      f'Mau13.DS_region.test-helicalc.coil{Coil_Num}_layer{Layer}',
                      all_helicalc_cols=False)

print('Finished integrating and saving!')
