import torch as tc
import numpy as np

from helicalc.coil import CoilIntegrator
from helicalc.tools import *
from helicalc.geometry import *
from helicalc.integrate import *

import sys

if len(sys.argv) == 1:
    i = 0
else:
    i = int(sys.argv[1])

param_dir = '../dev/params/'
# geom = 'DS_V13'
geom = 'DS_V13_adjusted'
# geom = 'DS_test'

# read in geometry files
geom_df = read_solenoid_geom_combined(param_dir, geom, sep=',', skiprows=1)

CoilIG = CoilIntegrator(geom_df.iloc[i], dxyz=np.array([1e-3, 1e-3, 2e-4/geom_df.iloc[i].Ri]))
# CoilIG = CoilIntegrator(geom_df.iloc[i], dxyz=np.array([1e-3, 1e-3, 5e-4/geom_df.iloc[i].Ri]))
# CoilIG = CoilIntegrator(geom_df.iloc[i], dxyz=np.array([2e-3, 2e-3, 1e-3/geom_df.iloc[i].Ri]))
# CoilIG = CoilIntegrator(geom_df.iloc[i], dxyz=np.array([2e-3, 2e-3, 2e-3/geom_df.iloc[i].Ri]))

try:
    print(f'Estimated Init Mem: {CoilIG.est_mem_init_mb} MB, Actual Init Mem: {CoilIG.actual_mem_init_mb} MB')
    print(f'Estimated / Actual = {CoilIG.est_mem_init_mb/CoilIG.actual_mem_init_mb:0.3f}')
    print(f'Estimated Run Mem: {CoilIG.est_mem_run_mb} MB = {CoilIG.est_mem_run_mb*1e-3:0.3f} GB')
except:
    print('Actual was 0')

CoilIG.integrate(x0=geom_df.iloc[i].x,y0=geom_df.iloc[i].y,z0=geom_df.iloc[i].z)

try:
    print(f'Actual Run Mem: {CoilIG.actual_mem_run_mb} MB = {CoilIG.actual_mem_run_mb*1e-3:0.3f} GB')
    print(f'Estimated / Actual: {CoilIG.est_mem_run_mb / CoilIG.actual_mem_run_mb}')
    print(f'B_calc at center [T]: {CoilIG.last_result}')
except:
    print('Actual was 0')
