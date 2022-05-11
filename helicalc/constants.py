'''
Constants useful for Biot-Savart calculations
'''
import math
import numpy as np
from scipy.constants import mu_0

mu0 = mu_0 # 4*pi*1e-7    # permeability of free space

MAXMEM = 11019. # max memory per GPU, in MB
# FIXME! Should be able to get devices from pytorch
DEVICES = [0, 1, 2, 3] # GPU device numbers

# Mu2e Bmap coordinate definitions
# taken from Mau13 in CVMFS
PS_grid = {'X0':2.804, 'Y0':-1.200, 'Z0':-9.929,
           'nX':89, 'nY':97, 'nZ':281,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
# incorrect TSu grid!! X0 is wrong
#TSu_grid = {'X0':4.000, 'Y0':-1.200, 'Z0':-2.929,
#            'nX':201, 'nY':97, 'nZ':149,
#            'dX':0.025, 'dY':0.025, 'dZ':0.025}
TSu_grid = {'X0':0.004, 'Y0':-1.200, 'Z0':-2.929,
           'nX':201, 'nY':97, 'nZ':149,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
TSd_grid = {'X0':-5.096, 'Y0':-1.200, 'Z0':-0.829,
            'nX':205, 'nY':97, 'nZ':157,
            'dX':0.025, 'dY':0.025, 'dZ':0.025}
DS_grid = {'X0':-5.096, 'Y0':-1.200, 'Z0':3.071,
           'nX':97, 'nY':97, 'nZ':521,
           'dX':0.025, 'dY':0.025, 'dZ':0.025}
# additions outside solenoid regions
# note that for now we do not do "flipy", so should include Y<0 in grid
PStoDumpArea_grid = {'X0':0.004, 'Y0':-5.500, 'Z0':-14.929,
                     'nX':73, 'nY':111, 'nZ':51,
                     'dX':0.100, 'dY':0.100, 'dZ':0.100}
# "flipy"
ProtonDumpArea_grid = {'X0':-0.796, 'Y0':-5.600, 'Z0':-20.929,
                       'nX':20, 'nY':57, 'nZ':31,
                       'dX':0.200, 'dY':0.200, 'dZ':0.200}

# 2D cylindrical grid
DS_cyl2d_grid_5mm = {'X0':-3.904, 'Y0':0., 'Z0':4.101,
                     'nX':161, 'nY':1, 'nZ':1881,
                     'dX':0.005, 'dY':0.005, 'dZ':0.005}

# integrator specific constants
## HELICALC / COILS
# dxyz for helicalc (nominal values for Mu2e DS coils)
# radius is hard coded for now.
# coarse integration grid (3x3 in cross section, 1/(5cm) in R*phi)
dxyz_dict_coarse = {1: np.array([3e-3,1e-3, 5e-2/1.05]),
                    2: np.array([2e-3,1e-3, 5e-2/1.05])}
# fine integration grid (3x3 in cross section, 1/(1cm) in R*phi)
dxyz_dict = {1: np.array([3e-3,1e-3, 1e-2/1.05]),
             2: np.array([2e-3,1e-3, 1e-2/1.05])}

# dictionary for which coils & layers on which GPU (for full run)
# key is device, value is a list of dictionaries with coil + layer
helicalc_GPU_dict = {0: [{'coil': 56, 'layer': 1, 'name': 'DS-1'}, {'coil': 56, 'layer': 2, 'name': 'DS-1'},
                         {'coil': 60, 'layer': 1, 'name': 'DS-5'}, {'coil': 63, 'layer': 1, 'name': 'DS-8'}],
                     1: [{'coil': 57, 'layer': 1, 'name': 'DS-2'}, {'coil': 57, 'layer': 2, 'name': 'DS-2'},
                         {'coil': 60, 'layer': 2, 'name': 'DS-5'}, {'coil': 64, 'layer': 1, 'name': 'DS-9'}],
                     2: [{'coil': 58, 'layer': 1, 'name': 'DS-3'}, {'coil': 58, 'layer': 2, 'name': 'DS-3'},
                         {'coil': 61, 'layer': 1, 'name': 'DS-6'}, {'coil': 65, 'layer': 1, 'name': 'DS-10'}],
                     3: [{'coil': 59, 'layer': 1, 'name': 'DS-4'}, {'coil': 59, 'layer': 2, 'name': 'DS-4'},
                         {'coil': 61, 'layer': 2, 'name': 'DS-6'},
                         {'coil': 62, 'layer': 1, 'name': 'DS-7'}, {'coil': 62, 'layer': 2, 'name': 'DS-7'},
                         {'coil': 66, 'layer': 1, 'name': 'DS-11'}, {'coil': 66, 'layer': 2, 'name': 'DS-11'},],
                    }
## STRAIGHT BAR
# dxyz for straight bars (nominal values for Mu2e bars)
# fine integration grid (3x3 in cross section, 1/(0.5cm) in length)
dxyz_straight_bar_dict = {1: np.array([1e-3,3e-3, 5e-3]),
                          2: np.array([1e-3,2e-3, 5e-3])}
## ARC BAR
# dz (dphi) assumes R=1. Adjust this when using if this is not the case.
dxyz_arc_bar_dict = {1: np.array([1e-3,3e-3, 5e-3]),
                     2: np.array([1e-3,2e-3, 5e-3])}
