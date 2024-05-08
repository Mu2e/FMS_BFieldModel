import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from helicalc import helicalc_data
from helicalc.tools import config_plots
config_plots()
# face color issue
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'

from mu2e.mu2eplots import mu2e_plot3d

# set up directories, load data
# standard
# plotdir = helicalc_data+'plots/helicalc/validation/DS1/'
# radius optimize
plotdir = helicalc_data+'plots/helicalc/validation/DS1/optimize/'

# standard
# data_file = helicalc_data+'Bmaps/helicalc_validation/Mau14.DS1_region_plane.standard-helicalc.coil_56_full.pkl'
# radius optimize
# deltaR=-2.313e-3 # using best result from DS-8
# deltaR=-3.472e-3 # test 2, scaling based on T parameter of conductor
# deltaR=-2.5e-3 # test 3, small bump to smaller R from DS-8
deltaR=-2.1e-3 # test 4, small bump to larger R from DS-8
deltaR_mm = deltaR * 1e3
#data_file = helicalc_data+f'Bmaps/helicalc_validation/optimize/Mau14.DS1_region_plane_sparse.coil_dR_{deltaR_mm:0.3f}mm-helicalc.coil_56_full.pkl'
data_file = helicalc_data+f'Bmaps/helicalc_validation/optimize/Mau14.DS1_region_plane.coil_dR_{deltaR_mm:0.3f}mm-helicalc.coil_56_full.pkl'

df = pd.read_pickle(data_file)

# side by side plots
for y in [0.0]:
    for i in ['x', 'y', 'z']:
        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.view_init(elev=30., azim=30)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.view_init(elev=30., azim=30)
        # fig = mu2e_plot3d(df, 'X', 'Z', f'B{i}', f'-0.9<=X<=0.9 and Y=={y} and 3.348<=Z<=4.149', units='m', mode='mpl', fig=fig, ax=ax1)
        # fig = mu2e_plot3d(df, 'X', 'Z', f'B{i}', f'-0.9<=X<=0.9 and Y=={y} and 3.348<=Z<=4.149', units='m', mode='mpl', fig=fig, ax=ax2)
        fig = mu2e_plot3d(df, 'X', 'Z', f'B{i}_helicalc', f'Y=={y}', units='m', mode='mpl', fig=fig, ax=ax1)
        fig = mu2e_plot3d(df, 'X', 'Z', f'B{i}', f'Y=={y}', units='m', mode='mpl', fig=fig, ax=ax2)
        fig.suptitle(f'B{i} vs X and Z for DS-1\n-0.9<=X<=0.9, Y=={y}, 3.348<=Z<=4.149')
        ax1.set_xlabel('R (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('Helicalc\n(Helical Coil + Interlayer Connect)')
        ax2.set_title('OPERA\n(Helical Coil + Interlayer Connect)')
        fig.savefig(plotdir+f'DS1_B{i}_vs_X_Z_Y_{y:0.2f}_compare.pdf')
        fig.savefig(plotdir+f'DS1_B{i}_vs_X_Z_Y_{y:0.2f}_compare.png')

# delta plots
for y in [0.0]:
    for i in ['x', 'y', 'z']:
        fig, ax = plt.subplots()
        fig = mu2e_plot3d(df, 'X', 'Z', f'B{i}_delta', f'Y=={y}', units='m', mode='mpl', ptype='heat', fig=fig, ax=ax)
        fig.suptitle(f'(B{i}_helicalc - B{i}_OPERA) vs X and Z for DS-1\n-0.9<=X<=0.9, Y=={y}, 3.348<=Z<=4.149')
        ax.set_title(None)
        #         ax.set_xlabel('R (m)')
#         ax.set_ylabel('Z (m)')
        fig.savefig(plotdir+f'DS1_deltaB{i}_vs_X_Z_Y_{y:0.2f}_compare.pdf')
        fig.savefig(plotdir+f'DS1_deltaB{i}_vs_X_Z_Y_{y:0.2f}_compare.png')
