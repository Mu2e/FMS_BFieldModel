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
plotdir = helicalc_data+'plots/helicalc/validation/current_loop/'

# standard
data_file = helicalc_data+'Bmaps/helicalc_validation/current_loop/Mau14.loop_region.standard-helicalc-solcalc.current_loop_full.pkl'

df = pd.read_pickle(data_file)

def make_side_by_side(df):
    # side by side plots
    # helicalc and OPERA (solenoid) only
    for y in [0.0]:
        for i in ['x', 'y', 'z', '']:
            fig = plt.figure(figsize=(14, 8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.view_init(elev=30., azim=30)
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.view_init(elev=30., azim=30)
            fig = mu2e_plot3d(df, 'X', 'Z', f'B{i}_helicalc', f'-0.9<=X<=0.9 and Y=={y}', units='m', mode='mpl', fig=fig, ax=ax1)
            fig = mu2e_plot3d(df, 'X', 'Z', f'B{i}_gsol', f'-0.9<=X<=0.9 and Y=={y}', units='m', mode='mpl', fig=fig, ax=ax2)
            fig.suptitle(f'B{i} vs X and Z for Current Loop\n-0.9<=X<=0.9, Y=={y}, -0.4<=Z<=0.4')
            ax1.set_xlabel('R (m)')
            ax1.set_ylabel('Z (m)')
            if i == '':
                ax1.set_zlabel('\n\n\n'+rf'$|B|$ [Gauss]')
                ax2.set_zlabel('\n\n\n'+rf'$|B|$ [Gauss]')
            else:
                ax1.set_zlabel('\n\n\n'+rf'$B_{i}$ [Gauss]')
                ax2.set_zlabel('\n\n\n'+rf'$B_{i}$ [Gauss]')
            ax1.set_title('Helicalc\n(Loop with Pitch=0 m)')
            ax2.set_title('OPERA\n(GSOLENOID)')
            ax1.zaxis.labelpad = 10
            ax2.zaxis.labelpad = 10
            fig.savefig(plotdir+f'helicalc/B/loop_B{i}_vs_X_Z_Y_{y:0.2f}_compare.pdf')
            fig.savefig(plotdir+f'helicalc/B/loop_B{i}_vs_X_Z_Y_{y:0.2f}_compare.png')

def make_delta_plots(df, suff):
# delta plots
    df_ = df.copy()
    plotdir_full = plotdir+suff+'/'
    for y in [0.0]:
        for i in ['x', 'y', 'z', '']:
            df_.eval(f'B{i}_delta = B{i}_{suff} - B{i}_gsol', inplace=True)
            fig, ax = plt.subplots()
            fig = mu2e_plot3d(df_, 'X', 'Z', f'B{i}_delta', f'-0.9<=X<=0.9 and Y=={y}', units='m', mode='mpl', ptype='heat', fig=fig, ax=ax)
            fig.suptitle(f'(B{i}_{suff} - B{i}_gsol) vs X and Z for Current Loop\n-0.9<=X<=0.9, Y=={y}, -0.4<=Z<=0.4')
            ax.set_title(None)
            fig.savefig(plotdir_full+f'loop_deltaB{i}_vs_X_Z_Y_{y:0.2f}_compare.pdf')
            fig.savefig(plotdir_full+f'loop_deltaB{i}_vs_X_Z_Y_{y:0.2f}_compare.png')

def make_delta_plots_helicalc_solcalc(df):
# delta plots
    df_ = df.copy()
    plotdir_full = plotdir+'helicalc/solcalc/'
    for y in [0.0]:
        for i in ['x', 'y', 'z', '']:
            df_.eval(f'B{i}_delta = B{i}_helicalc - B{i}_solcalc', inplace=True)
            fig, ax = plt.subplots()
            fig = mu2e_plot3d(df_, 'X', 'Z', f'B{i}_delta', f'-0.9<=X<=0.9 and Y=={y}', units='m', mode='mpl', ptype='heat', fig=fig, ax=ax)
            fig.suptitle(f'(B{i}_helicalc - B{i}_solcalc) vs X and Z for Current Loop\n-0.9<=X<=0.9, Y=={y}, -0.4<=Z<=0.4')
            ax.set_title(None)
            fig.savefig(plotdir_full+f'loop_deltaB{i}_vs_X_Z_Y_{y:0.2f}_compare.pdf')
            fig.savefig(plotdir_full+f'loop_deltaB{i}_vs_X_Z_Y_{y:0.2f}_compare.png')

if __name__=='__main__':
    # helicalc - opera (gsol) side by side
    print('Making side-by-side (helicalc - gsolenoid)')
    make_side_by_side(df)
    print('Finished side-by-side (helicalc - gsolenoid)')
    # delta plots
    for suff in ['helicalc', 'solcalc', '5br', '10br', '50br', '100br']:
        print(f'Making delta ({suff} - gsolenoid)')
        make_delta_plots(df, suff)
        print(f'Finished delta ({suff} - gsolenoid)')
    # special set comparing helicalc and solcalc
    make_delta_plots_helicalc_solcalc(df)
