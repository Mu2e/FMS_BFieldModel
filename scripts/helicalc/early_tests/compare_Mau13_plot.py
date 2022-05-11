import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from helicalc.tools import config_plots

config_plots()
plt.rcParams['axes.linewidth'] = 2

# plotdir = '/home/ckampa/data/plots/helicalc/testing/coarse/'
# plotdir = '/home/ckampa/data/plots/helicalc/testing/fine/'
# plotdir = '/home/ckampa/data/plots/helicalc/testing/fine_helicity/'
# plotdir = '/home/ckampa/data/plots/helicalc/testing/fine_helicity_noadj/'
# plotdir = '/home/ckampa/data/plots/helicalc/testing/fine_helicity_noadj_helical_only/'
#plotdir = '/home/ckampa/data/plots/helicalc/testing/fine_helicity_helical_only/'
plotdir = '/home/ckampa/data/plots/helicalc/testing/coarse_helicity_helical_only_03-14-22/'

pkldir = '/home/ckampa/data/pickles/helicalc/testing/'
# save_name = 'Helicalc_v00_xz_plane.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine_helicity.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine_helicity_noadj_32bit.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine_helicity_noadj.pkl'
#save_name = 'Helicalc_v00_xz_plane_fine_helicity_helonly.pkl'
save_name = 'Helicalc_v01_xz_plane_coarse_helicity_helonly.pkl'

# xs = [0.]
xs = [-.8, -.4, 0., .4, .8]
Nx = len(xs)

# load Mau13 minus (PS+TS)
# df_true = pd.read_pickle('/home/shared_data/Bmaps/Mau13/subtracted/Mau13_1.00xDS_0.00xPS-TS_DSMap.p')
df_true = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13/DSMap_helical_windings_only.p')
# Mau13 nominal (BAD but just want to see)
# df_true = pd.read_pickle('/home/shared_data/Bmaps/Mau13/subtracted/Mau13_1.00xDS_1.00xPS-TS_DSMap.p')

# load run results
df_run = pd.read_pickle(pkldir+save_name)

# calculate BT
df_true.eval('Bt = (Bx**2+By**2)**(1/2)', inplace=True)
df_run.eval('Bt = (Bx**2+By**2)**(1/2)', inplace=True)

Bs = ['Bx','By','Bz']
NBs = len(Bs)

# plot!
# fig = plt.figure(figsize=(18,16))
# fig = plt.figure(figsize=(30,16))
fig = plt.figure(figsize=(NBs*6,Nx*6))
# gridspec_kw={'height_ratios': 3*[1,.35]}, figsize=(14, 16))
# outer_grid = gridspec.GridSpec(3, 1, )#wspace=0.1)
outer_grid = gridspec.GridSpec(Nx, 1, )#wspace=0.1)
# gs = [gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=outer_grid[i], height_ratios=[1,.4], hspace=0., wspace=.2) for i in range(3)]
gs = [gridspec.GridSpecFromSubplotSpec(2,3, subplot_spec=outer_grid[i], height_ratios=[1,.4], hspace=0., wspace=.2) for i in range(Nx)]

for ig, x in zip(gs, xs):# enumerate(xs):
    # inner_grid = gs[i]
    # ax = plt.subplot(inner_grid)
    # ax = plt.subplot(ig)
    df_true_line = df_true.query(f'X=={x} & Y==0')
    df_run_line = df_run.query(f'X=={x} & Y==0')
    df_true_line_slim = df_true_line[df_true_line['Z'].isin(df_run_line['Z'].unique())]
    for j, B in enumerate(['Bx','By','Bz']):
        dB = df_run_line[B].values - df_true_line_slim[B].values
        # print(len(df_run_line), len(dB))
        ax0 = fig.add_subplot(ig[0,j])
        ax1 = fig.add_subplot(ig[1,j])
        # compare plot
        # ax0.plot(df_true_line.Z, df_true_line[B], linewidth=2, zorder=100, label='Mau13 - (PS+TS)')
        ax0.plot(df_true_line.Z, df_true_line[B], linewidth=2, zorder=100, label='Mau13 (helical coils only)')
        ax0.scatter(df_run_line.Z, df_run_line[B], color='red', s=15, zorder=105, label='Helicalc (no bus bars)')
        ax0.set(ylabel=f'{B} [Gauss]', title=f'{B} vs. Z: X=={x:.2f}, Y==0 [m]')
        ax0.set_xticklabels([])
        # ax0.get_xaxis().set_visible(False)
        ax0.set(xticks=np.linspace(3,16, 14), xlim=(2.5,14.5))
        ax0.legend()
        # residuals
        ax1.plot([2.5,14.5], [0,0], 'k-', zorder=100, linewidth=0.5)
        # ax1.scatter(df_run_line.Z, df_run_line[f'd{B}'], color='red', s=15, zorder=105, label='Helicalc (no bus bars)')
        ax1.scatter(df_run_line.Z, dB, color='red', s=15, zorder=105, label='Helicalc (no bus bars)')
        # axs[i*2+1, j].set(xlabel='Z [m]', ylabel=r'$\Delta$'+f'{B} (Helicalc - Mau13) [Gauss]')
        yabs_max = abs(max(ax1.get_ylim(), key=abs))
        ax1.set(xlabel='Z [m]', ylabel=r'$\Delta$'+f'{B} [Gauss]')
        # ax1.set(xticks=np.linspace(3,16, 14))
        ax1.set(xticks=np.linspace(3,16, 14), xlim=(2.5,14.5), ylim=(-yabs_max, yabs_max))

        # move residual plot up
        # pos1 = axs[i*2+1, j].get_position()
        # pos2 = [pos1.x0, pos1.y0 + 0.1, pos1.width, pos1.height]
        # axs[i*2+1, j].set_position(pos2)

# plt.tight_layout(pad=2)
fig.tight_layout()
name = 'Mau13_vs_Helicalc_nobus'
fig.savefig(plotdir+name+'.pdf')
fig.savefig(plotdir+name+'.png')


'''
fig, axs = plt.subplots(6, 3, gridspec_kw={'height_ratios': 3*[1,.35]}, figsize=(14, 16))
for i, x in enumerate(xs):
    df_true_line = df_true.query(f'X=={x} & Y==0')
    df_run_line = df_run.query(f'X=={x} & Y==0')
    for j, B in enumerate(['Bx','By','Bz']):
        # compare plot
        axs[i*2, j].plot(df_true_line.Z, df_true_line[B], linewidth=2, zorder=100, label='Mau13 - (PS+TS)')
        axs[i*2, j].scatter(df_run_line.Z, df_run_line[B], color='red', s=15, zorder=105, label='Helicalc (no bus bars)')
        # axs[i*2, j].set(xlabel='Z [m]', ylabel=f'{B} [Gauss]', title=f'{B} vs. Z: X=={x:.2f}, Y==0 [m]')
        axs[i*2, j].set(ylabel=f'{B} [Gauss]', title=f'{B} vs. Z: X=={x:.2f}, Y==0 [m]')
        axs[i*2, j].get_xaxis().set_visible(False)
        axs[i*2, j].legend()
        # residuals
        axs[i*2+1, j].scatter(df_run_line.Z, df_run_line[f'd{B}'], color='red', s=15, zorder=105, label='Helicalc (no bus bars)')
        # axs[i*2+1, j].set(xlabel='Z [m]', ylabel=r'$\Delta$'+f'{B} (Helicalc - Mau13) [Gauss]')
        axs[i*2+1, j].set(xlabel='Z [m]', ylabel=r'$\Delta$'+f'{B} [Gauss]')
        # move residual plot up
        # pos1 = axs[i*2+1, j].get_position()
        # pos2 = [pos1.x0, pos1.y0 + 0.1, pos1.width, pos1.height]
        # axs[i*2+1, j].set_position(pos2)

fig.tight_layout()
name = 'Mau13_vs_Helicalc_nobus'
fig.savefig(plotdir+name+'.pdf')
fig.savefig(plotdir+name+'.png')
'''


'''
for x in xs:
    df_true_line = df_true.query(f'X=={x} & Y==0')
    df_run_line = df_run.query(f'X=={x} & Y==0')
    for B in ['Bx', 'By', 'Bz', 'B', 'dBx', 'dBy', 'dBz', 'dB']:
        fig = plt.figure()
        if B[0] != 'd':
            plt.plot(df_true_line.Z, df_true_line[B], linewidth=2, zorder=100, label='Mau13 - (PS+TS)')
        plt.scatter(df_run_line.Z, df_run_line[B], color='red', s=15, zorder=105, label='Helicalc (no bus bars)')
        plt.xlabel('Z [m]')
        if B[0] == 'd':
            plt.ylabel(f'{B} (Helicalc - Mau13) [Gauss]')
        else:
            plt.ylabel(f'{B} [Gauss]')
            plt.legend()
        fig.tight_layout()
        name = plotdir+f'{B}_vs_Z_X{x:.2f}'
        fig.savefig(name+'.pdf')
        fig.savefig(name+'.png')
'''
