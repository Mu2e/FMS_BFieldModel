import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.gridspec as gridspec
from helicalc.tools import config_plots

config_plots()
# plt.rcParams['axes.linewidth'] = 2

plotdir = '/home/ckampa/data/plots/helicalc/testing/loop_current/'

pkldir = '/home/ckampa/data/pickles/helicalc/testing/'
# save_name = 'Loop_Current_Test_Stream.pkl'
save_name = 'Loop_Current_Test_Stream2.pkl'

df = pd.read_pickle(pkldir+save_name)
# remove large radius
# x0 = 1.05
# R = 0.05 # 0.9
# z0 = 0.
# L = 0.05
# df = df.query(f'X > {x0-R} & X < {x0+R} & Z > {z0-L} & Z < {z0+L}').copy()


# def make_stream(df, x='Z', y='X', vec=['Bz_br','Bx_br']):
def make_stream(df, x='Z', y='X', vec=['Bz_calc','Bx_calc']):
# def make_stream(df, x='Z', y='X', vec=['Bz','Bx']):
    df_ = df.copy()
    # df_.sort_values(by=[x, y])
    df_.sort_values(by=[y, x])
    xs = df_[x].unique()
    ys = df_[x].unique()
    lx = len(df_[x].unique())
    ly = len(df_[y].unique())
    # VX = df_[vec[0]].values.reshape(lx,ly)
    # VY = df_[vec[1]].values.reshape(lx,ly)
    VX = df_[vec[0]].values.reshape(ly,lx)
    VY = df_[vec[1]].values.reshape(ly,lx)
    print(VX.shape, VY.shape, lx, ly)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = 2 * np.log(np.hypot(VX,VY))
    # color = np.hypot(VX,VY)
    # sp = ax.streamplot(df_[x].unique(), df_[y].unique(), VX, VY, color=color, linewidth=1, cmap=plt.cm.viridis,
    #                    density=1., arrowstyle='->', arrowsize=1.5, )#label='B, XZ plane')
    rp = ax.add_artist(Rectangle([-.00117, 1.0579], .00234, .00525, color='red', label='Conductor Cross-Section', zorder=20))
    # quiver
    C = 1e4*(df_[vec[0]]**2 + df_[vec[1]]**2)**(1/2)
    qp = ax.quiver(df_[x], df_[y], df_[vec[0]], df_[vec[1]], C, zorder=25, pivot='mid')
    cb = plt.colorbar(qp)
    cb.set_label(r"$|B|_{\mathrm{Z,X plane}}$ [Gauss]")

    plt.legend(handles=[rp])
    # plt.xlim([-.0021, .0021])
    plt.xlim([-.0051, .0051])
    plt.ylim([1.055, 1.067])
    plt.xlabel(x+' [m]')
    plt.ylabel(y+' [m]')
    ax.set_aspect('equal')
    # plt.set_aspect('equal')
    # fig.savefig(plotdir+'stream_plot_conductor_02.pdf')
    # fig.savefig(plotdir+'stream_plot_conductor_02.png')
    fig.savefig(plotdir+'quiver_plot_conductor_02.pdf')
    fig.savefig(plotdir+'quiver_plot_conductor_02.png')

make_stream(df)

'''
def make_pcolormesh(df, x='Z', y='X', scale=1e4, c='dB_SOL_BR', c_label=r'$\Delta B$ [Gauss] (Opera SOL - Opera BR20)', title='Opera "Solenoid" vs. Opera "Brick", Y==0 [m]'):
    # sort values and reshape
    lx = len(df[x].unique())
    ly = len(df[y].unique())
    df_ = df.sort_values(by=[x, y])
    X = df_[x].values.reshape(lx, ly)
    Y = df_[y].values.reshape(lx, ly)
    C = scale*df_[c].values.reshape(lx, ly)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    pc = plt.pcolormesh(X, Y, C, shading='nearest')
    cb = plt.colorbar(pc)
    cb.set_label(c_label)
    plt.xlabel(x + ' [m]')
    plt.ylabel(y + ' [m]')
    plt.title(title)
    fig.get_axes()[0].set_aspect('auto')
    fig.savefig(plotdir+c+'_pcolormesh.pdf')
    fig.savefig(plotdir+c+'_pcolormesh.png')

for i in ['', 'x','y','z']:
    cs = [f'dB{i}_SOL_BR', f'dB{i}_HEL_SOL', f'dB{i}_HEL_BR']
    if i != '':
        c_labels = [rf'$\Delta B_{i}$ [Gauss] (Opera SOL - Opera BR20)', rf'$\Delta B_{i}$ [Gauss] (Helicalc - Opera SOL)', rf'$\Delta B_{i}$ [Gauss] (Helicalc - Opera BR20)']
    else:
        c_labels = [rf'$\Delta B$ [Gauss] (Opera SOL - Opera BR20)', rf'$\Delta B$ [Gauss] (Helicalc - Opera SOL)', rf'$\Delta B$ [Gauss] (Helicalc - Opera BR20)']
    titles = ['Opera "Solenoid" vs. Opera "Brick", Y==0 [m]', 'Helicalc vs. Opera "Solenoid", Y==0 [m]', 'Helicalc vs. Opera "Brick", Y==0 [m]']

    for c, cl, t in zip(cs, c_labels, titles):
        make_pcolormesh(df=df, c=c, c_label=cl, title=t)
'''

# cs = ['dB_SOL_BR', 'dB_HEL_SOL', 'dB_HEL_BR']
# c_labels = [r'$\Delta B$ [Gauss] (Opera SOL - Opera BR20)$', r'$\Delta B$ [Gauss] (Helicalc - Opera SOL)', r'$\Delta B$ [Gauss] (Helicalc - Opera BR20)']
# titles = ['Opera "Solenoid" vs. Opera "Brick", Y==0 [m]', 'Helicalc vs. Opera "Solenoid", Y==0 [m]', 'Helicalc vs. Opera "Brick", Y==0 [m]']

# for c, cl, t in zip(cs, c_labels, titles):
#     make_pcolormesh(df=df, c=c, c_label=cl, title=t)

"""
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
"""
