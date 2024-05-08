import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from mu2e import mu2e_ext_path
from emtracks.plotting import config_plots
config_plots()
plt.rcParams['figure.figsize'] = [13, 11] # larger figures
plt.rcParams.update({'font.size': 18.0})   # increase plot font size
# plt.rc('text', usetex=True)


def load_dfs(mapname_base):
    # load dfs
    # full DS
    df_full = pd.read_pickle(mapname_base+'.p')
    # calculate residuals
    df_full.eval('B = (Bx**2 + By**2 + Bz**2)**(1/2)', inplace=True)
    df_full.eval('B_fit = (Bx_fit**2 + By_fit**2 + Bz_fit**2)**(1/2)', inplace=True)
    # df_nom.eval('B = (Bx**2 + By**2 + Bz**2)**(1/2)', inplace=True)
    df_full.loc[:, 'B_res'] = df_full['B'] - df_full['B_fit']
    df_full.loc[:, 'Bx_res'] = df_full['Bx'] - df_full['Bx_fit']
    df_full.loc[:, 'By_res'] = df_full['By'] - df_full['By_fit']
    df_full.loc[:, 'Bz_res'] = df_full['Bz'] - df_full['Bz_fit']
    # query for FMS measured areas
    df_full_fms = df_full.query('Z >= 4.25  & Z <= 14. & R <= 0.8').copy()
    # tracker fit
    # new: query full df
    df_fit = df_full.query('R <= 0.8 & Z >= 8.41 & Z <= 11.66').copy()
    # tracks, tracker
    df_run = pd.read_pickle(mapname_base+'_track_res.p')
    # tracks, tracker, straws
    df_run_straws = pd.read_pickle(mapname_base+'_track_res_Rcut.p')

    df_list = [df_full, df_full_fms, df_fit, df_run, df_run_straws]
    return df_list

def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def make_plot(df, plot_base, title_base, file_suffix='_tracker', title_suffix='Tracker Region', log=False):
    label_temp = r'$\mu = {0:.3E}$'+ '\n' + 'std' + r'$= {1:.3E}$' + '\n' +  'Integral: {2}'
    if log:
        log_str = '_log'
        yscale = 'log'
    else:
        log_str = ''
        yscale = 'linear'
    # print("Generating plots:"+file_suffix)
    # simple histograms
    plt.rcParams.update({'font.size': 18.0})   # increase plot font size
    plt.rcParams['axes.grid'] = True         # turn grid lines on
    plt.rcParams['axes.axisbelow'] = True    # put grid below points
    plt.rcParams['grid.linestyle'] = '--'    # dashed grid
    N_bins = 200
    lsize = 16
    xmin = df[['Bx_res','By_res','Bz_res','B_res']].min().min()
    xmax = df[['Bx_res','By_res','Bz_res','B_res']].max().max()+1e-5
    bins = np.linspace(xmin, xmax, N_bins+1)
    fig, axs = plt.subplots(2, 2, figsize=(13, 11))
    axs[0, 0].hist(df['Bx_res'], bins=bins, label=label_temp.format(df['Bx_res'].mean(), df['Bx_res'].std(), len(df)))
    axs[0, 0].set(xlabel=r"$\Delta B_x$"+" [Gauss]", yscale=yscale)
    # axs[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
    axs[0, 0].legend(prop={'size': lsize})
    axs[0, 1].hist(df['By_res'], bins=bins, label=label_temp.format(df['By_res'].mean(), df['By_res'].std(), len(df)))
    axs[0, 1].set(xlabel=r"$\Delta B_y$"+" [Gauss]", yscale=yscale)
    # axs[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
    axs[0, 1].legend(prop={'size': lsize})
    axs[1, 0].hist(df['Bz_res'], bins=bins, label=label_temp.format(df['Bz_res'].mean(), df['Bz_res'].std(), len(df)))
    axs[1, 0].set(xlabel=r"$\Delta B_z$"+" [Gauss]", yscale=yscale)
    # axs[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
    axs[1, 0].legend(prop={'size': lsize})
    axs[1, 1].hist(df['B_res'], bins=bins, label=label_temp.format(df['B_res'].mean(), df['B_res'].std(), len(df)))
    axs[1, 1].set(xlabel=r"$\Delta |B|$"+" [Gauss]", yscale=yscale)
    # axs[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False)
    axs[1, 1].legend(prop={'size': lsize})
    # change y format if not log
    if not log:
        for i in range(2):
            for j in range(2):
                axs[i, j].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1e'))
    # title & save
    fig.suptitle(title_base+title_suffix)
    # fig.tight_layout(rect=[0,0,1,0.9])
    fig.tight_layout()
    plot_file = plot_base+file_suffix+log_str
    fig.savefig(plot_file+'.pdf')
    fig.savefig(plot_file+'.png')
    # print("Generating plots complete.\n")
    return fig, axs

def make_plot_loop(Bfile, Bshort, file_suffs, title_suffs, plotdir_0, mapdir):
    # print(f"Starting plotting for {Bshort}...")
    plotdir = plotdir_0 + Bshort + '/'
    plot_base = plotdir + Bshort + '_residuals_hist'
    title_base = Bshort + '\n'
    map_base = mapdir+Bfile
    # load all the dataframes and calculate necessary residuals
    df_list = load_dfs(map_base)
    # check that plotdir exists
    check_dir(plotdir)
    # make_plot(df, plot_base, title_base, file_suffix=fsuff, title_suffix=tsuff, log=log)
    # loop to create plots for all dfs
    for log in [False, True]:
        for df, fsuff, tsuff in zip(df_list, file_suffs, title_suffs):
            make_plot(df, plot_base, title_base, file_suffix=fsuff, title_suffix=tsuff, log=log)
    # print(f"Finished plotting for {Bshort}!")


if __name__=='__main__':
    # choice of folder (change by hand for now)
    ######
    B_dir = 'cartesian_test' # done
    # B_dir = 'k2_test' # done
    # B_dir = 'delta_Z_tests' # done
    # B_dir = 'hp_bias' # done
    # B_dir = 'ensemble_random_scale_factor' # done
    ######
    # set up shared plot configs for each plot
    file_suffs = ['_full', '_fms', '_tracker', '_tracker_tracks', '_tracker_tracks_straws']
    title_suffs = ['Grid in Entire DS Map File', 'Grid in FMS Mapped Region of DS (4.25 <= Z <= 14 m) (R <= 0.8 m)', 'Grid in Tracker Region', 'Signal e- Tracks in Tracker Region', 'Signal e- Tracks in Tracker Region (40 cm <= R <= 70 cm)']
    # loop through all map files in a folder
    mapdir = mu2e_ext_path+'Bmaps/'+B_dir+'/'
    plotdir_0 = mu2e_ext_path+f'plots/html/deltaB/{B_dir}/'
    # get all Bfiles
    Bfiles = list(set([i[:-2] for i in os.listdir(mapdir) if i[-8:] == 'fit_df.p']))
    Bshorts = [i[:-7] for i in Bfiles]
    # run parallel for loop
    num_cpu = multiprocessing.cpu_count()

    Parallel(n_jobs=num_cpu)(delayed(make_plot_loop)(Bfile, Bshort, file_suffs, title_suffs, plotdir_0, mapdir) for Bfile, Bshort in tqdm(zip(Bfiles, Bshorts), file=sys.stdout, desc='Bfile #', total=len(Bfiles)))
