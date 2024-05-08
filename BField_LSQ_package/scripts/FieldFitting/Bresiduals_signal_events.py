import sys
import os
import time
from string import Template
from collections import namedtuple
import pandas as pd
import plotly.express as px
import pickle as pkl
from scipy.interpolate import griddata
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt

# This package (script in this directory)
from hallprobesim_redux import *
from mu2e import mu2e_ext_path

# emtracks package
# import pickle directly to avoid package install? FIX ME!
from emtracks.particle import trajectory_solver
from emtracks.plotting import config_plots

# start timing
start = time.time()

# run plot configs
config_plots()
plt.rcParams['figure.figsize'] = [13, 11] # larger figures

# globals
# FITNAME = "fma_mau13"
RUN = "run_04"
E_ANALYSIS = False
EMTRACK_RUN_DIR = f"/home/ckampa/data/pickles/distortions/linear_gradient/{RUN}/"
ESTRIDE = 50

# load validation dataframe
df = pd.read_pickle(mu2e_ext_path+'Bmaps/Mau13_validation_df.p')

# evaluate residuals on input data
df.eval('Bz_res = Bz - Bz_fit', inplace=True)
df.eval('Br_res = Br - Br_fit', inplace=True)
df.eval('Bphi_res = Bphi - Bphi_fit', inplace=True)
df.eval('B = (Br**2 + Bphi**2 + Bz**2)**(1/2)', inplace=True)
df.eval('B_fit = (Br_fit**2 + Bphi_fit**2 + Bz_fit**2)**(1/2)', inplace=True)
df.eval('B_res = B - B_fit', inplace=True)

def closest_residual(e_file):
    e = trajectory_solver.from_pickle(EMTRACK_RUN_DIR+e_file)
    df_e = e.dataframe.copy()[::ESTRIDE]
    df_e.reset_index(drop=True, inplace=True)
    for res in ['B_res', 'Br_res', 'Bphi_res', 'Bz_res']:
        df_e.loc[:, res] = griddata(df[['X','Y','Z']].values, df[res], df_e[["x","y","z"]].values, method='nearest')
    return df_e.values

# this takes awhile (~20 min for 1000 tracks, 100 locations per track)...most time spent in griddata part. maybe switch to mapinterp. FIX ME!
if E_ANALYSIS:
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")

    print("Starting e track analysis")
    # get run files
    e_files = [f for f in os.listdir(EMTRACK_RUN_DIR) if ('run' in f) & ('nom' in f)]
    e = trajectory_solver.from_pickle(EMTRACK_RUN_DIR+e_files[0])
    cols = list(e.dataframe.columns) + ['B_res', 'Br_res', 'Bphi_res', 'Bz_res']

    df_nps = Parallel(n_jobs=num_cpu)(delayed(closest_residual)(ef) for ef in tqdm(e_files, file=sys.stdout, desc='trajectory #'))
    df_nps = np.vstack(df_nps)
    df_results = pd.DataFrame(df_nps, columns=cols)
    df_results.to_pickle(mu2e_ext_path+'Bmaps/Mau13_etracks_residuals.p')
else:
    if 'Mau13_etracks_residuals.p' not in os.listdir(mu2e_ext_path+'Bmaps'):
        raise Exception('Please run with E_ANALYSIS = True to generate the residuals dataframe')

    df_results = pd.read_pickle(mu2e_ext_path+'Bmaps/Mau13_etracks_residuals.p')

# generate plots
alpha=0.7
ax_tup = namedtuple('ax_tup', ['row', 'column', 'title', 'df_col'])
ax_tup2 = namedtuple('ax_tup2', ['row', 'column', 'title', 'df_col'])
axtups = [ax_tup(row=0, column=0, title=r'$|B|$', df_col='B_res'),
          ax_tup(row=0, column=1, title=r'$B_r$', df_col='Br_res'),
          ax_tup(row=1, column=0, title=r'$B_z$', df_col='Bz_res'),
          ax_tup(row=1, column=1, title=r'$B_\phi$', df_col='Bphi_res'),
        ]
axtups = [ax_tup(row=0, column=0, title='', df_col='B_res'),
          ax_tup(row=0, column=1, title=r'$B_r$', df_col='Br_res'),
          ax_tup(row=1, column=0, title=r'$B_z$', df_col='Bz_res'),
          ax_tup(row=1, column=1, title=r'$B_\phi$', df_col='Bphi_res'),
        ]
# label_temp = Template(rf'$df \n ')
label_temp = '{0}\n' + r'$\mu = {1:.3E}$'+ '\n' + 'std' + r'$= {2:.3E}$' + '\n'
# combined
def plot_residuals_comparison(trackercut=False):
    if trackercut:
        df_ = df.query("Z >= 8.54 & Z <= 11.810")
        df_results_ = df_results.query("z >= 8.54 & z <= 11.810")
        title = "DS (Tracker Only): 'Z >= 8.540 m & Z <= 11.810 m'\n"
        f = 'tracker'
    else:
        df_ = df
        df_results_ = df_results
        title = "DS (Full): 'Z >= 4.221 m & Z <= 13.896 m'\n"
        f = 'ds'

    for LOGSCALE in [True, False]:
        fig, axs = plt.subplots(2, 2)
        for t in axtups:
            n, bins, patches = axs[t.row, t.column].hist(df_[t.df_col], bins=200, alpha=alpha, density=True,
                                                         label=label_temp.format('DS ('+r'$R \leq 0.8$'+' m):', df_[t.df_col].mean(), df_[t.df_col].std()))
            axs[t.row, t.column].hist(df_results_[t.df_col], bins=bins, density=True, alpha=alpha,
                                      label=label_temp.format('1000 e- tracks:', df_results_[t.df_col].mean(), df_results_[t.df_col].std()))
            axs[t.row, t.column].legend(prop={'size': 10})
            if LOGSCALE:
                axs[t.row, t.column].set_yscale('log')
            axs[t.row, t.column].set_ylabel('Count (Density)')
            axs[t.row, t.column].set_xlabel('Data - Fit [Gauss]')
            axs[t.row, t.column].set_title(t.title)

        fig.suptitle(title, fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if LOGSCALE:
            fname = mu2e_ext_path+f'plots/residuals_emtracks/{f}_residuals_comparison_log'
        else:
            fname = mu2e_ext_path+f'plots/residuals_emtracks/{f}_residuals_comparison'

        fig.savefig(fname+'.pdf')
        fig.savefig(fname+'.png')


def plot_large_residuals_3d(threshold=1):
    # filter large residuals
    df_ = df_results.query(f'abs(B_res) >= {threshold}')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(df_.z, df_.y, df_.x, c=df_.B_res, cmap='viridis', s=2, alpha=1., zorder=101)
    cb = fig.colorbar(p)
    cb.set_label('|B|: \nData - Fit \n[Gauss]', rotation=0.)
    ax.set_xlabel('\nZ [m]', linespacing=3.0)
    ax.set_ylabel('\nY [m]', linespacing=3.0)
    ax.set_zlabel('\nX [m]', linespacing=3.0)
    ax.view_init(elev=10., azim=-95.)
    fig.suptitle(f'|B|: Data - Fit [Gauss] >= {threshold}')
    fig.tight_layout()
    fname = mu2e_ext_path+f'plots/residuals_emtracks/3d_track_residuals_geq_{threshold}_gauss'
    fig.savefig(fname+'.pdf')
    fig.savefig(fname+'.png')
    return fig, ax

def plot_large_residuals_2d(threshold=1):
    # filter large residuals
    df_ = df_results.query(f'abs(B_res) >= {threshold}')
    df_.eval('r = (x**2+y**2)**(1/2)', inplace=True)
    cols = ['x', 'y', 'z', 'r', 'Br_res', 'Bz_res', 'Bphi_res', 'B_res']
    fig = px.scatter_matrix(df_[cols], color='B_res')
    fig.update_traces(marker=dict(size=1))
    fig.update_layout(
        title=f'|B|: Data - Fit [Gauss] >= {threshold}',
        autosize=False,
        width=1200,
        height=1000
    )
    fname = mu2e_ext_path+f'plots/residuals_emtracks/scatter_matrix_colorresiduals_geq_{threshold}_gauss'
    fig.write_image(fname+'.pdf')
    fig.write_image(fname+'.png')
    fig.write_html(fname+'.html')
    return fig

# plot_residuals_comparison(True)
# plot_residuals_comparison(False)
thresholds = [1, 2, 5]
for threshold in thresholds:
    # fig, ax = plot_large_residuals_3d(threshold)
    fig = plot_large_residuals_2d(threshold)

print(f"Completed Analysis: {time.time() - start} s")
