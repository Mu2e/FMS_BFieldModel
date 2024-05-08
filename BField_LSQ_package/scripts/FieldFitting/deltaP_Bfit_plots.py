import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mu2e import mu2e_ext_path
from emtracks.plotting import config_plots

config_plots()

def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# a few functions to keep code DRY
def get_data(df, var, query=None):
    if query is None:
        df_ = df
    else:
        df_ = df.query(query)
    data = df_[var].values
    return data

# reference label
# label_temp = '{0}\n' + r'$\mu = {1:.3E}$'+ '\n' + 'std' + r'$= {2:.3E}$' + '\n' +  'Integral: {3}\n' + 'Underflow: {4}\nOverflow: {5}'

def get_label(name, data, bins):
    if type(bins) == int:
        over=0
        under=0
        # sci
        # mean = f'{np.mean(data):.3E}'
        # std = f'{np.std(data, ddof=1):.3E}'
        # label = f'mean: {mean:>15}' + '\n' + f'stddev: {std:>15}' + '\n' + f'Integral: {len(data):>17}\nUnderflow: {under:>16}\nOverflow: {over:>16}'
        # float
        # mean = f'{np.mean(data):.1f}'
        # std = f'{np.std(data, ddof=1):.1f}'
        mean = f'{np.mean(data):.2f}'
        std = f'{np.std(data, ddof=1):.2f}'
        label = f'mean: {mean:>9}' + '\n' + f'stddev: {std:>8}' + '\n' + f'Integral: {len(data):>5}\nUnder: {under:>11}\nOver: {over:>13}'
    else:
        over = (data > np.max(bins)).sum()
        under = (data < np.min(bins)).sum()
        data = data[(data <= np.max(bins)) & (data >= np.min(bins))]
        # sci
        # mean = f'{np.mean(data):.3E}'
        # std = f'{np.std(data, ddof=1):.3E}'
        # label = f'mean: {mean:>15}' + '\n' + f'stddev: {std:>15}' + '\n' + f'Integral: {len(data)-over-under:>17}\nUnderflow: {under:>16}\nOverflow: {over:>16}'
        # float
        # mean = f'{np.mean(data):.1f}'
        # std = f'{np.std(data, ddof=1):.1f}'
        mean = f'{np.mean(data):.2f}'
        std = f'{np.std(data, ddof=1):.2f}'
        label = f'mean: {mean:>9}' + '\n' + f'stddev: {std:>8}' + '\n' + f'Integral: {len(data)-over-under:>5}\nUnder: {under:>11}\nOver: {over:>13}'
    # label = f'{name}\n' + rf'$\mu: {np.mean(tand):.3E}$' + '\n' + rf'$\sigma: {np.std(tand):.3E}$' + '\n' + f'Integral: {len(tand)}\nUnderflow: {under}\nOverflow: {over}'
    # std = f'{np.std(data):.3E}'
    # n = 15
    return label

def make_plot_hist(df, name='Mau9 70%', var='tand_Mau9_70', xl=r'$\tan(\mathrm{dip})$', query=None, queryn='full', bins=20, legendloc='upper right', plotdir=None, Bname="", fig=None, ax=None, save=True):
    data = get_data(df, var, query)
    if ax is None:
        fig, ax = plt.subplots()
    sname = re.search('_(.*)_', Bname).group(1).replace('_','\ ')
    # sname = Bname[:-9] # re.search('_(.*)_', Bname).group(1)
    # sname = ""# Bname[:-9] # re.search('_(.*)_', Bname).group(1)
    ax.hist(data, bins=bins, histtype='step', linewidth=1.5, label=rf'$\bf{{{sname}}}$'+'\n'+get_label(None, data, bins))
    # ax.hist(data, bins=bins, histtype='step', linewidth=1.5, label=get_label(None, data, bins))
    ax.set_ylabel('Events')
    ax.set_xlabel(xl)
    if query is None:
        plt.title(Bname)
    else:
        plt.title(Bname+'\n'+queryn)
    plt.legend(loc=legendloc, fontsize=14)
    # plt.xlim([-1.,1.]) # MeV
    # plt.xlim([-1000.,1000.]) # keV
    if save:
        h = ''
        # h = '_huber'
        fig.savefig(plotdir+Bname+h+'.pdf')
        fig.savefig(plotdir+Bname+h+'.png')
    return fig, ax

if __name__=='__main__':
    # check plot output directory and create if doesn't exist
    #######
    #B_dir = 'cartesian_test' # done
    # B_dir = 'default' # done
    B_dir = 'k2_test' # done
    # B_dir = 'delta_Z_tests' # done
    # B_dir = 'hp_bias' # done
    # B_dir = 'ensemble_random_scale_factor' # done
    #######
    pdir = mu2e_ext_path + 'pickles/Bfit_CE_reco/' + B_dir + '/'
    # plotdir = mu2e_ext_path + 'plots/html/deltaP/' + B_dir + '/'
    plotdir = mu2e_ext_path + 'plots/html/deltaP/' + B_dir + '/single/'
    check_dir(plotdir)

    # get files in pickle directory
    files = sorted(os.listdir(pdir))
    # files_run = [i for i in files if i != "Mau13_sparserz_huber_deltaP.p"]
    # files_run = [i for i in files if i != "Mau13_sparserz_deltaP.p"]
    files_run = files
    # print(files_run)

    # xl = r'$p_{\mathrm{MC}} -p_{\mathrm{fit}}$ [keV]'
    xl = 'Momentum Shift [keV]'

    fig = None; ax = None
    for f in files_run:
        fig = None; ax = None
        df = pd.read_pickle(pdir+f)
        # make_plot_hist(df, var='deltaP', xl=xl, query=None, queryn='', bins=20, legendloc='best', plotdir=plotdir, Bname=f[:-2])
        # UNITS keV
        df.eval('deltaP = deltaP * 1000', inplace=True)
        # fig, ax = make_plot_hist(df, var='deltaP', xl=xl, query=None, queryn='', bins=np.linspace(-1000,1000,201), legendloc='upper left', plotdir=plotdir, Bname=f[:-2], fig=fig, ax=ax)
        # fig, ax = make_plot_hist(df, var='deltaP', xl=xl, query=None, queryn='', bins=np.linspace(-50,50,51), legendloc='upper left', plotdir=plotdir, Bname=f[:-2], fig=fig, ax=ax)
        fig, ax = make_plot_hist(df, var='deltaP', xl=xl, query=None, queryn='', bins=20, legendloc='upper left', plotdir=plotdir, Bname=f[:-2], fig=fig, ax=ax, save=True)
        # ALTERATION FOR MSC SEMINAR
        ax.plot([0., 0.], [0., 150.], 'r--')
        ax.set_xlim([-100, 100])
        ax.set_ylim([0, 145])
        fig.savefig(f'/home/ckampa/data/plots/html/deltaP/MSC_June2023/{B_dir}_{f}_wide.pdf')
        fig.savefig(f'/home/ckampa/data/plots/html/deltaP/MSC_June2023/{B_dir}_{f}_wide.png')
        # plt.show()
