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
        mean = f'{np.mean(data):.1f}'
        std = f'{np.std(data, ddof=1):.1f}'
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
        mean = f'{np.mean(data):.1f}'
        std = f'{np.std(data, ddof=1):.1f}'
        label = f'mean: {mean:>9}' + '\n' + f'stddev: {std:>8}' + '\n' + f'Integral: {len(data)-over-under:>5}\nUnder: {under:>11}\nOver: {over:>13}'
    # label = f'{name}\n' + rf'$\mu: {np.mean(tand):.3E}$' + '\n' + rf'$\sigma: {np.std(tand):.3E}$' + '\n' + f'Integral: {len(tand)}\nUnderflow: {under}\nOverflow: {over}'
    # std = f'{np.std(data):.3E}'
    # n = 15
    return label

def make_plot_hist(df, name='Mau9 70%', var='tand_Mau9_70', xl=r'$\tan(\mathrm{dip})$', query=None, queryn='full', bins=20, legendloc='upper right', plotdir=None, Bname="", fig=None, ax=None, save=True):
    data = get_data(df, var, query)
    if ax is None:
        fig, ax = plt.subplots()
    sname = re.search('_(.*)_', Bname).group(1)
    # sname = Bname[:-9] # re.search('_(.*)_', Bname).group(1)
    # sname = ""# Bname[:-9] # re.search('_(.*)_', Bname).group(1)
    # ax.hist(data, bins=bins, histtype='step', linewidth=1.5, label=rf'$\bf{{{sname}}}$'+'\n'+get_label(None, data, bins))
    ax.hist(data, bins=bins, histtype='step', linewidth=1.5, label=get_label(None, data, bins))
    ax.set_ylabel('Events')
    ax.set_xlabel(xl)
    if query is None:
        plt.title(Bname)
    else:
        plt.title(Bname+'\n'+queryn)
    plt.legend(loc=legendloc)
    # plt.xlim([-1.,1.]) # MeV
    # plt.xlim([-1000.,1000.]) # keV
    if save:
        fig.savefig(plotdir+Bname+'.pdf')
        fig.savefig(plotdir+Bname+'.png')
    return fig, ax

def make_plot_scatter(df, x='hall_probe', y='p_res_mean', yerr='p_res_std', xl='Hall Probe Location (Increases with Radius)', yl=r'$p_{\mathrm{MC}} - p_{\mathrm{fit}}$ [keV]', legendloc='upper left', fig=None, ax=None, label=None):
    data_x = get_data(df, x, None)
    data_y= get_data(df, y, None)
    data_yerr = get_data(df, yerr, None)
    if ax is None:
        fig, ax = plt.subplots()
    if label is None:
        label= "point - mean\nerrorbar - stddev.\n("+r"$\chi^2$ Loss)"
    else:
        label= "point - mean\nerrorbar - stddev.\n(Huber Loss)"

    ax.errorbar(data_x, data_y, yerr=data_yerr, fmt='.', elinewidth=2, capsize=5., markersize=10, label=label)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    labels = ['0', '1', '2', '3', '4', 'Nominal']
    plt.xticks(ticks=[0,1,2,3,4,5], labels=labels)
    # ax.set_xticklabels(['0', '1', '2', '3', '4', 'Nominal'])
    # fig.canvas.draw()
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[-1] = 'Nominal'
    # ax.set_xticklabels(labels)

    ax.set_title('Momentum Reconstruction Error vs. Single Biased Hall Probe\nScale Factor = 1 + 2e-4')
    ax.legend(loc=legendloc, fontsize=14)
    fig.savefig(plotdir+'perr_vs_single_biased_2E-4.png')
    fig.savefig(plotdir+'perr_vs_single_biased_2E-4.pdf')

    return fig, ax


if __name__=='__main__':
    # check plot output directory and create if doesn't exist
    #######
    # B_dir = 'delta_Z_tests' # done
    B_dir = 'hp_bias' # done
    # B_dir = 'ensemble_random_scale_factor' # done
    #######
    pdir = mu2e_ext_path + 'pickles/Bfit_CE_reco/' + B_dir + '/'
    plotdir = mu2e_ext_path + 'plots/html/deltaP/' + B_dir + '/'
    check_dir(plotdir)

    # get files in pickle directory
    files = sorted(os.listdir(pdir))
    files_run = []
    files_huber = []
    for f in files:
        if "2E-04" in f:
            if "huber" in f:
                files_huber.append(f)
            else:
                files_run.append(f)
    # files_run = [i for i in files if "2E-04" in i]
    offset = 0.1
    # chi2
    xs = []
    means = []
    stds = []
    for f in files_run:
        x_ = int(re.search('hp_(.*)_bias', f).group(1))
        xs.append(x_-offset)
        # xs.append(int(re.search('hp_(.*)_bias', f).group(1)))
        df = pd.read_pickle(pdir+f)
        df.eval('deltaP = deltaP * 1000', inplace=True)
        means.append(df['deltaP'].mean())
        stds.append(df['deltaP'].std())
    # add nominal
    xs.append(4.9)
    df = pd.read_pickle('/home/ckampa/data/pickles/Bfit_CE_reco/delta_Z_tests/Mau13_standard_deltaP.p')
    xs = np.array(xs)
    df.eval('deltaP = deltaP * 1000', inplace=True)
    means.append(df['deltaP'].mean())
    stds.append(df['deltaP'].std())
    means = np.array(means)
    stds = np.array(stds)
    df_stats = pd.DataFrame({'x':xs, 'mean':means, 'std':stds})

    # huber
    # xs_huber = np.array([0])
    xs_huber = []
    means_huber = []
    stds_huber = []
    # df = pd.read_pickle(pdir+files_huber[0])
    # df.eval('deltaP = deltaP * 1000', inplace=True)
    # means_huber.append(df['deltaP'].mean())
    # stds_huber.append(df['deltaP'].std())
    for f in files_huber:
        x_ = int(re.search('hp_(.*)_bias', f).group(1))
        xs_huber.append(x_+offset)
        df = pd.read_pickle(pdir+f)
        df.eval('deltaP = deltaP * 1000', inplace=True)
        means_huber.append(df['deltaP'].mean())
        stds_huber.append(df['deltaP'].std())
    # add nominal
    xs_huber.append(5.1)
    df = pd.read_pickle('/home/ckampa/data/pickles/Bfit_CE_reco/delta_Z_tests/Mau13_standard_huber_deltaP.p')
    df.eval('deltaP = deltaP * 1000', inplace=True)
    means_huber.append(df['deltaP'].mean())
    stds_huber.append(df['deltaP'].std())
    xs_huber = np.array(xs_huber)
    means_huber = np.array(means_huber)
    stds_huber = np.array(stds_huber)
    df_huber = pd.DataFrame({'x':xs_huber, 'mean':means_huber, 'std':stds_huber})
    # add huber df
    # xs_huber = np.array([0])
    # means_huber = []
    # stds_huber = []
    # df = pd.read_pickle(pdir+files_huber[0])
    # df.eval('deltaP = deltaP * 1000', inplace=True)
    # means_huber.append(df['deltaP'].mean())
    # stds_huber.append(df['deltaP'].std())
    # means_huber = np.array(means_huber)
    # stds_huber = np.array(stds_huber)
    # df_huber = pd.DataFrame({'x':xs_huber, 'mean':means_huber, 'std':stds_huber})

    # yl = r'$p_{\mathrm{MC}} - p_{\mathrm{fit}}$ [keV]'
    yl = "Momentum Shift [keV]"

    # plot!
    fig, ax = make_plot_scatter(df_stats, x='x', y='mean', yerr='std', xl='Hall Probe Location (Increases with Radius)', yl=yl, legendloc='best', fig=None, ax=None, label=None)
    # add huber
    fig, ax = make_plot_scatter(df_huber, x='x', y='mean', yerr='std', xl='Hall Probe Location (Increases with Radius)', yl=yl, legendloc='best', fig=fig, ax=ax, label=True)

    ######
    #fig = None; ax = None
    #for f in files_run:
    #    fig = None; ax = None
    #    df = pd.read_pickle(pdir+f)
    #    # make_plot_hist(df, var='deltaP', xl=r'$p_{\mathrm{MC}} - p_{\mathrm{fit}}$ [MeV]', query=None, queryn='', bins=20, legendloc='best', plotdir=plotdir, Bname=f[:-2])
    #    # UNITS keV
    #    df.eval('deltaP = deltaP * 1000', inplace=True)
    #    # fig, ax = make_plot_hist(df, var='deltaP', xl=r'$p_{\mathrm{MC}} - p_{\mathrm{fit}}$ [keV]', query=None, queryn='', bins=np.linspace(-1000,1000,201), legendloc='upper left', plotdir=plotdir, Bname=f[:-2], fig=fig, ax=ax)
    #    fig, ax = make_plot_hist(df, var='deltaP', xl=r'$p_{\mathrm{MC}} - p_{\mathrm{fit}}$ [keV]', query=None, queryn='', bins=20, legendloc='upper left', plotdir=plotdir, Bname=f[:-2], fig=fig, ax=ax, save=True)
    #    # plt.show()
