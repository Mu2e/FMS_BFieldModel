import re
import os
import argparse
import numpy as np
import pandas as pd
import lmfit as lm
import matplotlib.pyplot as plt

from mu2e import mu2e_ext_path
from emtracks.plotting import config_plots

config_plots()


def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def lin(x, **params):
    return params['m'] * x + params['b']

def run_fit(x, y, yerr, modelf=lin, loss='linear'):
    model = lm.Model(modelf, independent_vars=['x'])
    params = lm.Parameters()
    params.add('m', value=0, vary=True)
    params.add('b', value=0, vary=True)
    result = model.fit(y, x=x, params=params,
                       weights=1/yerr,
                       method='least_squares',
                       fit_kws={'loss': loss})
    print(result.fit_report())
    return result

def make_scatter_fit(x, y, yerr=None, y_fit=None, dlabel='Ensemble Random Biases', flabel=None):
    fig, ax = plt.subplots()

    # ax.scatter(x, y, s=5, label=dlabel)
    ax.errorbar(x, y, yerr=yerr, fmt='.', elinewidth=2, capsize=5., markersize=10, label=dlabel)

    if y_fit is not None:
        ax.plot(x, y_fit, 'r-', label=flabel)

    ax.set_xlabel('mean(Scale Factors) - 1')
    ax.set_ylabel('mean(Momentum Shift) [keV]')
    ax.legend(loc='best', fontsize=14)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    fig.tight_layout()
    # fig.savefig()
    return fig, ax


if __name__ == '__main__':
    pdir = mu2e_ext_path+'pickles/'
    plotdir = mu2e_ext_path+'plots/html/deltaP/ensemble_random_scale_factor/summary/'
    # load data
    df_init = pd.read_pickle(pdir+'Mu2E/ensemble_random_scale_factor/run000_random_sfs_init_conds_df.p')
    mean_sfs = np.mean(df_init[[f'hp_{i}_sf' for i in range(5)]].values, axis=1)
    mean_deltaPs = []
    std_deltaPs = []
    exps = np.arange(50)
    for e in exps:
        fname = pdir+f'Bfit_CE_reco/ensemble_random_scale_factor/Mau13_run000_exp0{e:02d}_deltaP.p'
        df_ = pd.read_pickle(fname)
        mean_deltaPs.append(df_.deltaP.mean())
        std_deltaPs.append(df_.deltaP.std())
    mean_deltaPs = 1000*np.array(mean_deltaPs)
    std_deltaPs = 1000*np.array(std_deltaPs)

    args = np.argsort(mean_sfs)
    x = mean_sfs[args] - 1.
    y = mean_deltaPs[args]
    yerr = std_deltaPs[args]

    # fit
    # first fit
    result = run_fit(x, y, yerr=yerr, loss='huber')
    # result = run_fit(x, y, yerr=1, loss='huber')
    y_fit = result.best_fit

    # clean out outliers
    cut = 2
    r = result.residual
    med = np.median(r)
    mask_good = (r >= med-cut) & (r <= med+cut)
    x2 = x[mask_good]
    y2 = y[mask_good]
    yerr2 = yerr[mask_good]

    result2 = run_fit(x2, y2, yerr2, loss='linear')
    y_fit2 = result2.best_fit

    m = result2.params['m'].value / 1e4
    b = result2.params['b'].value

    # single hall probe bias data
    x_b = [np.mean([1+2e-4]+4*[1]) - 1]
    df_ = pd.read_pickle(pdir+'Bfit_CE_reco/hp_bias/Mau13_hp_1_bias_up_2E-04_deltaP.p')
    y_b = [1000*df_.deltaP.mean()]
    yerr_b = [1000*df_.deltaP.std()]

    # plot
    # fig, ax = make_scatter_fit(x, y, yerr=yerr, y_fit=y_fit, flabel='fit (Huber loss)')
    fig, ax = make_scatter_fit(x, y, yerr=yerr, y_fit=None, flabel='fit (Huber loss)')
    ax.errorbar(x_b, y_b, yerr=yerr_b, fmt='x', elinewidth=3, capsize=5., markersize=9, label='Hall Probe 1: (SF-1) = 2E-4', zorder=100, c='orange')
    # ax.scatter(x_b, y_b, s=75, marker='+', c='green', label='Hall Probe 1: SF-1 = 2E-4', zorder=100)
    ax.legend(loc='best', fontsize=14)
    fig.savefig(plotdir+'mean_dP_vs_mean_SF-1.pdf')
    fig.savefig(plotdir+'mean_dP_vs_mean_SF-1.png')
    # fig, ax = make_scatter_fit(x, yerr, yerr=None, y_fit=None, flabel='fit (Huber loss)')
    flabel = rf"$y = ({m:0.2f}) (x * 10^4) + ({b:0.2f})$"
    fig, ax = make_scatter_fit(x2, y2, yerr=yerr2, y_fit=y_fit2, flabel=flabel)
    ax.errorbar(x_b, y_b, yerr=yerr_b, fmt='x', elinewidth=3, capsize=5., markersize=9, label='Hall Probe 1: (SF-1) = 2E-4', zorder=100, c='orange')
    # ax.scatter(x_b, y_b, s=75, marker='+', c='green', label='Hall Probe 1: SF-1 = 2E-4', zorder=100)
    ax.legend(loc='best', fontsize=14)
    fig.savefig(plotdir+'mean_dP_vs_mean_SF-1_fit.pdf')
    fig.savefig(plotdir+'mean_dP_vs_mean_SF-1_fit.png')
    # fig, ax = make_scatter_fit(x, y, yerr=None, y_fit=y_fit, flabel='fit (Huber loss)')
    # fig, ax = make_scatter_fit(x, y, yerr, y_fit, flabel='fit')

    plt.show()

