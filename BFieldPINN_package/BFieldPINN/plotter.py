import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
#from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from datetime import datetime
from mu2e.mu2eplots import mu2e_plot3d_nonuniform_test

mono_font = plt.rcParams['font.monospace'][0]

### check plotdir
def check_plot_dir(model_fname):
    plotdir = os.path.join(model_fname, 'plots')
    trackdir = os.path.join(plotdir, 'tracking')
    for d in [plotdir, trackdir]:
        os.makedirs(d, exist_ok=True)
    return plotdir, trackdir

### FORMATS
# nicer plot formatting
def config_plots():
    #must run twice for some reason (glitch in Jupyter)
    for i in range(2):
        plt.rcParams['figure.figsize'] = [10, 8] # larger figures
        plt.rcParams['axes.grid'] = True         # turn grid lines on
        plt.rcParams['axes.axisbelow'] = True    # put grid below points
        plt.rcParams['grid.linestyle'] = '--'    # dashed grid
        plt.rcParams.update({'font.size': 18.0})   # increase plot font size
        #plt.rcParams.update({"text.usetex": True})
        plt.rcParams.update({"text.usetex": False})

def ticks_in(ax, top_and_right=True):
    if top_and_right:
        ax.tick_params(which='both', direction='in', right=True, top=True)
    else:
        ax.tick_params(which='both', direction='in')
    return ax

def ticks_sizes(ax, major={'L':20,'W':2}, minor={'L':10,'W':1}):
    ax.tick_params('both', length=major['L'], width=major['W'], which='major')
    ax.tick_params('both', length=minor['L'], width=minor['W'], which='minor')
    return ax

# label for histogram
def get_label(data, bins, data_max=1e7):
    over = str((data > np.max(bins)).sum())
    over = f'{over:>12}'
    under = str((data < np.min(bins)).sum())
    under = f'{under:>12}'
    data_ = data[(data <= np.max(bins)) & (data >= np.min(bins))]
    N = len(data_)
    if N > data_max:
        N = f'{N:0.3E}'
    else:
        N = f'{N}'
    N = f'{N:>12}'
    # ORIGINAL
    # mean = f'{np.mean(data_):.3E}'
    # std = f'{np.std(data_, ddof=1):.3E}'
    # label = f'mean: {mean:>15}\nstddev: {std:>15}\nIntegral: {len(data):>17}\n'\
    # +f'Underflow: {under:>16}\nOverflow: {over:>16}'
    # fixed
    mean = f'{np.mean(data_):.3E}'
    mean = f'{mean:>12}'
    std = f'{np.std(data_, ddof=1):.3E}'
    std = f'{std:>12}'
    # strings left shifted
    label = f'{"mean:":<11}{mean}\n'
    label += f'{"stddev:":<11}{std}\n'
    label += f'{"Integral:":<11}{N}\n'
    label += f'{"Underflow:":<11}{under}\n'
    label += f'{"Overflow:":<11}{over}'
    return label

### 1. Value vs. Epoch plots
def create_dict_from_history(history):
    keys_dict = {
        'Loss': {'nice_name': 'Train (total)', 'linestyle': '-', 'linecolor': 'red', 'linewidth': 2},
        'Loss_curl': {'nice_name': 'Train (curl B)', 'linestyle': '--', 'linecolor': 'green', 'linewidth': 1},
        'Loss_div': {'nice_name': 'Train (div B)', 'linestyle': '--', 'linecolor': 'orange', 'linewidth': 1},
        'Loss_B': {'nice_name': 'Train (B)', 'linestyle': '--', 'linecolor': 'blue', 'linewidth': 1},
        'Loss_val': {'nice_name': 'Validation (B)', 'linestyle': '-', 'linecolor': 'black', 'linewidth': 2},
        ## in a separate plot
        'lr': {'nice_name': 'Learning Rate', 'linestyle': '-', 'linecolor': 'blue', 'linewidth': 2},
        'lambda_': {'nice_name': r'$\lambda$', 'linestyle': '-', 'linecolor': 'blue', 'linewidth': 2},
        'epsilon_': {'nice_name': r'$\epsilon$', 'linestyle': '-', 'linecolor': 'blue', 'linewidth': 2},
    }
    eps_vals = np.array(history['epsilon_'])
    keys_dict['epsilon_']['vals'] = eps_vals
    keys_dict['lambda_']['vals'] = np.array(history['lambda_'])
    keys_dict['lr']['vals'] = np.array(history['lr'])
    # epsilon only applied at "Loss" level. All others are correct
    keys_dict['Loss']['vals'] = np.array(history['Loss']) / eps_vals
    keys_dict['Loss_curl']['vals'] = np.array(history['Loss_curl'])
    keys_dict['Loss_div']['vals'] = np.array(history['Loss_div'])
    keys_dict['Loss_B']['vals'] = np.array(history['Loss_B'])
    # val
    keys_dict['Loss_val']['vals'] = np.array(history['Loss_val'])
    parsed_history = keys_dict

    return parsed_history

def make_plots_history_vs_epoch(parsed_history, plotdir=None, model_num='1'):
    plot_dict = {
        'Loss_vs_Epoch': {'cols': ['Loss', 'Loss_curl', 'Loss_div', 'Loss_B', 'Loss_val'],
                          'ylims': [1e-3, None], 'ylabel': 'Loss',
                          'title': 'Loss vs. Epoch', 'log': True},
        'LR_vs_Epoch': {'cols': ['lr'], 'ylims': [None, None], 'ylabel': 'LR',
                        'title': 'Learning Rate vs. Epoch', 'log': True},
        'lambda_vs_Epoch': {'cols': ['lambda_'], 'ylims': [None, None],
                            'ylabel': parsed_history['lambda_']['nice_name'],
                            'title': r'$\lambda$ vs. Epoch', 'log': False},
        'epsilon_vs_Epoch': {'cols': ['epsilon_'], 'ylims': [None, None],
                             'ylabel': parsed_history['epsilon_']['nice_name'],
                             'title': r'$\epsilon$ vs. Epoch', 'log': False},
    }

    epochs = np.arange(1, len(parsed_history['Loss']['vals'])+1)

    for plotname, plotconfig in plot_dict.items():
        if len(plotconfig['cols']) > 1:
            figsize = (14, 8)
            #w_pad = 2./12.
            #h_pad = 0.
        else:
            figsize = (12, 8)
            #w_pad = 0.
            #h_pad = 2./12.
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        #fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad,
        #    hspace=0., wspace=0.)
        ax = ticks_in(ax, top_and_right=True)
        ax = ticks_sizes(ax, major={'L':10,'W':1}, minor={'L':5,'W':0.75})
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        for col in plotconfig['cols']:
            ph = parsed_history[col]
            vals = ph['vals']
            if len(plotconfig['cols']) > 1:
                label = ph['nice_name']
                legend = True
            else:
                label = None
                legend = False
            ax.plot(epochs, vals, color=ph['linecolor'], linestyle=ph['linestyle'],
                    linewidth=ph['linewidth'], label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(plotconfig['ylabel'])
        ax.set_title(plotconfig['title'])
        ax.set_ylim([plotconfig['ylims'][0], plotconfig['ylims'][1]])
        if legend:
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        # savefig
        if not plotdir is None:
            fname = os.path.join(plotdir, model_num+'_'+plotname)
            fig.savefig(fname+'.pdf', bbox_inches='tight', pad_inches=0.25)
            fig.savefig(fname+'.png', bbox_inches='tight', pad_inches=0.25)
        # log version
        if plotconfig['log']:
            ax.set_yscale('log')
            if not plotdir is None:
                fname = os.path.join(plotdir, model_num+'_'+'log'+plotname)
                fig.savefig(fname+'.pdf', bbox_inches='tight', pad_inches=0.25)
                fig.savefig(fname+'.png', bbox_inches='tight', pad_inches=0.25)
        plot_dict[plotname]['fig'] = fig
        plot_dict[plotname]['ax'] = ax
    return plot_dict

def make_input_data_profile(df_meas, df_test, q_str='(X == -0.8) & (Y == 0.0)',
                            noise=None, plotdir=None, model_num='1'):
    df_m = df_meas.query(q_str)
    df_t = df_test.query(q_str)
    fig_dict = {}
    for i in ['x', 'y', 'z', 'r', 'phi']:
        if i == 'phi':
            yl = r'\phi'
        else:
            yl = i
        figsize = (14, 6)
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax = ticks_in(ax, top_and_right=True)
        ax = ticks_sizes(ax, major={'L':10,'W':1}, minor={'L':5,'W':0.75})
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        if not noise is None:
            ax.errorbar(df_m.Z, df_m[f'dB{i}'], yerr=noise, c='blue', fmt='o', ls='none', ms=3, capsize=2, label='Training Dataset', zorder=99)
        else:
            ax.scatter(df_m.Z, df_m[f'dB{i}'], s=3, c='blue', label='Training Dataset',
                       zorder=99)
        ax.scatter(df_t.Z, df_t[f'dB{i}'], s=1, c='red', label='Test Dataset',
                   zorder=98)
        ax.set_label('Z [m]')
        ax.set_ylabel(rf'$\Delta B_{{ {yl}, \mathrm{{expansion}} }}$ [Gauss]')
        ax.set_title(f'Input Data: {q_str}')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        # savefig
        if not plotdir is None:
            plotname = f'input_data_dB{i}_vs_Z'
            fname = os.path.join(plotdir, model_num+'_'+plotname)
            fig.savefig(fname+'.pdf', bbox_inches='tight', pad_inches=0.25)
            fig.savefig(fname+'.png', bbox_inches='tight', pad_inches=0.25)
        fig_dict[i] = {'fig': fig, 'ax': ax}
    return fig_dict

def make_Bi_residual_1D_hist(df, bin_orig=True, nbins=200, title_suff=' (Test Dataset)',
                             add_noise_model=False, noise=None, noise_on_final=True,
                             fname_suff='_df_test', plotdir=None, model_num='1'):
    fig_dict = {}
    for i in ['x', 'y', 'z', 'r', 'phi']:
        if i == 'phi':
            yl = r'\phi'
        else:
            yl = i
        #figsize = (12, 6)
        figsize = (16, 8)
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax = ticks_in(ax, top_and_right=True)
        ax = ticks_sizes(ax, major={'L':10,'W':1}, minor={'L':5,'W':0.75})
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # calculate the residuals
        # residual = B - model = B - (B_fit + dB_NN)
        res1 = (df[f'B{i}'] - (df[f'B{i}_fit'])).values
        res2 = (df[f'B{i}'] - (df[f'B{i}_fit'] + df[f'dB{i}_NN'])).values
        res3 = (df[f'B{i}'] - (df[f'B{i}_fit_full'])).values
        if bin_orig:
            vals = res1
        else:
            vals = res2
        # set up bins
        yma = np.max(np.absolute(vals))
        yra = (np.max(vals) - np.min(vals))
        bins = np.linspace(-yma - 0.05*yra, yma + 0.05*yra, nbins)

        n, _, _ = ax.hist(res1, bins=bins, histtype='step', linewidth=1.75,
                          color='black', alpha=1.0,
                          label='LSQ Fit\n'+get_label(res1, bins), zorder=9)
        n_NN, _, _ = ax.hist(res2, bins=bins, histtype='bar', linewidth=0.0,
                             color='blue', edgecolor='black', alpha=0.7,
                             label='PINN Training\n'+get_label(res2, bins), zorder=10)
        n_refit, _, _ = ax.hist(res3, bins=bins, histtype='step', linewidth=1.75,
                             color='limegreen', hatch='/', alpha=1.0,
                             label='LSQ Refit\n'+get_label(res3, bins), zorder=11)
        #_, _, _ = ax.hist(res2, bins=bins, histtype='step', linewidth=2.0,
        #                     color='darkblue', zorder=11)

        # add noise model?
        if add_noise_model:
            if noise_on_final:
                n_ = n_refit
            else:
                n_ = n
            integral = n_.sum() * (bins[1]-bins[0])
            xs = np.linspace(-yma, yma, 500)
            ys = integral * scipy.stats.norm.pdf(xs, loc=0.0, scale=noise)
            y_max = max(n_.max(), ys.max()) * 2
            ax.plot(xs, ys, '--', color='red', linewidth=1.5, alpha=1.0, label='Injected Noise Model', zorder=12)
            ax.set_ylim([0.5, y_max])

        ax.set_yscale('log')

        ax.set_xlabel(rf'$\Delta B_{{ {yl} }}$ [Gauss] (Data - Fit)')
        ax.set_title(rf'Fit Residuals: $\Delta B_{{ {yl} }}${title_suff}')

        L = ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.setp(L.texts, family=mono_font)
        # savefig
        if not plotdir is None:
            plotname = f'residual_dB{i}_1D_hist{fname_suff}'
            fname = os.path.join(plotdir, model_num+'_'+plotname)
            fig.savefig(fname+'.pdf', bbox_inches='tight', pad_inches=0.25)
            fig.savefig(fname+'.png', bbox_inches='tight', pad_inches=0.25)
        fig_dict[i] = {'fig': fig, 'ax': ax}
    return fig_dict

def make_deriv_1D_hist(df, bin_numerical=True, nbins=200, title_suff=' (Test Dataset)',
                       include_numerical=True, include_exact=True,
                       fname_suff='_df_test', plotdir=None, model_num='1'):
    fig_dict = {}
    cols = ['divB', 'curlB', 'curlB_x', 'curlB_y', 'curlB_z']
    col_names = [r'$\nabla \cdot \vec{B}$', r'$\nabla \times \vec{B}$', r'$(\nabla \times \vec{B})_x$',
             r'$(\nabla \times \vec{B})_y$', r'$(\nabla \times \vec{B})_z$']
    col_suffs = []
    label_pres = []
    colors = ['black', 'blue']
    hatches = ['/', None]
    fname_type = ''
    if include_numerical:
        col_suffs.append('')
        label_pres.append('Numerical\n')
        fname_type += '_with_numerical'
    if include_exact:
        col_suffs.append('_exact')
        label_pres.append('Exact (autodiff.)\n')
        fname_type += '_with_exact'
    for col, name in zip(cols, col_names):
        #figsize = (12, 6)
        figsize = (16, 8)
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        ax = ticks_in(ax, top_and_right=True)
        ax = ticks_sizes(ax, major={'L':10,'W':1}, minor={'L':5,'W':0.75})
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # grab the appropriate values
        vals_list = []
        bvals = None
        for cs in col_suffs:
            vals = df[f'{col}{cs}'].values
            vals_list.append(vals)
            if bin_numerical:
                if cs == '':
                    bvals = vals
            else:
                if cs == '_exact':
                    bvals = vals
        if bvals is None:
            raise RuntimeError(f"bin_numerical={bin_numerical} is not consistent with include_numerical={include_numerical} and include_exact={include_excat}!")
        yma = np.max(np.absolute(bvals))
        yra = (np.max(bvals) - np.min(bvals))
        bins = np.linspace(-yma - 0.05*yra, yma + 0.05*yra, nbins)

        for i, tup in enumerate(zip(vals_list, col_suffs, label_pres, colors, hatches)):
            vals, cs, lp, c, h = tup
            _, _, _ = ax.hist(vals, bins=bins, histtype='step', linewidth=1.75,
                              color=c, alpha=1.0, hatch=h,
                              label=lp+get_label(vals, bins), zorder=9+i)

        ax.set_yscale('log')

        ax.set_xlabel(name+' [Gauss/m]')
        ax.set_title('PINN '+name+f'{title_suff}')

        L = ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        plt.setp(L.texts, family=mono_font)
        # savefig
        if not plotdir is None:
            plotname = f'derivs_{col}_1D_hist{fname_type}{fname_suff}'
            fname = os.path.join(plotdir, model_num+'_'+plotname)
            fig.savefig(fname+'.pdf', bbox_inches='tight', pad_inches=0.25)
            fig.savefig(fname+'.png', bbox_inches='tight', pad_inches=0.25)
        fig_dict[col] = {'fig': fig, 'ax': ax}
    return fig_dict

def make_fit_data_profile(df_meas, df_test, x='Z',
                          q_str='(X == -0.8) & (Y == 0.0)', title_suff='',
                          fname_suff='_X_m0p8_Y_0p0',
                          noise=None, plotdir=None, model_num='1'):
    df_m = df_meas.query(q_str)
    df_t = df_test.query(q_str)
    # check whether "fit point" is in the columns
    if 'fit_point' in df_m.columns:
        mask_fit = df_m.fit_point
        mask_no_fit = (~mask_fit)
        N_no_fit = np.sum(mask_no_fit)
    else:
        mask_fit = np.ones(len(df_m)).astype(bool)
        mask_no_fit = np.zeros(len(df_m)).astype(bool)
        N_no_fit = 0
    fig_dict = {}
    for i in ['x', 'y', 'z', 'r', 'phi']:
        if i == 'phi':
            yl = r'\phi'
        else:
            yl = i
        figsize = (14, 10)
        #fig, ax = plt.subplots(figsize=figsize, layout='constrained')
        fig, axs = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1], sharex=True)
        if x == 'Z':
            ml = 0.5
        else:
            ml = 0.05
        for ax in axs:
            ax = ticks_in(ax, top_and_right=True)
            ax = ticks_sizes(ax, major={'L':10,'W':1}, minor={'L':5,'W':0.75})
            ax.xaxis.set_minor_locator(MultipleLocator(ml))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax, ax2 = axs
        ms = 5
        x_m = df_m[x].values
        vals_m = df_m[f'dB{i}'].values
        fit_m = df_m[f'dB{i}_NN'].values
        x_t = df_t[x].values
        fit_t = df_t[f'dB{i}_NN'].values
        # append final data for clarity
        if x_m.max() > x_t.max():
            x_t = np.concatenate([x_t, [x_m[-1]]])
            fit_t = np.concatenate([fit_t, [fit_m[-1]]])
        if not noise is None:
            ax.errorbar(x_m[mask_fit], vals_m[mask_fit], yerr=noise, c='black', fmt='o',
                        ls='none', ms=ms, capsize=2,
                        label='Data', zorder=99)
            if N_no_fit > 0:
                ax.errorbar(x_m[mask_no_fit], vals_m[mask_no_fit], yerr=noise, c='red', fmt='o',
                        ls='none', ms=ms, capsize=2, alpha=0.8,
                        label='Excluded Data', zorder=99)
        else:
            ax.scatter(x_m[mask_fit], vals_m[mask_fit], s=ms**2, c='black',
                       label='Data', zorder=99)
            if N_no_fit > 0:
                ax.scatter(x_m[mask_no_fit], vals_m[mask_no_fit], s=ms**2, c='red',
                       label='Excluded Data', zorder=99, alpha=0.8)
        ax.plot(x_t, fit_t, 'g--', linewidth=2,
                label='Fit (PINN)', zorder=100)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
        # dB
        dB = (df_m[f'dB{i}'] - df_m[f'dB{i}_NN']).values
        yma = np.max(np.absolute(dB))
        if not noise is None:
            yma = yma + noise
        yra = (np.max(dB) - np.min(dB))
        ylim = max(1.8, yma+0.2*yra)
        ax2.set_ylim([-ylim, ylim])
        if not noise is None:
            ax2.errorbar(x_m[mask_fit], dB[mask_fit], yerr=noise, c='black', fmt='o',
                        ls='none', ms=ms, capsize=2,
                        label='Data', zorder=99)
            if N_no_fit > 0:
                ax2.errorbar(x_m[mask_no_fit], dB[mask_no_fit], yerr=noise, c='red', fmt='o',
                        ls='none', ms=ms, capsize=2, alpha=0.8,
                        label='Data', zorder=99)
        else:
            ax2.scatter(x_m[mask_fit], dB[mask_fit], s=ms**2, c='black',
                       label='Data', zorder=99)
            if N_no_fit > 0:
                ax2.scatter(x_m[mask_no_fit], dB[mask_no_fit], s=ms**2, c='red',
                       label='Data', zorder=99, alpha=0.8)

        ax2.set_xlabel(f'{x} [m]')
        ax.set_ylabel(rf'$\Delta B_{{ {yl}, \mathrm{{expansion}} }}$ [Gauss]')
        ax2.set_ylabel('Data - Fit [Gauss]')
        q_clean = q_str.replace(')', ' m)')
        ax.set_title(f'PINN Results: {q_clean}{title_suff}')
        plt.subplots_adjust(hspace=0.01)
        # savefig
        if not plotdir is None:
            plotname = f'NN_dB{i}_vs_{x}{fname_suff}'
            fname = os.path.join(plotdir, model_num+'_'+plotname)
            fig.savefig(fname+'.pdf', bbox_inches='tight', pad_inches=0.25)
            fig.savefig(fname+'.png', bbox_inches='tight', pad_inches=0.25)
        fig_dict[i] = {'fig': fig, 'axs': axs}
    return fig_dict

def make_mu2e_plot3d(df_meas, steps=[0.0, np.pi/2], steps_nice=[r'0', r'\pi/2'],
                     conditions=('Z > 4.200', 'Z < 13.900'),
                     coord_to_data=None, coord_to_fit=None,
                     fname_suff='_PINNfit', title_nice=True,
                     title_suff='\nPINN Model',
                     plotdir=None, model_num='1'):
    fig_dict = {}
    geom = 'cyl'
    ABC_geom = {'cyl': [['R', 'Z', 'Bz'], ['R', 'Z', 'Br'], ['R', 'Z', 'Bphi']],}
    df_ = df_meas.copy()
    if coord_to_data is None:
        coord_to_data = lambda i: f'dB{i}'
    if coord_to_fit is None:
        coord_to_fit = lambda i: f'dB{i}_NN'
    for i in ['r', 'phi', 'z']:
        data_i = coord_to_data(i)
        fit_i = coord_to_fit(i)
        df_.eval(f'B{i} = {data_i}', inplace=True)
        df_.eval(f'B{i}_fit = {fit_i}', inplace=True)
    cols_save = ['X', 'Y', 'Z', 'HP', 'R', 'Phi', 'Br', 'Br_fit', 'Bphi', 'Bphi_fit', 'Bz', 'Bz_fit', 'fit_point']
    if not 'fit_point' in df_.columns:
        df_.loc[:,'fit_point'] = True
    df_ = df_[cols_save].copy()
    df_.sort_values(by=['Z', 'R', 'Phi'], inplace=True)
    for step, step_nice in zip(steps, steps_nice):
        step_str = f'{step:0.2f}'
        for ABC in ABC_geom[geom]:
            c_list = []
            for c in conditions:
                if 'Phi' in c:
                    continue
                c_clean = c.replace('(', '').replace(')', '').replace(' ', '')
                c_clean = c_clean.replace('<', '==').replace('>', '==')
                var = c_clean.split('==')[0]
                val = c_clean.split('==')[-1]
                c_list.append(f'{var}{val}')
            # by hand add phi
            c_list.append(f'Phi{step:0.2f}')
            save_name = f'{model_num}_{ABC[2]}_{ABC[0]}{ABC[1]}_'+'_'.join(c_list)
            save_name += fname_suff
            # title?
            if title_nice:
                i = ABC[2][1:]
                if i == 'phi':
                    l = r'\phi'
                else:
                    l = i
                x_ = ABC[0].lower()
                y_ = ABC[1].lower()
                title_simp = rf'$B_{l}$ vs. ${x_}, {y_}$ ($\phi = {step_nice}$)'
                title_simp += title_suff
            else:
                title_simp = None
            conditions_str = ' and '.join(conditions+('Phi=={}'.format(step),))
            _ = mu2e_plot3d_nonuniform_test(df_, ABC[0], ABC[1], ABC[2],
                                            conditions=conditions_str,
                                            df_fit=True, mode='mpl_nonuni',
                                            save_dir=plotdir, save_name=save_name,
                                            do_title=True, title_simp=title_simp,
                                            do2pi=False, units='m', df_fine=None,
                                            show_plot=True, legend=True)
            fig, axs = _
            fig_dict[step_str] = {'fig': fig, 'axs': axs}
    return fig_dict


