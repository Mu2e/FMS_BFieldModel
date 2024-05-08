#! /usr/bin/env python
"""Module for generating Mu2e plots.

This module defines a host of useful functions for generating plots for various Mu2E datasets.  It
is expected that these datasets be in the form of :class:`pandas.DataFrame` objects, and the
independent and dependent variables are strings that correspond to DF columns.  These functions wrap
calls to either :mod:`matplotlib` or :mod:`plotly`.  The latter has numerous modes, depending on how
the user wants to share the plots.


Example:
    Plotting in a `jupyter notebook`:

    .. code-block:: python

        In [1]: import os
        ...     from mu2e import mu2e_ext_path
        ...     from mu2e.dataframeprod import DataFrameMaker
        ...     from mu2e.mu2eplots import mu2e_plot, mu2e_plot3d
        ...     %matplotlib inline # for embedded mpl plots
        ...     plt.rcParams['figure.figsize'] = (12,8) # make them bigger

        In [2]: df = DataFrameMaker(
        ...         mu2e_ext_path + 'path/to/Mu2e_DSMap',
        ...         use_pickle = True
        ...     ).data_frame

        In [3]: mu2e_plot(df, 'Z', 'Bz', 'Y==0 and X==0', mode='mpl')
        ...     #static image of 2D matplotlib plot

        In [4]: mu2e_plot3d(
        ...         df, 'R', 'Z', 'Bz',
        ...         'R<=800 and Phi==0 and 12700<Z<12900',
        ...         mode='plotly_nb')
        ...     #interactive 3D plot in plotly for jupyter nb



Todo:
    * Allow for **kwargs inheriting from mpl, map them for plotly as well.
    * Build in more flexibility for particle trapping plots.
    * Particle animation GraphWidget will not work with updated jupyter, keep an eye on dev.


*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import pandas as pd
import plotly.tools as tls
import plotly.graph_objs as go
from mpldatacursor import datacursor
import ipywidgets as widgets
from IPython.display import display
# from plotly.widgets import GraphWidget
# from chart_studio.widgets import GraphWidget
from plotly.offline import init_notebook_mode, iplot, plot
from six.moves import range
from six.moves import zip
from mu2e import mu2e_ext_path
# face color issue
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'


def mu2e_plot(df, x, y, conditions=None, mode='mpl', info=None, savename=None, ax=None,
              auto_open=True):
    """Generate 2D plots, x vs y.

    Generate a 2D plot for a given DF and two columns. An optional selection string is applied to
    the data via the :func:`pandas.DataFrame.query` interface, and is added to the title.  This
    function supports matplotlib and various plotly plotting modes.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x' and 'y' args.
        x (str): Name of the independent variable.
        y (str): Name of the dependent variable.
        conditions (str, optional): A string adhering to the :mod:`numexpr` syntax used in
            :func:`pandas.DataFrame.query`.
        mode (str, optional): A string indicating which plotting package and method should be used.
            Default is 'mpl'. Valid values: ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
        info (str, optional): Extra information to add to the legend.
        savename (str, optional): If not `None`, the plot will be saved to the indicated path and
            file name.
        ax (matplotlib.axis, optional): Use existing mpl axis object.
        auto_open (bool, optional): If `True`, automatically open a plotly html file.

    Returns:
        axis object if 'mpl', else `None`.
    """

    _modes = ['mpl', 'plotly', 'plotly_html', 'plotly_nb']

    if mode not in _modes:
        raise ValueError(mode+' not one of: '+', '.join(_modes))

    if conditions:
        df, conditions_title = conditions_parser(df, conditions)

    leg_label = y+' '+info if info else y
    ax = df.plot(x, y, ax=ax, kind='line', label=leg_label, legend=True, linewidth=2)
    ax.grid(True)
    plt.ylabel(y)
    plt.title(' '.join([x for x in [x, 'v', y, conditions] if x]))

    if 'plotly' in mode:
        fig = ax.get_figure()
        py_fig = tls.mpl_to_plotly(fig)
        py_fig['layout']['showlegend'] = True

        if mode == 'plotly_nb':
            # init_notebook_mode()
            iplot(py_fig)
        elif mode == 'plotly_html':
            if savename:
                plot(py_fig, filename=savename, auto_open=auto_open)
            else:
                plot(py_fig, auto_open=auto_open)

    elif savename:
        plt.savefig(savename)
    if mode == 'mpl':
        return ax
    else:
        return None


def mu2e_plot3d(df, x, y, z, conditions=None, mode='mpl', info=None, save_dir=None, save_name=None,
                df_fit=None, ptype='3d', aspect='square', cmin=None, cmax=None, fig=None, ax=None,
                do_title=True, title_simp=None, do2pi=False, units='mm',show_plot=True,df_fine=None):
    """Generate 3D plots, x and y vs z.

    Generate a 3D surface plot for a given DF and three columns. An optional selection string is
    applied to the data via the :func:`pandas.DataFrame.query` interface, and is added to the title.
    Extra processing is done to plot cylindrical coordinates if that is detected from `conditions`.
    This function supports matplotlib and various plotly plotting modes.  If `df_fit` is specified,
    `df` is plotted as a a scatter plot, and `df_fit` as a wireframe plot.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x', 'y', and 'z' args.
        x (str): Name of the first independent variable.
        y (str): Name of the second independent variable.
        z (str): Name of the dependent variable.
        conditions (str, optional): A string adhering to the :mod:`numexpr` syntax used in
            :func:`pandas.DataFrame.query`.
        mode (str, optional): A string indicating which plotting package and method should be used.
            Default is 'mpl'. Valid values: ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
        info (str, optional): Extra information to add to the title.
        save_dir (str, optional): If not `None`, the plot will be saved to the indicated path. The
            file name is automated, based on the input args.
        save_name (str, optional): If `None`, the plot name will be generated based on the 'x', 'y',
            'z', and 'condition' args.
        df_fit (bool, optional): If the input df contains columns with the suffix '_fit', plot a
            scatter plot using the normal columns, and overlay a wireframe plot using the '_fit'
            columns.  Also generate a heatmap showing the residual difference between the two plots.
        ptype (str, optional): Choose between '3d' and 'heat'.  Default is '3d'.

    Returns:
        Name of saved image/plot or None.
    """

    # _modes = ['mpl', 'mpl_none', 'plotly', 'plotly_html', 'plotly_nb']
    _modes = ['mpl', 'mpl_none', 'plotly', 'plotly_html', 'plotly_nb', 'plotly_html_img']

    if mode not in _modes:
        raise ValueError(mode+' not one of: '+', '.join(_modes))

    if conditions:
        df, conditions_title = conditions_parser(df, conditions, do2pi)
        if df_fine is not None:
            df_fine, bla = conditions_parser(df_fine, conditions, do2pi)


    # Format the coordinates
    piv = df.pivot(x, y, z)
    X = piv.index.values
    Y = piv.columns.values
    Z = np.transpose(piv.values)
    Xi, Yi = np.meshgrid(X, Y)
    if df_fit:
        piv_fit = df.pivot(x, y, z+'_fit')
        Z_fit = np.transpose(piv_fit.values)
        if df_fine is None:
            df_fine = df
        df_fine = df_fine.eval(f'{z}_diff={z}-{z}_fit')
        piv_fine = df_fine.pivot(x, y, z+'_diff')
        data_fit_diff = np.transpose(piv_fine.values)
        X_fine = piv_fine.index.values
        Y_fine = piv_fine.columns.values
        Xa = np.concatenate(([X_fine[0]], 0.5*(X_fine[1:]+X_fine[:-1]), [X_fine[-1]]))
        Ya = np.concatenate(([Y_fine[0]], 0.5*(Y_fine[1:]+Y_fine[:-1]), [Y_fine[-1]]))

    # Prep save area
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_name is None:
            save_name = '{0}_{1}{2}_{3}'.format(
                z, x, y, '_'.join([i for i in conditions_title.split(', ') if i != 'and']))
            save_name = re.sub(r'[<>=!\s]', '', save_name)

            if df_fit:
                save_name += '_fit'

    # Start plotting
    if 'mpl' in mode:
        if not ax:
            if ptype.lower() == '3d' and not df_fit:
                # fig = plt.figure().gca(projection='3d')
                fig = plt.figure()
            elif ptype.lower() == 'heat':
                fig = plt.figure()
            else:
                fig = plt.figure(figsize=plt.figaspect(0.4), constrained_layout=True)
                fig.set_constrained_layout_pads(hspace=0., wspace=0.15)
                # fig.set_constrained_layout_pads(hpad=0.5, wpad=0.5, hspace=0., wspace=0.15)

        if df_fit:
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot(Xi.ravel(), Yi.ravel(), Z.ravel(), 'ko', markersize=2, zorder=100)
            ax.plot_wireframe(Xi, Yi, Z_fit, color='green', zorder=99)
        elif ptype.lower() == '3d':
            if not ax:
                ax = fig.gca(projection='3d')
            ax.plot_surface(Xi, Yi, Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'),
                            linewidth=0, antialiased=False)
        elif ptype.lower() == 'heat':
            if ax:
                pcm = ax.pcolormesh(Xi, Yi, Z, cmap=plt.get_cmap('viridis'))
            else:
                plt.pcolormesh(Xi, Yi, Z, cmap=plt.get_cmap('viridis'))
        else:
            raise KeyError(ptype+' is an invalid type!, must be "heat" or "3D"')

        plt.xlabel(f'{x} ({units})', fontsize=18)
        plt.ylabel(f'{y} ({units})', fontsize=18)
        if ptype.lower() == '3d':
            ax.set_zlabel(z+' (G)', fontsize=18)
            ax.ticklabel_format(style='sci', axis='z')
            ax.zaxis.labelpad = 20
            ax.zaxis.set_tick_params(direction='out', pad=10)
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 20
        elif do_title:
            if ax:
                cb = plt.colorbar(pcm)
            else:
                cb = plt.colorbar()
            cb.set_label(z+' (G)', fontsize=18)
        if do_title:
            if title_simp:
                plt.title(title_simp)
            elif info is not None:
                plt.title(f'{info} {z} vs {x} and {y} for DS\n{conditions_title}',
                          fontsize=20)
            else:
                plt.title('{0} vs {1} and {2} for DS\n{3}'.format(z, x, y, conditions_title),
                          fontsize=20)
        if ptype.lower() == '3d':
            # ax.view_init(elev=35., azim=30)
            ax.view_init(elev=30., azim=30)
        if save_dir:
            plt.savefig(save_dir+'/'+save_name+'.png')

        if df_fit:
            ax2 = fig.add_subplot(1, 2, 2)
            max_val = np.max(data_fit_diff)
            min_val = np.min(data_fit_diff)
            abs_max_val = max(abs(max_val), abs(min_val))
            # if (abs_max_val) > 2:
            #     heat = ax2.pcolormesh(Xa, Ya, data_fit_diff, vmin=-1, vmax=1,
            #                           cmap=plt.get_cmap('viridis'))
            #     cb = plt.colorbar(heat, aspect=7)
            #     cb_ticks = cb.ax.get_yticklabels()
            #     cb_ticks[0] = '< -2'
            #     cb_ticks[-1] = '> 2'
            #     cb_ticks = cb.ax.set_yticklabels(cb_ticks)
            # else:
            heat = ax2.pcolormesh(Xa, Ya, data_fit_diff, cmap=plt.get_cmap('viridis'),
                                  vmin=-abs_max_val, vmax=abs_max_val)
            cb = plt.colorbar(heat, aspect=20)
            plt.title('Residual, Data-Fit', fontsize=20)
            cb.set_label('Data-Fit (G)', fontsize=18)
            ax2.set_xlabel(f'{x} ({units})', fontsize=18)
            ax2.set_ylabel(f'{y} ({units})', fontsize=18)
            # datacursor(heat, hover=True, bbox=dict(alpha=1, fc='w'))
            ax.dist = 11 # default 10
            if save_dir:
                # fig.tight_layout()
                plt.savefig(save_dir+'/'+save_name+'_heat.pdf')

    elif 'plotly' in mode:
        if z == 'Bz':
            z_fancy = '$B_z$'
        elif z == 'Br':
            z_fancy = '$B_r$'
        elif z == 'Bphi':
            z_fancy = r'$B_{\theta}$'
        if aspect == 'square':
            ar = dict(x=1, y=1, z=1)
            width = 800
            height = 650
        elif aspect == 'rect':
            ar = dict(x=1, y=4, z=1)
        elif aspect == 'rect2':
            ar = dict(x=1, y=2, z=1)
            width = 900
            height = 750
        axis_title_size = 25
        axis_tick_size = 16
        if title_simp:
            title = title_simp
        elif info is not None:
            title = '{0} {1} vs {2} and {3} for DS<br>{4}'.format(info, z, x, y, conditions_title)
        else:
            title = '{0} vs {1} and {2} for DS<br>{3}'.format(z, x, y, conditions_title)

        if ptype == 'heat':
            layout = go.Layout(
                title=title,
                # ticksuffix is a workaround to add a bit of padding
                width=height,
                height=width,
                autosize=False,
                xaxis=dict(
                    title=f'{x} ({units})',
                    titlefont=dict(size=axis_title_size, family='Arial Black'),
                    tickfont=dict(size=axis_tick_size),
                ),
                yaxis=dict(
                    title=f'{y} ({units})',
                    titlefont=dict(size=axis_title_size, family='Arial Black'),
                    tickfont=dict(size=axis_tick_size),
                ),
            )

        elif ptype == '3d':
            layout = go.Layout(
                title=title,
                titlefont=dict(size=30),
                autosize=False,
                width=width,
                height=height,
                scene=dict(
                    xaxis=dict(
                        title=f'{x} ({units})',
                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                        tickfont=dict(size=axis_tick_size),
                        # dtick=400,
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                    ),
                    yaxis=dict(
                        title=f'{y} ({units})',
                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                        tickfont=dict(size=axis_tick_size),
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                    ),
                    zaxis=dict(
                        # title='B (G)',
                        title='',
                        titlefont=dict(size=axis_title_size, family='Arial Black'),
                        tickfont=dict(size=axis_tick_size),
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                        showticklabels=False,
                        showaxeslabels=False,
                    ),
                    aspectratio=ar,
                    aspectmode='manual',
                    camera=dict(
                        center=dict(x=0,
                                    y=0,
                                    z=-0.3),
                        eye=dict(x=3.4496546255787175/1.6,
                                 y=2.4876029142395506/1.6,
                                 z=1.5875472335683052/1.6)
                        # eye=dict(x=2.4496546255787175/1.6,
                        #          y=2.4876029142395506/1.6,
                        #          z=2.5875472335683052/1.6)
                    ),

                ),
                showlegend=True,
                legend=dict(x=0.8, y=0.9, font=dict(size=18, family='Overpass')),
            )
        if df_fit:
            scat = go.Scatter3d(
                x=Xi.ravel(), y=Yi.ravel(), z=Z.ravel(), mode='markers',
                marker=dict(size=3, color='rgb(0, 0, 0)',
                            line=dict(color='rgb(0, 0, 0)'), opacity=1,
                            # colorbar=go.ColorBar(title='Tesla',
                            #                      titleside='right',
                            #                      titlefont=dict(size=20),
                            #                      tickfont=dict(size=18),
                            #                      ),
                            ),
                name='Data',
            )
            lines = [scat]
            line_marker = dict(color='green', width=2)
            do_leg = True
            for i, j, k in zip(Xi, Yi, Z_fit):
                if do_leg:
                    lines.append(
                        go.Scatter3d(x=i, y=j, z=k, mode='lines',
                                     line=line_marker, name='Fit',
                                     legendgroup='fitgroup')
                    )
                else:
                    lines.append(
                        go.Scatter3d(x=i, y=j, z=k, mode='lines',
                                     line=line_marker, name='Fit',
                                     legendgroup='fitgroup', showlegend=False)
                    )
                do_leg = False

            # For hovertext
            # z_offset = (np.min(Z)-abs(np.min(Z)*0.3))*np.ones(Z.shape)
            # textz = [['x: '+'{:0.5f}'.format(Xi[i][j])+'<br>y: '+'{:0.5f}'.format(Yi[i][j]) +
            #           '<br>z: ' + '{:0.5f}'.format(data_fit_diff[i][j]) for j in
            #           range(data_fit_diff.shape[1])] for i in range(data_fit_diff.shape[0])]

            # For heatmap
            # projection in the z-direction
            # def proj_z(x, y, z):
            #     return z
            # colorsurfz = proj_z(Xi, Yi, data_fit_diff)
            # tracez = go.Surface(
            #     z=z_offset, x=Xi, y=Yi, colorscale='Viridis', zmin=-2, zmax=2, name='residual',
            #     showlegend=True, showscale=True, surfacecolor=colorsurfz, text=textz,
            #     hoverinfo='text',
            #     colorbar=dict(
            #         title='Data-Fit (G)',
            #         titlefont=dict(size=18),
            #         tickfont=dict(size=20),
            #         xanchor='center'),
            # )
            # lines.append(tracez)

        else:
            if ptype == '3d':
                surface = go.Surface(
                    x=Xi, y=Yi, z=Z,
                    colorbar=go.ColorBar(title='Gauss',
                                         titleside='right',
                                         titlefont=dict(size=25),
                                         tickfont=dict(size=18),
                                         ),
                    colorscale='Viridis')
                lines = [surface]
            elif ptype == 'heat':
                heat = go.Heatmap(x=X, y=Y, z=Z,
                                  colorbar=go.ColorBar(
                                      title='{0} (G)'.format(z),
                                      titleside='top',
                                      titlefont=dict(size=18),
                                      tickfont=dict(size=20),
                                  ),
                                  colorscale='Viridis', showscale=True, zmin=cmin, zmax=cmax)
                lines = [heat]

        fig = go.Figure(data=lines, layout=layout)

        # Generate Plot
        if show_plot:
            if mode == 'plotly_nb':
                init_notebook_mode()
                iplot(fig)
                return fig
            elif mode == 'plotly_html_img':
                if save_dir:
                    plot(fig, filename=save_dir+'/'+save_name+'.html', image='jpeg',
                         image_filename=save_name)
                else:
                    plot(fig)
            elif mode == 'plotly_html':
                if save_dir:
                    plot(fig, filename=save_dir+'/'+save_name+'.html', auto_open=False)
                else:
                    plot(fig)

    # return save_name
    return fig

'''
Method to plot DSFM-like field map
For now, assuming idealized case where there are no missing measurements
TODO fix this -- build flexibility to handle missing measurements
'''

def mu2e_plot3d_nonuniform_test(df, x, y, z, conditions=None, mode='mpl', info=None, save_dir=None, save_name=None,
                                df_fit=None, ptype='3d', aspect='square', cmin=None, cmax=None, fig=None, ax=None,
                                do_title=True, title_simp=None, do2pi=False, units='mm',show_plot=True,df_fine=None):

    _modes = ['mpl_nonuni']

    if mode not in _modes:
        raise ValueError(mode+' not one of: '+', '.join(_modes))

    if conditions:
        df, conditions_title = conditions_parser(df, conditions, do2pi)
        if df_fine is not None:
            df_fine, bla = conditions_parser(df_fine, conditions, do2pi)

    X = df[x]
    Y = df[y]
    Z = df[z]

    zmax_SP = np.max(df[df['HP'].str.contains('SP')].Z)
    zmin_BP = np.min(df[df['HP'].str.contains('BP')].Z)
    
    # Process dataframe -- add NaNs and interpolate
    # Some dataframes (ex. PINN) will drop R=0 point
    if df[df.HP=='SP1'].shape[0] != 0 :
        r_SP = [-0.095, -0.054, -0.0, 0.0, 0.054, 0.095]
    else:
        r_SP = [-0.095, -0.054, 0.054, 0.095]
    r_BP = [-0.8, -0.656, -0.488, -0.319, -0.044, 0.044, 0.319, 0.488, 0.656, 0.8]
    nSP = len(r_SP)
    nBP = len(r_BP)
    
    #Z slices with SP points only
    dfs_SP = []
    df_SP = df[df.Z < zmin_BP-0.025]
    for iSP in range(int(df_SP.shape[0]/nSP)):
        df_SP_slice = df_SP[iSP*nSP:(iSP+1)*nSP]
        df_BP = pd.DataFrame(columns=df.columns,index=range(nBP))
        df_BP.loc[:,'R'] = r_BP
        df_slice = pd.concat([df_SP_slice,df_BP])
        df_slice.sort_values(by=['R'],inplace=True)
        dfs_SP.append(df_slice.set_index('R').interpolate(method='index',axis=0,limit_area='inside').reset_index())
        dfs_SP.append(df_slice)
    
    # Z slices with BP points only
    dfs_BP = []
    df_BP = df[df.Z > zmax_SP+0.025]
    for iBP in range(int(df_BP.shape[0]/nBP)):
        df_BP_slice = df_BP[iBP*nBP:(iBP+1)*nBP]
        df_SP = pd.DataFrame(columns=df.columns,index=range(nSP))
        df_SP.loc[:,'R'] = r_SP
        df_slice = pd.concat([df_BP_slice,df_SP])
        df_slice.sort_values(by=['R'],inplace=True)
        dfs_BP.append(df_slice)
        dfs_BP.append(df_slice.set_index('R').interpolate(method='index',axis=0,limit_area='inside').reset_index())

    #Z slices with both points
    df_both = df[(df.Z > zmin_BP-0.025) & (df.Z < zmax_SP+0.025)]
    slices = int(df_both.shape[0]/(nSP+nBP))
    for idx in range(slices):
        df_both.iloc[idx*(nSP+nBP):(idx+1)*(nSP+nBP)] = df_both[idx*(nSP+nBP):(idx+1)*(nSP+nBP)].sort_values(by=['R'])

    df_wire = pd.concat(dfs_SP+[df_both]+dfs_BP)
    
    Xi    = np.array(df_wire[x       ]).reshape(int(df_wire.shape[0]/(nSP+nBP)),(nSP+nBP))
    Yi    = np.array(df_wire[y       ]).reshape(int(df_wire.shape[0]/(nSP+nBP)),(nSP+nBP))
    Z_fit = np.array(df_wire[z+'_fit']).reshape(int(df_wire.shape[0]/(nSP+nBP)),(nSP+nBP))

    # If no fine-grained map provided, use fit map for heatmap
    # However since this is a 'histogram', set SP Z values equal to closest BP Z
    if df_fine is None:
        df_fine = df
        df_fine.loc[np.array(df_fine['HP'].str.contains('SP')),'Z'] -= 0.015
        df_fine = df_fine.round({'Z':3})

    df_fine = df_fine.eval(f'{z}_diff={z}-{z}_fit')
    piv_fine = df_fine.pivot_table(z+'_diff',x,y)
    X_fine = piv_fine.index.values
    Y_fine = piv_fine.columns.values
    dZ = np.transpose(piv_fine.values)
    Xa = np.concatenate(([X_fine[0]], 0.5*(X_fine[1:]+X_fine[:-1]), [X_fine[-1]]))
    Ya = np.concatenate(([Y_fine[0]], 0.5*(Y_fine[1:]+Y_fine[:-1]), [Y_fine[-1]]))
    # TODO turn this on/off with a flag in cfg_plot
    #errZ = np.transpose(df_fine.pivot_table(f'{z}_unc',x,y).values)
    #relZ = np.abs(np.divide(dZ,errZ))
    
    # Prep save area
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_name is None:
            save_name = '{0}_{1}{2}_{3}'.format(
                z, x, y, '_'.join([i for i in conditions_title.split(', ') if i != 'and']))
            save_name = re.sub(r'[<>=!\s]', '', save_name)

            if df_fit:
                save_name += '_fit'

    # Start plotting
    if 'mpl' in mode:
        if not ax:
            if ptype.lower() == '3d' and not df_fit:
                fig = plt.figure()
            elif ptype.lower() == 'heat':
                fig = plt.figure()
            else:
                fig = plt.figure(figsize=plt.figaspect(0.4), constrained_layout=True)
                fig.set_constrained_layout_pads(hspace=0., wspace=0.15)

        if df_fit:
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.plot(X, Y, Z, 'ko', markersize=2, zorder=100)
            ax.plot_wireframe(Xi, Yi, Z_fit, color='green', zorder=99)
        elif ptype.lower() == '3d':
            if not ax:
                ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'),
                            linewidth=0, antialiased=False)
        elif ptype.lower() == 'heat':
            if ax:
                pcm = ax.pcolormesh(X, Y, Z, cmap=plt.get_cmap('viridis'))
            else:
                plt.pcolormesh(X, Y, Z, cmap=plt.get_cmap('viridis'))
        else:
            raise KeyError(ptype+' is an invalid type!, must be "heat" or "3D"')

        plt.xlabel(f'{x} ({units})', fontsize=18)
        plt.ylabel(f'{y} ({units})', fontsize=18)
        if ptype.lower() == '3d':
            ax.set_zlabel(z+' (G)', fontsize=18)
            ax.ticklabel_format(style='sci', axis='z')
            ax.zaxis.labelpad = 20
            ax.zaxis.set_tick_params(direction='out', pad=10)
            ax.xaxis.labelpad = 20
            ax.yaxis.labelpad = 20
        elif do_title:
            if ax:
                cb = plt.colorbar(pcm)
            else:
                cb = plt.colorbar()
            cb.set_label(z+' (G)', fontsize=18)
        if do_title:
            if title_simp:
                plt.title(title_simp)
            elif info is not None:
                plt.title(f'{info} {z} vs {x} and {y} for DS\n{conditions_title}',
                          fontsize=20)
            else:
                plt.title('{0} vs {1} and {2} for DS\n{3}'.format(z, x, y, conditions_title),
                          fontsize=20)
        if ptype.lower() == '3d':
            ax.view_init(elev=30., azim=30)
        if save_dir and not df_fit:
            plt.savefig(save_dir+'/'+save_name+'.png')
            plt.clf()

        if df_fit:
            ax2 = fig.add_subplot(1, 2, 2)
            max_val = np.nanmax(dZ)
            min_val = np.nanmin(dZ)
            abs_max_val = max(abs(max_val), abs(min_val))
            heat = ax2.pcolormesh(Xa, Ya, dZ, cmap=plt.get_cmap('viridis'), vmin=-abs_max_val, vmax=abs_max_val)
            cb = plt.colorbar(heat, aspect=20)
            plt.title('Residual, Data-Fit', fontsize=20)
            cb.set_label('Data-Fit (G)', fontsize=18)
            ax2.set_xlabel(f'{x} ({units})', fontsize=18)
            ax2.set_ylabel(f'{y} ({units})', fontsize=18)
            ax.dist = 11 # default 10
            if save_dir:
                plt.savefig(save_dir+'/'+save_name+'_heat.pdf')
            plt.clf()

            # TODO turn this on/off with a flag in cfg_plot
            '''
            fig2 = plt.figure(figsize=plt.figaspect(0.4), constrained_layout=True)
            fig2.set_constrained_layout_pads(hspace=0., wspace=0.2)

            ax  = fig2.add_subplot(1, 2, 1)
            max_val = np.nanmax(errZ)
            heat = ax.pcolormesh(Xa, Ya, errZ, cmap=plt.get_cmap('viridis'), vmin=0.0, vmax=abs_max_val)
            cb = plt.colorbar(heat, aspect=20)
            plt.title('Fit Uncertainty', fontsize=20)
            cb.set_label('Fit Unc (G)', fontsize=18)
            ax.set_xlabel(f'{x} ({units})', fontsize=18)
            ax.set_ylabel(f'{y} ({units})', fontsize=18)

            ax2 = fig2.add_subplot(1, 2, 2)
            max_val = np.nanmax(relZ)
            heat2 = ax2.pcolormesh(Xa, Ya, relZ, cmap=plt.get_cmap('viridis'), vmin=0.0, vmax=abs_max_val)
            cb2 = plt.colorbar(heat2, aspect=20)
            plt.title('Residual / Uncertainty', fontsize=20)
            cb2.set_label('Res / Unc', fontsize=18)
            ax2.set_xlabel(f'{x} ({units})', fontsize=18)
            ax2.set_ylabel(f'{y} ({units})', fontsize=18)

            if save_dir:
                plt.savefig(save_dir+'/'+save_name+'_unc.pdf')
            plt.clf()
            '''

def mu2e_plot3d_nonuniform_cyl(df, x, y, z, conditions=None, mode='mpl', cut_color=5, info=None, save_dir=None, save_name=None,
                               df_fit=None, ptype='3d', aspect='square', cmin=None, cmax=None, fig=None, ax=None,
                               do_title=True, title_simp=None, do2pi=False, units='mm',show_plot=True,parallel=False):
    """FIXME! This docstring was copied from mu2e_plot3d and needs to be edited.
    Generate 3D plots, x and y vs z.

    Generate a 3D surface plot for a given DF and three columns. An optional selection string is
    applied to the data via the :func:`pandas.DataFrame.query` interface, and is added to the title.
    Extra processing is done to plot cylindrical coordinates if that is detected from `conditions`.
    This function supports matplotlib and various plotly plotting modes.  If `df_fit` is specified,
    `df` is plotted as a a scatter plot, and `df_fit` as a wireframe plot.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x', 'y', and 'z' args.
        x (str): Name of the first independent variable.
        y (str): Name of the second independent variable.
        z (str): Name of the dependent variable.
        conditions (str, optional): A string adhering to the :mod:`numexpr` syntax used in
            :func:`pandas.DataFrame.query`.
        mode (str, optional): A string indicating which plotting package and method should be used.
            Default is 'mpl'. Valid values: ['mpl', 'plotly', 'plotly_html', 'plotly_nb']
        info (str, optional): Extra information to add to the title.
        save_dir (str, optional): If not `None`, the plot will be saved to the indicated path. The
            file name is automated, based on the input args.
        save_name (str, optional): If `None`, the plot name will be generated based on the 'x', 'y',
            'z', and 'condition' args.
        df_fit (bool, optional): If the input df contains columns with the suffix '_fit', plot a
            scatter plot using the normal columns, and overlay a wireframe plot using the '_fit'
            columns.  Also generate a heatmap showing the residual difference between the two plots.
        ptype (str, optional): Choose between '3d' and 'heat'.  Default is '3d'.

    Returns:
        Name of saved image/plot or None.
    """

    # _modes = ['mpl', 'mpl_none', 'plotly', 'plotly_html', 'plotly_nb']
    #_modes = ['mpl', 'mpl_none', 'plotly', 'plotly_html', 'plotly_nb', 'plotly_html_img']
    #_modes = ['mpl', 'mpl_none']
    #_modes = ['mpl_nonuni', 'mpl_nonuni_none']
    _modes = ['mpl_nonuni']

    if mode not in _modes:
        raise ValueError(mode+' not one of: '+', '.join(_modes))
    # if running in parallel, need to set some rcParams
    if parallel:
        plt.rcParams.update({'font.size': 18.0})   # increase plot font size
        # figsize = (16, 16)
        figsize = (15, 15)
        # figsize = (8, 8)
        # figsize = (12, 12)
        dist = 11
    else:
        figsize = (16, 16)
        dist = 10
    # print('Before parsing:')
    # print(df)
    if conditions:
        df, conditions_title = conditions_parser_nonuni(df, conditions, do2pi)
    # df.sort_values(by=[y,x], inplace=True)
    df.sort_values(by=[x,y], inplace=True)
    # print('After parsing:')
    # print(df)
    X = df[x].values
    Y = df[y].values
    Z = df[z].values
    if df_fit:
        Z_fit = df[z+'_fit']
        data_fit_diff = (Z - Z_fit)
        map_color = np.abs(data_fit_diff) > cut_color
        m = map_color

    # Prep save area
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if save_name is None:
            save_name = '{0}_{1}{2}_{3}'.format(
                z, x, y, '_'.join([i for i in conditions_title.split(', ') if i != 'and']))
            save_name = re.sub(r'[<>=!\s]', '', save_name)

            if df_fit:
                save_name += '_fit'

    # Start plotting
    if 'mpl' in mode:
        if not ax:
            if ptype.lower() == '3d' and not df_fit:
                fig = plt.figure(figsize=(16,8))
            elif ptype.lower() == 'heat':
                #fig = plt.figure()
                raise NotImplementedError("Only '3d' ptype has been implemented.")
            else:
                gridspec_kw = dict(height_ratios=[2, 1], width_ratios=[1, 8])
                fig, axs = plt.subplots(2, 2, figsize=figsize, gridspec_kw=gridspec_kw)#, constrained_layout=True)
                axs[0, 0].remove()
                axs[0, 1].remove()
                axs[1, 0].remove()
                axs[1, 1].remove()
                gs = gridspec.GridSpec(2, 2, **gridspec_kw)

        if df_fit:
            ax = fig.add_subplot(gs[0, :], projection='3d')
            ax.scatter(X[~m], Y[~m], Z[~m], c='black', s=8, zorder=100)
            ax.scatter(X[m], Y[m], Z[m], c='red', s=40, marker='*', zorder=101)
            trs = ax.plot_trisurf(X, Y, Z_fit, cmap='viridis', edgecolor=None, zorder=99, alpha=0.4)
            cb_3d = plt.colorbar(trs, fraction=0.046, pad=0.08)
            cb_3d.set_label(label=z+' (G)', fontsize=18)
            if aspect != 'square':
                ax.set_box_aspect((np.ptp(X), 3*np.ptp(Y), 3*np.ptp(Y)))  # aspect ratio a bit closer to reality
        elif ptype.lower() == '3d':
            if not ax:
                ax = fig.gca(projection='3d')
            ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor=None, zorder=99, alpha=0.8)
        elif ptype.lower() == 'heat':
            # if ax:
            #     pcm = ax.pcolormesh(Xi, Yi, Z, cmap=plt.get_cmap('viridis'))
            # else:
            #     plt.pcolormesh(Xi, Yi, Z, cmap=plt.get_cmap('viridis'))
            raise NotImplementedError("Only '3d' ptype has been implemented.")
        else:
            raise KeyError(ptype+' is an invalid type!, must be "heat" or "3D"')

        plt.xlabel(f'{x} ({units})', fontsize=18)
        plt.ylabel(f'{y} ({units})', fontsize=18)
        if ptype.lower() == '3d':
            ax.set_zlabel(z+' (G)', fontsize=18)
            ax.ticklabel_format(style='sci', axis='z')
            ax.zaxis.labelpad = 40
            ax.zaxis.set_tick_params(which='both', direction='out', pad=20)
            ax.xaxis.set_tick_params(which='both', direction='in', pad=15)
            ax.yaxis.set_tick_params(which='both', direction='in', pad=15)
            ax.xaxis.labelpad = 30
            ax.yaxis.labelpad = 30
        # only applies for "heat", which I haven't implemented yet.
        # elif do_title:
        #     if ax:
        #         cb = plt.colorbar(pcm)
        #     else:
        #         cb = plt.colorbar()
        #     cb.set_label(z+' (G)', fontsize=18)
        if do_title:
            if title_simp:
                plt.title(title_simp)
            elif info is not None:
                plt.title(f'{info} {z} vs {x} and {y} for DS\n{conditions_title}',
                          fontsize=20)
            else:
                plt.title(f'{z} vs {x} and {y} for DS\n{conditions_title}',
                          fontsize=20)
        if ptype.lower() == '3d':
            ax.view_init(elev=50., azim=-85)
        # why do we save a png here?
        # if save_dir:
        #     plt.savefig(save_dir+'/'+save_name+'.png')

        if df_fit:
            ax2 = fig.add_subplot(gs[1, 1])
            max_val = np.max(data_fit_diff)
            min_val = np.min(data_fit_diff)
            abs_max_val = max(abs(max_val), abs(min_val))
            # create custom colormap
            GreysBig = get_cmap('Greys', 512)
            newGreys = ListedColormap(GreysBig(np.linspace(0.25, 1.0, 256)))
            # plot differently based on size of residual
            max_dev = np.max(np.abs(data_fit_diff))
            sc = ax2.scatter(X[~m], Y[~m], c=data_fit_diff[~m], s=2, cmap=newGreys)
            cb = plt.colorbar(sc, fraction=0.046, pad=0.08)
            cb.set_label(label=f'Data-Fit (G) |<={cut_color}|', fontsize=18)
            if len(X[m]) > 0:
                sc_m = ax2.scatter(X[m], Y[m], c=data_fit_diff[m], s=20, cmap='bwr')
                cb_m = plt.colorbar(sc_m, fraction=0.046, pad=0.05)
                cb_m.set_label(label=f'Data-Fit (G)', fontsize=18)
            # if scale by size, use below
            # sc_m = ax2.scatter(X[m], Y[m], c=data_fit_diff[m], s=20*np.abs(data_fit_diff[m])/max_dev, cmap='bwr')
            plt.title('Residual, Data-Fit', fontsize=20)
            ax2.set_xlabel(f'{x} ({units})', fontsize=18)
            ax2.set_ylabel(f'{y} ({units})', fontsize=18)
            ax.dist = dist # default 10
        # various remaining formatting steps
        X_l = np.min(X)
        X_h = np.max(X)
        X_r = X_h - X_l
        Y_l = np.min(Y)
        Y_h = np.max(Y)
        Y_r = Y_h - Y_l
        ax.set_xlim([X_l-0.05*X_r, X_h+0.05*X_r])
        ax.set_ylim([Y_l-0.05*Y_r, Y_h+0.05*Y_r])
        fig.tight_layout()
        if save_dir:
            # fig.tight_layout()
            #plt.savefig(save_dir+'/'+save_name+'_heat.pdf')
            plt.savefig(save_dir+'/'+save_name+'.pdf')
            plt.savefig(save_dir+'/'+save_name+'.png')

    elif 'plotly' in mode:
        raise NotImplementedError("Plotly has not been implemented for non-uniform plots. Please include 'mpl' in the 'mode' parameter.")

    return fig, ax


def mu2e_plot3d_ptrap(df, x, y, z, save_name=None, color=None, df_xray=None, x_range=None,
                      y_range=None, z_range=None, title=None, symbol='o',
                      color_min=None, color_max=None):
    """Generate 3D scatter plots, typically for visualizing 3D positions of charged particles.

    Generate a 3D scatter plot for a given DF and three columns. Due to the large number of points
    that are typically plotted, :mod:`matplotlib` is not supported.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x', 'y', and 'z' args.
        x (str): Name of the first variable.
        y (str): Name of the second variable.
        z (str): Name of the third variable.
        save_name (str, optional): If not `None`, the plot will be saved to
            `mu2e_ext_path+ptrap/save_name.html` (or `.jpeg`)
        color: (str, optional): Name of fourth varible, represented by color of marker.
        df_xray: (:class:`pandas.DataFrame`, optional): A seperate DF, representing the geometry of
            the material that is typically included during particle simulation.

    Returns:
        Name of saved image/plot, or None.

    Notes:
        Growing necessity for many input args, should implement `kwargs` in future.
    """

    init_notebook_mode()
    layout = ptrap_layout(title=title)
    scat_plots = []

    # Set the xray image if available
    if isinstance(df_xray, pd.DataFrame):
        xray_maker(df_xray, scat_plots)
    if not hasattr(df, 'name'):
        df.name = 'Particle'

    # Plot the actual content
    if isinstance(df, pd.DataFrame):
        if color:
            if not color_min:
                color_min = df[color].min()
            if not color_max:
                color_max = df[color].max()
            scat_plots.append(
                go.Scatter3d(
                    x=df[x], y=df[y], z=df[z], mode='markers',
                    marker=dict(size=4, color=df[color], colorscale='Viridis', opacity=1,
                                line=dict(color='black', width=1),
                                showscale=True, cmin=color_min, cmax=color_max,
                                symbol=symbol,
                                colorbar=dict(title='Momentum (MeV)')),
                    text=df[color].astype(int),
                    name=df.name))

        else:
            scat_plots.append(
                go.Scatter3d(
                    x=df[x], y=df[y], z=df[z], mode='markers',
                    marker=dict(size=3, color='blue', opacity=0.5),
                    name=df.name))

    fig = go.Figure(data=scat_plots, layout=layout)
    iplot(fig)

    return save_name


def mu2e_plot3d_ptrap_traj(df, x, y, z, save_name=None, df_xray=None, x_range=(3700, 17500),
                           y_range=(-1000, 1000), z_range=(-1000, 1000),
                           title=None, aspect='cosmic', color_mode='time'):
    """Generate 3D line plots, typically for visualizing 3D trajectorys of charged particles.

    Generate 3D line plot for a given DF and three columns. Due to the large number of points
    that are typically plotted, :mod:`matplotlib` is not supported.

    Args:
        df (pandas.DataFrame): The input DF, must contain columns with labels corresponding to the
            'x', 'y', and 'z' args.
        x (str): Name of the first variable.
        y (str): Name of the second variable.
        z (str): Name of the third variable.
        save_name (str, optional): If not `None`, the plot will be saved to
            `mu2e_ext_path+ptrap/save_name.html` (or `.jpeg`)
        color: (str, optional): Name of fourth varible, represented by color of marker.
        df_xray: (:class:`pandas.DataFrame`, optional): A seperate DF, representing the geometry of
            the material that is typically included during particle simulation.

    Returns:
        Name of saved image/plot, or None.

    Notes:
        Growing necessity for many input args, should implement `kwargs` in future.
    """

    init_notebook_mode()
    layout = ptrap_layout(title=title, x_range=x_range, y_range=y_range, z_range=z_range,
                          aspect=aspect)
    line_plots = []

    if isinstance(df_xray, pd.DataFrame):
        xray_maker(df_xray, line_plots)

    if isinstance(df, pd.DataFrame):
        # Define what color will represent
        if color_mode == 'time':
            c = df.time
            c_title = 'Global Time (ns)'
            c_min = df.time.min()
            c_max = df.time.max()

        elif color_mode == 'mom':
            c = df.p
            c_title = 'Momentum (MeV)'
            c_min = df.p.min()
            c_max = df.p.max()

        # Set the xray image if available

        # Plot the actual content
        try:
            name = df.name
        except AttributeError:
            name = 'Particle'

        line_plots.append(
            go.Scatter3d(
                x=df[x], y=df[y], z=df[z],
                marker=dict(size=0.1, color=c, colorscale='Viridis',
                            line=dict(color=c, width=5, colorscale='Viridis'),
                            showscale=True, cmin=c_min, cmax=c_max,
                            colorbar=dict(title=c_title)),
                line=dict(color=c, width=5, colorscale='Viridis'),
                name=name
            )
        )
    elif isinstance(df, list):
        for d in df:
            try:
                name = d.name
            except AttributeError:
                name = 'Particle'

            line_plots.append(
                go.Scatter3d(
                    x=d[x], y=d[y], z=d[z],
                    marker=dict(size=0.1,
                                line=dict(width=5)),
                    line=dict(width=5),
                    name=name
                )
            )

    fig = go.Figure(data=line_plots, layout=layout)
    iplot(fig)

    return save_name


def mu2e_plot3d_ptrap_anim(df_group1, x, y, z, df_xray, df_group2=None, color=None, title=None):
    """Generate 3D scatter plots, with a slider widget typically for visualizing 3D positions of
    charged particles.

    Generate a 3D scatter plot for a given DF and three columns. Due to the large number of points
    that are typically plotted, :mod:`matplotlib` is not supported. A slider widget is generated,
    corresponding to the evolution in time.

    Notes:
        * Currently many options are hard-coded.
        * Use `g.plot(fig)` after executing this function.
    Args:
        df_group1 (pandas.DataFrame): The input DF, must contain columns with labels corresponding
        to the 'x', 'y', and 'z' args.
        x (str): Name of the first variable.
        y (str): Name of the second variable.
        z (str): Name of the third variable.
        df_xray: (:class:`pandas.DataFrame`): A seperate DF, representing the geometry of the
            material that is typically included during particle simulation.
        df_group2: (:class:`pandas.DataFrame`, optional): A seperate DF, representing the geometry
            of the material that is typically included during particle simulation.
        color (optional): Being implemented...
        title (str, optional): Title.

    Returns:
        (g, fig) for plotting.
    """
    init_notebook_mode()
    layout = ptrap_layout(title=title)

    class time_shifter:
        def __init__(self, group2=False, color=False):
            self.x = df_group1[df_group1.sid == sids[0]][x]
            self.y = df_group1[df_group1.sid == sids[0]][y]
            self.z = df_group1[df_group1.sid == sids[0]][z]
            self.group2 = group2
            self.docolor = color
            if self.docolor:
                self.color = df_group1[df_group1.sid == sids[0]].p

            if self.group2:
                self.x2 = df_group2[df_group2.sid == sids[0]][x]
                self.y2 = df_group2[df_group2.sid == sids[0]][y]
                self.z2 = df_group2[df_group2.sid == sids[0]][z]
                if self.docolor:
                    self.color2 = df_group2[df_group2.sid == sids[0]].p

        def on_time_change(self, name, old_value, new_value):
            try:
                self.x = df_group1[df_group1.sid == sids[new_value]][x]
                self.y = df_group1[df_group1.sid == sids[new_value]][y]
                self.z = df_group1[df_group1.sid == sids[new_value]][z]
                if self.docolor:
                    self.color = df_group1[df_group1.sid == sids[new_value]].p
            except:
                self.x = []
                self.y = []
                self.z = []
                if self.docolor:
                    self.color = []

            if self.group2:
                try:
                    self.x2 = df_group2[df_group2.sid == sids[new_value]][x]
                    self.y2 = df_group2[df_group2.sid == sids[new_value]][y]
                    self.z2 = df_group2[df_group2.sid == sids[new_value]][z]
                    if self.docolor:
                        self.color2 = df_group2[df_group2.sid == sids[new_value]].p
                except:
                    self.x2 = []
                    self.y2 = []
                    self.z2 = []
                    if self.docolor:
                        self.color2 = []
            self.replot()

        def replot(self):
            if self.docolor:
                g.restyle({'x': [self.x], 'y': [self.y], 'z': [self.z],
                           'marker': dict(size=5, color=self.color,
                                          opacity=1, colorscale='Viridis', cmin=0, cmax=100,
                                          showscale=True,
                                          line=dict(color='black', width=1),
                                          colorbar=dict(title='Momentum (MeV)', xanchor='left'))
                           },
                          indices=[0])
            else:
                g.restyle({'x': [self.x], 'y': [self.y], 'z': [self.z]}, indices=[0])
            if self.group2:
                if self.docolor:
                    g.restyle({'x': [self.x2], 'y': [self.y2], 'z': [self.z2],
                               'marker': dict(size=5, color=self.color2,
                                              opacity=1, colorscale='Viridis', cmin=0, cmax=100,
                                              symbol='x',
                                              showscale=True,
                                              line=dict(color='black', width=1),
                                              colorbar=dict(title='Momentum (MeV)',
                                                            xanchor='left')),
                               },
                              indices=[1])
                else:
                    g.restyle({'x': [self.x2], 'y': [self.y2], 'z': [self.z2]}, indices=[1])
    try:
        g1_name = df_group1.name
    except:
        g1_name = 'Group 1'

    group2 = False
    if isinstance(df_group2, pd.DataFrame):
        group2 = True
        try:
            g2_name = df_group2.name
        except:
            g2_name = 'Group 1'

    if group2:
        if len(df_group1.sid.unique()) > len(df_group2.sid.unique()):
            sids = np.sort(df_group1.sid.unique())
        else:
            sids = np.sort(df_group2.sid.unique())
    else:
        sids = np.sort(df_group1.sid.unique())

    scats = []
    if color:
        mdict = dict(size=5, color=df_group1[df_group1.sid == sids[0]].p, opacity=1,
                     colorscale='Viridis', cmin=0, cmax=100,
                     showscale=True,
                     line=dict(color='black', width=1),
                     colorbar=dict(title='Momentum (MeV)', xanchor='left'))
    else:
        mdict = dict(size=5, color='red', opacity=0.7)

    init_scat = go.Scatter3d(
        x=df_group1[df_group1.sid == sids[0]][x],
        y=df_group1[df_group1.sid == sids[0]][y],
        z=df_group1[df_group1.sid == sids[0]][z],
        mode='markers',
        # marker=dict(size=5, color='red', opacity=0.7),
        marker=mdict,
        name=g1_name,
    )
    scats.append(init_scat)

    if group2:
        if color:
            mdict2 = dict(size=5, color=df_group2[df_group2.sid == sids[0]].p, opacity=1,
                          colorscale='Viridis', cmin=0, cmax=100,
                          showscale=True,
                          line=dict(color='black', width=1),
                          colorbar=dict(title='Momentum (MeV)', xanchor='left'))
        else:
            mdict2 = dict(size=5, color='blue', opacity=0.7)
        init_scat2 = go.Scatter3d(
            x=df_group2[df_group2.sid == sids[0]][x],
            y=df_group2[df_group2.sid == sids[0]][y],
            z=df_group2[df_group2.sid == sids[0]][z],
            mode='markers',
            marker=mdict2,
            name=g2_name)
        scats.append(init_scat2)

    xray_maker(df_xray, scats)

    p_slider = widgets.IntSlider(min=0, max=500, value=0, step=1, continuous_update=False)
    p_slider.description = 'Time Shift'
    p_slider.value = 0
    p_state = time_shifter(group2, color)
    p_slider.on_trait_change(p_state.on_time_change, 'value')

    fig = go.Figure(data=scats, layout=layout)
    # FIXME!
    # g = GraphWidget('https://plot.ly/~BroverCleveland/70/')
    # display(g)
    # display(p_slider)
    # return g, fig
    return fig


def ptrap_layout(title=None, x='Z', y='X', z='Y', x_range=(3700, 17500), y_range=(-1000, 1000),
                 z_range=(-1000, 1000), aspect='default'):
    if aspect == 'default':
        ar = dict(x=6, y=1, z=1)
    elif aspect == 'cosmic':
        ar = dict(x=6, y=1, z=4)
    else:
        raise ValueError('bad value for `aspect`')
    axis_title_size = 18
    axis_tick_size = 14
    layout = go.Layout(
        title=title if title else 'Particle Trapping Time Exercise',
        titlefont=dict(size=30),
        autosize=False,
        width=1400,
        height=650,
        scene=dict(
            xaxis=dict(
                title='{} (mm)'.format(x),
                titlefont=dict(size=axis_title_size, family='Arial Black'),
                tickfont=dict(size=axis_tick_size),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=x_range,
                # range=[7500, 12900],
            ),
            yaxis=dict(
                title='{} (mm)'.format(y),
                titlefont=dict(size=axis_title_size, family='Arial Black'),
                tickfont=dict(size=axis_tick_size),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=y_range
            ),
            zaxis=dict(
                title='{} (mm)'.format(z),
                titlefont=dict(size=axis_title_size, family='Arial Black'),
                tickfont=dict(size=axis_tick_size),
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)',
                range=z_range
            ),
            # aspectratio=dict(x=6, y=1, z=4),
            aspectratio=ar,
            # aspectratio=dict(x=4, y=1, z=1),
            aspectmode='manual',
            camera=dict(
                eye=dict(x=1.99, y=-2, z=2)
                # eye=dict(x=-0.4, y=-2.1, z=0.1),
                # center=dict(x=-0.4, y=0.1, z=0.1),
            ),
        ),
        showlegend=True,
        legend=dict(x=0.7, y=0.9, font=dict(size=18, family='Overpass')),
    )
    return layout


def xray_maker(df_xray, scat_plots):
    '''Helper function to generate the x-ray visualization for particle trapping plots.'''

    print('binning...')
    xray_query = 'xstop<1000 and tstop< 200 and sqrt(xstop*xstop+ystop*ystop)<900'
    df_xray.query(xray_query, inplace=True)

    h, edges = np.histogramdd(np.asarray([df_xray.zstop, df_xray.xstop, df_xray.ystop]).T,
                              bins=(100, 25, 25))
    centers = [(e[1:]+e[:-1])/2.0 for e in edges]
    hadj = np.rot90(h).flatten()/h.max()
    cx, cy, cz = np.meshgrid(*centers)
    df_binned = pd.DataFrame(dict(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), h=hadj))
    df_binned = df_binned.query('h>0')
    df_binned['cats'] = pd.cut(df_binned.h, 200)
    print('binned')
    groups = np.sort(df_binned.cats.unique())
    for i, group in enumerate(groups[0:]):
        df_tmp = df_binned[df_binned.cats == group]
        if i == 0:
            sl = True
        else:
            sl = False
        scat_plots.append(
            go.Scatter3d(
                x=df_tmp.x, y=df_tmp.y, z=df_tmp.z, mode='markers',
                marker=dict(sizemode='diameter', size=4,
                            color='black', opacity=0.02*(i+1), symbol='square'),
                name='X-ray', showlegend=sl, legendgroup='xray'))

    # df_newmat = df_binned.query('12800<x<13000 and sqrt(y**2+z**2)<800')
    # if len(df_newmat) > 0:
    #     scat_plots.append(
    #         go.Scatter3d(
    #             x=df_newmat.x, y=df_newmat.y, z=df_newmat.z, mode='markers',
    #             marker=dict(sizemode='diameter', size=6,
    #                         color='red', opacity=0.4, symbol='square'),
    #             name='X-ray', showlegend=False, legendgroup='xray'))


def conditions_parser(df, conditions, do2pi=False):
    '''Helper function for parsing queries passed to a plotting function.  Special care is taken if
    a 'Phi==X' condition is encountered, in order to select Phi and Pi-Phi'''

    # Special treatment to detect cylindrical coordinates:
    # Some regex matching to find any 'Phi' condition and treat it as (Phi & Phi-Pi).
    # This is done because when specifying R-Z coordinates, we almost always want the
    # 2-D plane that corresponds to that (Phi & Phi-Pi) pair.  In order to plot this,
    # we assign a value of -R to all R points that correspond to the 'Phi-Pi' half-plane

    p = re.compile(r'(?:and)*\s*Phi\s*==\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))')
    phi_str = p.search(conditions)
    conditions_nophi = p.sub('', conditions)
    conditions_nophi = re.sub(r'^\s*and\s*', '', conditions_nophi)
    conditions_nophi = conditions_nophi.strip()
    try:
        phi = float(phi_str.group(1))
    except:
        phi = None

    df = df.query(conditions_nophi)

    # Make radii negative for negative phi values (for plotting purposes)
    if phi is not None:
        isc = np.isclose
        nphi = phi + np.pi if do2pi else phi - np.pi
        df = df[(isc(phi, df.Phi)) | (isc(nphi, df.Phi))]
        df.loc[isc(nphi, df.Phi), 'R'] *= -1

    conditions_title = conditions_nophi.replace(' and ', ', ')
    conditions_title = conditions_title.replace('R!=0', '')
    conditions_title = conditions_title.strip()
    conditions_title = conditions_title.strip(',')
    if phi is not None:
        conditions_title += ', Phi=={0:.2f}'.format(phi)

    return df, conditions_title

def conditions_parser_nonuni(df, conditions, do2pi=False):
    '''Helper function for parsing queries passed to a plotting function.  Special care is taken if
    a 'Phi==X' condition is encountered, in order to select Phi and Pi-Phi'''

    # Special treatment to detect cylindrical coordinates:
    # Some regex matching to find any 'Phi' condition and treat it as (Phi & Phi-Pi).
    # This is done because when specifying R-Z coordinates, we almost always want the
    # 2-D plane that corresponds to that (Phi & Phi-Pi) pair.  In order to plot this,
    # we assign a value of -R to all R points that correspond to the 'Phi-Pi' half-plane

    p = re.compile(r'(?:and)*\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))\s*\<=*\s*Phi\s*\<=*\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))')
    phi_str = p.search(conditions)
    conditions_nophi = p.sub('', conditions)
    conditions_nophi = re.sub(r'^\s*and\s*', '', conditions_nophi)
    conditions_nophi = conditions_nophi.strip()
    try:
        phi_l = float(phi_str.group(1))
        phi_h = float(phi_str.group(2))
        phi_mean = (phi_l + phi_h) / 2
    except:
        phi_l = None
        phi_h = None
        phi_mean = None

    df = df.query(conditions_nophi)

    # Make radii negative for negative phi values (for plotting purposes)
    if phi_l is not None:
        isc = np.isclose
        if do2pi:
            #nphi = phi+np.pi
            raise NotImplementedError("do2pi==True not implemented for non-uniform data.")
        else:
            if isc(phi_mean, 0):
                nphi_l = phi_l + np.pi
                nphi_h = phi_h + np.pi
                use_abs = True
            else:
                nphi_l = phi_l-np.pi
                nphi_h = phi_h-np.pi
                use_abs = False
        #df = df[(isc(phi, df.Phi)) | (isc(nphi, df.Phi))]
        if use_abs:
            df = df.query(f'({phi_l} <= abs(Phi) <= {phi_h}) or ({nphi_l} <= abs(Phi) <= {nphi_h})').copy()
            df.loc[(df.Phi.abs() >= nphi_l) & (df.Phi.abs() <= nphi_h), 'R'] *= -1
        else:
            df = df.query(f'({phi_l} <= Phi <= {phi_h}) or ({nphi_l} <= Phi <= {nphi_h})').copy()
            #df.loc[isc(nphi, df.Phi), 'R'] *= -1
            df.loc[(df.Phi >= nphi_l) & (df.Phi <= nphi_h), 'R'] *= -1

    conditions_title = conditions_nophi.replace(' and ', ', ')
    # seems wrong to strip this out of the title
    # conditions_title = conditions_title.replace('R!=0', '')
    conditions_title = conditions_title.strip()
    conditions_title = conditions_title.strip(',')
    if phi_mean is not None:
        #conditions_title += f', Phi=={phi_mean:.2f}+-{phi_h - phi_mean:0.2f}'
        conditions_title += f', {phi_l:.2f}<=Phi<={phi_h:.2f}'

    return df, conditions_title
#re.compile(r'(?:and)*\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))\s*\<=*\s*Phi\s*\<=*\s*([-+]?(?:[0-9]*\.[0-9]+|[0-9]+))')

def xray_maker_2(df_xray, bz=50, bx=15, by=15):
    '''Helper function to generate the x-ray visualization for particle trapping plots.'''

    print('binning...')
    init_notebook_mode()
    xray_query = 'xstop<1000 and tstop< 200 and sqrt(xstop*xstop+ystop*ystop)<900'
    df_xray.query(xray_query, inplace=True)

    h, e = np.histogramdd(np.asarray([df_xray.zstop, df_xray.xstop, df_xray.ystop]).T,
                          bins=(bz, bx, by), range=[(3500, 15000), (-900, 900), (-900, 900)])
    h_adj = h/h.max()

    cube_list = []
    for i in range(bz):
        for j in range(bx):
            for k in range(by):
                if h_adj[i][j][k] == 0:
                    continue
                cube_list.append(
                    go.Mesh3d(
                        x=[e[0][i], e[0][i], e[0][i+1], e[0][i+1],
                           e[0][i], e[0][i], e[0][i+1], e[0][i+1]],
                        y=[e[1][j], e[1][j+1], e[1][j+1], e[1][j],
                           e[1][j], e[1][j+1], e[1][j+1], e[1][j]],
                        z=[e[2][k], e[2][k], e[2][k], e[2][k],
                           e[2][k+1], e[2][k+1], e[2][k+1], e[2][k+1]],
                        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                        color='00FFFF',
                        opacity=h_adj[i][j][k]*0.5
                    )
                )

    layout = ptrap_layout(title='test xray')
    fig = go.Figure(data=go.Data(cube_list), layout=layout)
    iplot(fig)


def mu2e_plot3d_ptrap_anim_2(df, x, y, z, df_xray):
    """Generate 3D scatter plots, with a slider widget typically for visualizing 3D positions of
    charged particles.

    Generate a 3D scatter plot for a given DF and three columns. Due to the large number of points
    that are typically plotted, :mod:`matplotlib` is not supported. A slider widget is generated,
    corresponding to the evolution in time.

    Notes:
        * Currently many options are hard-coded.
        * Use `g.plot(fig)` after executing this function.
    Args:
        df_group1 (pandas.DataFrame): The input DF, must contain columns with labels corresponding
        to the 'x', 'y', and 'z' args.
        x (str): Name of the first variable.
        y (str): Name of the second variable.
        z (str): Name of the third variable.
        df_xray: (:class:`pandas.DataFrame`): A seperate DF, representing the geometry of the
            material that is typically included during particle simulation.
        df_group2: (:class:`pandas.DataFrame`, optional): A seperate DF, representing the geometry
            of the material that is typically included during particle simulation.
        color (optional): Being implemented...
        title (str, optional): Title.

    Returns:
        (g, fig) for plotting.
    """
    init_notebook_mode()
    sids = np.sort(df.sid.unique())
    layout = ptrap_layout(title='anim test')
    figure = {
        'data': [],
        'layout': layout,
        'frames': [],
    }

    # figure['layout']['sliders'] = [{
    #     'args': [
    #         'transition', {
    #             'duration': 400,
    #             'easing': 'cubic-in-out'
    #         }
    #     ],
    #     'initialValue': '0',
    #     'plotlycommand': 'animate',
    #     'values': sids,
    #     'visible': True
    # }]
    figure['layout']['updatemenus'] = [
        {'type': 'buttons', 'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None]},
                                        {'args': [[None], {'frame': {'duration': 0, 'redraw': False,
                                                                     'easing': 'quadratic-in-out'},
                                                           'mode': 'immediate', 'transition':
                                                           {'duration': 0}}], 'label': 'Pause',
                                         'method': 'animate'}]}]

    sliders_dict = {
        'active': 0, 'yanchor': 'top', 'xanchor': 'left',
        'currentvalue': {'font': {'size': 20}, 'prefix': 'SID:', 'visible': True, 'xanchor':
                         'right'}, 'transition': {'duration': 100, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50}, 'len': 0.9, 'x': 0.1, 'y': 0, 'steps': []}
    scats = []
    xray_maker(df_xray, scats)

    for sid in sids:
        slider_step = {'args': [
            [sid],
            {'frame': {'duration': 300, 'redraw': True},
             'mode': 'immediate',
             'transition': {'duration': 300}}], 'label': sid, 'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    # init_scat = dict(
    #     x=df[df.sid == sids[0]][x],
    #     y=df[df.sid == sids[0]][y],
    #     z=df[df.sid == sids[0]][z],
    #     mode='markers',
    #     type='scatter3d'
    # )
    # scats.append(init_scat)
    frames = [dict(data=[dict(
        x=df[df.sid == sids[k]][x],
        y=df[df.sid == sids[k]][y],
        z=df[df.sid == sids[k]][z],
        mode='markers',
        type='scatter3d',
        marker=dict(color='red', size=10, symbol='circle', opacity=0.5)
    )]) for k in range(len(sids))]

    figure['layout']['sliders'] = [sliders_dict]
    figure['data'] = scats
    figure['frames'] = frames
    # fig = go.Figure(data=scats, layout=layout, frames=frames)
    iplot(figure)
