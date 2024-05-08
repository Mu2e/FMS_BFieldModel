#! /usr/bin/env python
"""Module for generating mock FMS measurements.

The Field Mapping System (FMS) hall probe equipment will travel through the PS and DS, measuring a
sparse set of magnetic field components at regular positions.  These field measurements will then be
fed to the fitting software, which should be able to reproduce the entire magnetic field map within
a given region of interest.

The :class:`mu2e.hallprober.HallProbeGenerator` takes an external field simulation, made available
by the Mu2E collaboration, and generates a set of mock measurements, subject to many potential
variations.  The default geometry is cylindrical, but can be changed to cartesian.  The default
measurements are "perfect," such that no additional sources of error are introduced into each
component value.  There are optional functions that modify the measurements to reflect various
errors that could arise during data-taking.

This module also contains functions that assist in the full scope of operation, from field
simulation input, to hall probe data generation, to field fitting, to final plotting and analysis.

Example:
    Incomplete excerpt, see :func:`mu2e.fieldfitter.field_map_analysis` and `scripts/hallprobesim`
    for more typical use cases:

    .. code-block:: python

        # assuming config files already defined...

        In [10]: input_data = DataFileMaker(cfg_data.path, use_pickle=True).data_frame
        ...      input_data.query(' and '.join(cfg_data.conditions))

        In [11]: hpg = HallProbeGenerator(
        ...         input_data, z_steps = cfg_geom.z_steps,
        ...         r_steps = cfg_geom.r_steps, phi_steps = cfg_geom.phi_steps,
        ...         x_steps = cfg_geom.x_steps, y_steps = cfg_geom.y_steps)


        # Introduce miscalibrations
        # Do something here with uncertainties

        In [13]: print hpg.get_toy().head()
        Out[13]:
        ...                  X    Y       Z        Bx   By        Bz      R       Phi  Bphi
        ...      833646 -800.0  0.0  4221.0  0.039380  0.0  1.976202  800.0  3.141593  -0.0
        ...      833650 -800.0  0.0  4321.0 -0.015489  0.0  1.985269  800.0  3.141593   0.0
        ...      833654 -800.0  0.0  4421.0 -0.068838  0.0  1.975510  800.0  3.141593   0.0
        ...      833658 -800.0  0.0  4521.0 -0.122017  0.0  1.944508  800.0  3.141593   0.0
        ...      833662 -800.0  0.0  4621.0 -0.170256  0.0  1.885879  800.0  3.141593   0.0

        ...                  Br     Bzerr     Brerr       Bphierr     Bxerr         Byerr
        ...      833646 -0.039099  0.000198  0.000004  1.000000e-15  0.000004  1.000000e-15
        ...      833650  0.015771  0.000199  0.000002  1.000000e-15  0.000002  1.000000e-15
        ...      833654  0.069119  0.000198  0.000007  1.000000e-15  0.000007  1.000000e-15
        ...      833658  0.122293  0.000194  0.000012  1.000000e-15  0.000012  1.000000e-15
        ...      833662  0.170523  0.000189  0.000017  1.000000e-15  0.000017  1.000000e-15

Notes:
    * Remove 'getter' usage (non-pythonic)
    * Static method could probably be placed elsewhere
    * Interpolation scheme is not accurate enough for this analysis, eventually replace with
        something more powerful
    * Should analysis functions be located in this module?


*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import time
import shutil
import math
import copy
import collections
import warnings
import six.moves.cPickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import Rbf
import mu2e
from mu2e.dataframeprod import DataFrameMaker
from mu2e.fieldfitter_redux2 import FieldFitter
from mu2e.mu2eplots import mu2e_plot3d, mu2e_plot3d_nonuniform_test, mu2e_plot3d_nonuniform_cyl
from mu2e.syst_unc import laserunc, metunc, apply_field_unc
from mu2e import mu2e_ext_path
import imp
from six.moves import range
# parallelization
from joblib import Parallel, delayed
import multiprocessing
from lmfit import Model
import re
from scipy.stats import t
from scipy.special import erf
interp_studies = imp.load_source(
    'interp_studies', '/home/ckampa/coding/Mu2E/scripts/FieldFitting/interp_studies.py')

warnings.simplefilter('always', DeprecationWarning)


class HallProbeGenerator(object):
    """Class for generating toy outputs for mimicing the Mu2E FMS hall probe measurements.

    This class takes an input DF, and a set of config namedtuples that contain necessary information
    regarding the desired geometry and spacing of the mock data that will be produced.  Unless
    interpolation is specified, the mock data must always be a subset (usually sparse) of the input
    data. Additional methods may be used to manipulate the output data in order to represent
    different measurement scenarios (like introducing measurement error).

    Args:
        input_data (:class:`pandas.DataFrame`): The input magnetic field datal properly formatted
            via :mod:`mu2e.dataframeprod`.
        z_steps (int or List[numbers], optional): If an int, will select `z_steps` number of evenly
            spaced z values for sampling. If a list of ints, will select those specific z values.
        x_steps (int or List[numbers], optional): If an int, will select `2*x_steps` number of
            evenly.  spaced x values for sampling, symmetrically about 0. If a list of ints, will
            select those specific x values. This arg is overridden if 'r_steps' is not None.
        y_steps (int or List[numbers], optional): If an int, will select `2*y_steps` number of
            evenly spaced y values for sampling, symmetrically about 0. If a list of ints, will
            select those specific y values. This arg is overridden if 'r_steps' is not None.
        r_steps (int or List[numbers], optional): If an int, will select `r_steps` number of evenly
            spaced r values for sampling. If a list of ints, will select those specific r values.
            This arg is overrides `x_steps` and `y_steps`.
        phi_steps (List[numbers], optional): Will select specified `phi` values, along with their
            `pi-val` counterparts. Pairs with `r_steps`.
        interpolate (bool, optional): If true, perform the interpolation procedure instead of
            selecting values directly from the input field map.

    Attributes:
        cylindrical_norm (func): Calculate norm (distance) in cylindrical coordinates.
        full_field (:class:`pandas.DataFrame`): Input data, before selection.
        sparse_field (:class:`pandas.DataFrame`): Output data, after selection is applied. This is
            the primary object that is produced by the `hallprober` class.
        r_steps (int or List[numbers]): The input arg `r_steps`.
        phi_steps (List[numbers]): The input arg `phi_steps`.
        z_steps (int or List[numbers]): The input arg `z_steps`.

    Notes:
        Some clean-up and rewrites should be done in the future, may lead to slight changes in call
        signature. Interpolation methods will be expanded shortly.

    """

    @staticmethod
    def cylindrical_norm(x1, x2):
        """Define a distance metric in cylindrical coordinates."""
        return np.sqrt(
            (x1[0, :]*np.cos(x1[1, :])-x2[0, :]*np.cos(x2[1, :]))**2 +
            (x1[0, :]*np.sin(x1[1, :])-x2[0, :]*np.sin(x2[1, :]))**2 +
            (x1[2, :]-x2[2, :])**2)

    def __init__(self, input_data, z_steps=None, x_steps=None,
                 y_steps=None, r_steps=None, phi_steps=None, interpolate=False,
                 do2pi=False, do_selection=True):
        self.full_field = input_data        
        self.sparse_field = self.full_field
        self.r_steps = r_steps
        self.z_steps = z_steps
        self.x_steps = x_steps
        self.y_steps = y_steps
        self.phi_steps = phi_steps

        # modify sparse_field if necessary
        if do_selection:
            if self.phi_steps:
                self.phi_nphi_steps = list(self.phi_steps[:])
                for phi in self.phi_steps:
                    if do2pi:
                        # determine phi and phi+pi
                        self.phi_nphi_steps.append(phi+np.pi)
                    else:
                        # determine phi and negative phi
                        if phi == 0:
                            self.phi_nphi_steps.append(np.pi)
                        else:
                            self.phi_nphi_steps.append(phi-np.pi)

            if self.x_steps or self.y_steps:
                raise NotImplementedError('Oh no! you got lazy during refactoring')

            if interpolate is not False:
                self.interpolate_points(interpolate)
            else:
                # Require exact match for z, 'isclose' for R / Phi (in case of rounding
                # errors due to conversion from cartesian)
                self.sparse_field = self.sparse_field[
                    (np.isclose(self.sparse_field.Phi.values[:, None],
                                self.phi_nphi_steps).any(axis=1)) &
                    (np.isclose(self.sparse_field.R.values[:, None],
                                np.ravel(self.r_steps)).any(axis=1)) &
                    (self.sparse_field.Z.isin(self.z_steps))]
                self.sparse_field = self.sparse_field.sort_values(['Z', 'R', 'Phi'])
        # otherwise use the entire dataset
        else:
            self.sparse_field = self.sparse_field.sort_values(['Z', 'R', 'Phi'])

    def takespread(self, sequence, num):
        """Return an evenly-spaced sequence of length `num` from the input sequence.

        Args:
            sequence (collections.Sequence): A list-like object of numeric values.
            num (int): Number of desired values to be selected.

        Returns:
            spread (List[numbers]):
        """
        length = float(len(sequence))
        spread = []
        for i in range(num):
            spread.append(sequence[int(math.ceil(i * length / num))])
        return spread

    def apply_selection(self, coord, steps):
        """
        Apply selections to different coordinate values in order to create a sparse dataframe from
        the input data.

        Args:
            coord (str): Label of the dataframe column that will be queried.  Typically a positional
                coordinate ('X', 'Y', 'Z', 'R', 'Phi').
            steps (List[numbers]): If a list, those values will be taken
                (along with their respective negative or inverses, if applicable)

        Returns:
            Nothing. Generates `spare_field` class member.
        """

        if isinstance(steps, int):
            if coord in ['Z', 'R']:
                coord_vals = np.sort(self.full_field[coord].unique())
                coord_vals = self.takespread(coord_vals, steps)

            else:
                coord_vals = np.sort(self.full_field[coord].abs().unique())[:steps]
                coord_vals = np.concatenate((coord_vals, -coord_vals[np.where(coord_vals > 0)]))

        elif isinstance(steps, collections.Sequence) and type(steps) != str:
            if coord == 'Phi':
                coord_vals = []
                for step in steps:
                    coord_vals.append(step)
                    if step != 0:
                        coord_vals.append(step-np.pi)
                    else:
                        coord_vals.append(step+np.pi)
            elif coord == 'R':
                if isinstance(steps[0], collections.Sequence):
                    coord_vals = np.sort(np.unique([val for sublist in steps for val in sublist]))
                else:
                    coord_vals = steps
            elif coord in ['Z', 'X', 'Y']:
                coord_vals = steps
        elif steps == 'all':
                coord_vals = np.sort(self.full_field[coord].unique())
        else:
            raise TypeError(coord+" steps must be scalar or list of values!")

        if coord == 'R' or coord == 'Phi':
            self.sparse_field = self.sparse_field.query(
                '|'.join(['(-1e-6<'+coord+'-'+str(i)+'<1e-6)' for i in coord_vals])
            )
        else:
            self.sparse_field = self.sparse_field[self.sparse_field[coord].isin(coord_vals)]
        if len(self.sparse_field[coord].unique()) != len(coord_vals):
            print('Warning!: specified vals:')
            print(np.sort(coord_vals))
            print('remaining vals:')
            print(np.sort(self.sparse_field[coord].unique()))

    def get_toy(self):
        """Return `sparse_field`. Deprecated."""
        warnings.warn(("`get_toy()` is deprecated, please use the `sparse_field` class member"),
                      DeprecationWarning)
        return self.sparse_field

    def interpolate_points(self, version=1):
        """Method for obtaining required selection through interpolation.  Work in progress."""
        if version == 'load1':
            if os.path.isfile(mu2e_ext_path+'tmp_rbf.p'):
                self.sparse_field = pkl.load(open(mu2e_ext_path+'tmp_rbf.p', "rb"))
                return
        elif version == 'load2':
            if os.path.isfile(mu2e_ext_path+'tmp_phi.p'):
                self.sparse_field = pkl.load(open(mu2e_ext_path+'tmp_phi.p', "rb"))
                return
        elif version == 'load3':
            if os.path.isfile(mu2e_ext_path+'tmp_quad.p'):
                self.sparse_field = pkl.load(open(mu2e_ext_path+'tmp_quad.p', "rb"))
                return

        elif version == 1:
            row_list = []
            all_phis = []
            for phi in self.phi_steps:
                all_phis.append(phi)
                if phi == 0:
                    all_phis.append(np.pi)
                else:
                    all_phis.append(phi-np.pi)
            print('interpolating data points')
            for r in tqdm(self.r_steps[0], desc='R (mm)', leave=False):
                for p in tqdm(all_phis, desc='Phi (rads)', leave=False):
                    for z in tqdm(self.z_steps, desc='Z (mm)', leave=False):
                        x = r*math.cos(p)
                        y = r*math.sin(p)
                        field_subset = self.full_field.query(
                            '{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
                                x-100, x+100, y-100, y+100, z-100, z+100))

                        rbf = Rbf(field_subset.X, field_subset.Y, field_subset.Z,
                                  field_subset.Bz, function='linear')
                        bz = rbf(x, y, z)
                        rbf = Rbf(field_subset.X, field_subset.Y, field_subset.Z,
                                  field_subset.Bx, function='linear')
                        bx = rbf(x, y, z)
                        rbf = Rbf(field_subset.X, field_subset.Y, field_subset.Z,
                                  field_subset.By, function='linear')
                        by = rbf(x, y, z)

                        br = bx*math.cos(p)+by*math.sin(p)
                        bphi = -bx*math.sin(p)+by*math.cos(p)

                        row_list.append([r, p, z, br, bphi, bz])

            row_list = np.asarray(row_list)
            self.sparse_field = pd.DataFrame({
                'R': row_list[:, 0], 'Phi': row_list[:, 1], 'Z': row_list[:, 2],
                'Br': row_list[:, 3], 'Bphi': row_list[:, 4], 'Bz': row_list[:, 5]})

        elif version == 2:
            row_list = []
            all_phis = []
            for phi in self.phi_steps:
                all_phis.append(phi)
                if phi == 0:
                    all_phis.append(np.pi)
                else:
                    all_phis.append(phi-np.pi)
            print('interpolating data points')
            for r in tqdm(self.r_steps[0], desc='R (mm)', leave=False):
                for p in tqdm(all_phis, desc='Phi (rads)', leave=False):
                    for z in tqdm(self.z_steps, desc='Z (mm)', leave=False):
                        x = r*math.cos(p)
                        y = r*math.sin(p)
                        field_subset = self.full_field.query(
                            '{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
                                x-100, x+100, y-100, y+100, z-100, z+100))

                        _, b_lacey = interp_studies.interp_phi(field_subset, x, y, z, plot=False)
                        bx = b_lacey[0]
                        by = b_lacey[1]
                        bz = b_lacey[2]

                        br = bx*math.cos(p)+by*math.sin(p)
                        bphi = -bx*math.sin(p)+by*math.cos(p)

                        row_list.append([r, p, z, br, bphi, bz])

            row_list = np.asarray(row_list)
            self.sparse_field = pd.DataFrame({
                'R': row_list[:, 0], 'Phi': row_list[:, 1], 'Z': row_list[:, 2],
                'Br': row_list[:, 3], 'Bphi': row_list[:, 4], 'Bz': row_list[:, 5]})

        elif version == 3:
            row_list = []
            all_phis = []
            for phi in self.phi_steps:
                all_phis.append(phi)
                if phi == 0:
                    all_phis.append(np.pi)
                else:
                    all_phis.append(phi-np.pi)
            print('interpolating data points')
            for r in tqdm(self.r_steps[0], desc='R (mm)', leave=False):
                for p in tqdm(all_phis, desc='Phi (rads)', leave=False):
                    for z in tqdm(self.z_steps, desc='Z (mm)', leave=False):
                        x = r*math.cos(p)
                        y = r*math.sin(p)
                        field_subset = self.full_field.query(
                            '{0}<=X<={1} and {2}<=Y<={3} and {4}<=Z<={5}'.format(
                                x-100, x+100, y-100, y+100, z-100, z+100))

                        _, b_lacey = interp_studies.interp_phi_quad(field_subset, x, y, z,
                                                                    plot=False)
                        bx = b_lacey[0]
                        by = b_lacey[1]
                        bz = b_lacey[2]

                        br = bx*math.cos(p)+by*math.sin(p)
                        bphi = -bx*math.sin(p)+by*math.cos(p)

                        row_list.append([r, p, z, br, bphi, bz])

            row_list = np.asarray(row_list)
            self.sparse_field = pd.DataFrame({
                'R': row_list[:, 0], 'Phi': row_list[:, 1], 'Z': row_list[:, 2],
                'Br': row_list[:, 3], 'Bphi': row_list[:, 4], 'Bz': row_list[:, 5]})

        self.sparse_field = self.sparse_field[['R', 'Phi', 'Z', 'Br', 'Bphi', 'Bz']]
        if version == 1:
            pkl.dump(self.sparse_field, open(mu2e_ext_path+'tmp_rbf.p', "wb"), pkl.HIGHEST_PROTOCOL)
        elif version == 2:
            pkl.dump(self.sparse_field, open(mu2e_ext_path+'tmp_phi.p', "wb"), pkl.HIGHEST_PROTOCOL)
        elif version == 3:
            pkl.dump(self.sparse_field, open(mu2e_ext_path+'tmp_quad.p', "wb"),
                     pkl.HIGHEST_PROTOCOL)

        print('interpolation complete')
# Not being used - check if we would want this for some purpose
'''
def plot_parallel_helper(step, ABC, conditions, df, cfg_plot, save_dir, aspect, cfg_geom, parallel):
    # FIXME! Better way than hardcoding "dPhi" in the query?
    if cfg_geom.geom == 'cyl':
        conditions_str = ' and '.join(conditions+(f'{step-np.pi/32} <= Phi <= {step+np.pi/32}',))
    else:
        raise NotImplementedError('geom=="cart" is not implemented for the non-uniform grid plotting.')
    fig, ax = mu2e_plot3d_nonuniform_cyl(df, ABC[1], ABC[0], ABC[2], conditions=conditions_str, mode=cfg_plot.plot_type, cut_color=cfg_plot.zlims, info=None, save_dir=save_dir, save_name=None,
                                         df_fit=True, ptype='3d', aspect=aspect, cmin=None, cmax=None, fig=None, ax=None,
                                         do_title=True, title_simp=None, do2pi=cfg_geom.do2pi, units='m', show_plot=False, parallel=parallel)
    return fig, ax

'''
def make_fit_plots(df, cfg_data, cfg_geom, cfg_plot, name, aspect='square', parallel=True, df_fine=None):
    """Make a series of comparison plots with the fit output and hallprobe input.

    This function takes input DFs and `namedtuple` config files, and generates a comprehensive
    set of comparison plots in 3D.  The plots are typically of the form 'B-component vs two
    positional variables', where the input hall probe measurements are displayed as a scatter plot,
    and the resulting fit is displayed as wireframe plot.  Additionally, heatmaps are produced to
    display the absolute residuals between the data and fit.  The heatmaps are produced separately
    in 'mpl' mode, or are integrated into the main plot in 'plotly' mode.

    Example:
        Incomplete excerpt, see `scripts/hallprobesim` for more typical use cases:

        .. code-block:: python

            # assuming config files already defined...

            In [12]: ff = FieldFitter(sparse_field)

            In [13]: ff.fit(cfg_params, cfg_pickle)
            ...      # This will take some time, especially for many data points and free params

            In [14]: ff.merge_data_fit_res() # merge the results in for easy plotting

            In [15]: cfg_plot = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir')

            In [16]: cfg_plot_plotly = cfg_plot('plotly',[-10,10],'html', None)
            ...      # make plotly plots, set limits, save loc, etc.

            In [17]: make_fit_plots(ff.input_data, cfg_data, cfg_geom, cfg_plot, name)


    Args:
       df (:class:`pandas.DataFrame`): DF that contains both the input data and the fit data.
       cfg_data (namedtuple): Data config file.
       cfg_geom (namedtuple): Geometry config file.
       cfg_plot (namedtuple): Plotting config file.
       name (str): Name of output save directory.

    Returns:
       Nothing.

    Todo:
        * Move this function to more logical module.
    """

    geom = cfg_geom.geom
    plot_type = cfg_plot.plot_type
    if geom == 'cyl':
        steps = cfg_geom.phi_steps
    if geom == 'cart':
        steps = cfg_geom.y_steps
    conditions = cfg_data.conditions

    ABC_geom = {'cyl': [['R', 'Z', 'Bz'], ['R', 'Z', 'Br'], ['R', 'Z', 'Bphi']],
                'cart': [['X', 'Z', 'Bx'], ['X', 'Z', 'By'], ['X', 'Z', 'Bz']]}

    if cfg_plot.save_loc == 'local':
        save_dir = mu2e.mu2e_ext_path+'plots/'+name
    elif cfg_plot.save_loc == 'html':
        save_dir = mu2e.mu2e_ext_path+'plots/html/'+name
        # save_dir = '/home/ckampa/Plots/FieldFitting/'+name

    for step in steps:
        for ABC in ABC_geom[geom]:
            if geom == 'cyl':
                conditions_str = ' and '.join(conditions+('Phi=={}'.format(step),))
            else:
                conditions_str = ' and '.join(conditions+('Y=={}'.format(step),))
            # FIXME! This is the figure object, not save name
            if plot_type == 'mpl_nonuni':
                mu2e_plot3d_nonuniform_test(df, ABC[0], ABC[1], ABC[2], conditions=conditions_str,
                                            df_fit=True, mode=plot_type, save_dir=save_dir,
                                            do2pi=cfg_geom.do2pi, units='m', df_fine=df_fine)
            else:
                save_name = mu2e_plot3d(df, ABC[0], ABC[1], ABC[2], conditions=conditions_str,
                                        df_fit=True, mode=plot_type, save_dir=save_dir,
                                        do2pi=cfg_geom.do2pi, units='m', df_fine=df_fine)
                

    # TODO add flag to turn on / off showing plots
    #if plot_type in ['mpl', 'mpl_nonuni']:
    #    plt.show()


def field_map_analysis(name, cfg_data, cfg_geom, cfg_params, cfg_pickle, cfg_plot, profile=False, aspect='square', parallel_plots=True, iterative=False):
    """Universal function to perform all types of hall probe measurements, plots, and further
    analysis.

    Args:
        name (str): Name of output directory.
        cfg_data (namedtuple): Data config file.
        cfg_geom (namedtuple): Geometry config file.
        cfg_params (namedtuple): Fit parameter config file.
        cfg_pickle (namedtuple): Pickling config file.
        cfg_plot (namedtuple): Plotting config file.
        profile (bool, optional): If True, return data before fitting for the purposes of continuing
            on to profiling methods.

    Returns:
        If `profile==False`, returns a DF of the hall probe data, and the FieldFitter object. If
        `profile==True`, returns field components and position values.
    """

    plt.close('all')

    # For position uncertainties, use posunc class method get_shifted_field to get map with
    # field assessed at alternate coordinates
    if cfg_geom.systunc is not None and ('LaserUnc' in cfg_geom.systunc or 'MetUnc' in cfg_geom.systunc):
        print(f'Applying {cfg_geom.systunc} position uncertainty')
        if cfg_geom.systunc.startswith('LaserUnc'):
            myunc = laserunc(cfg_data.path)
            myunc.transformations = pkl.load(open('transformations.pkl',"rb"))
        else:
            toy = cfg_geom.systunc.replace('MetUnc','')
            myunc = metunc(cfg_data.path,toy)
        input_data = myunc.get_shifted_field()
    else:
        input_data = DataFrameMaker(cfg_data.path, input_type='pkl').data_frame

    if not cfg_geom.do_selection:
        input_data = input_data.query(' and '.join(cfg_data.conditions))

    hpg = HallProbeGenerator(input_data, z_steps=cfg_geom.z_steps,
                             r_steps=cfg_geom.r_steps, phi_steps=cfg_geom.phi_steps,
                             x_steps=cfg_geom.x_steps, y_steps=cfg_geom.y_steps,
                             interpolate=cfg_geom.interpolate, do2pi=cfg_geom.do2pi,
                             do_selection=cfg_geom.do_selection)

    if cfg_geom.systunc is not None and ('Calib' in cfg_geom.systunc or 'TempUnc' in cfg_geom.systunc):
        print(f'Applying {cfg_geom.systunc} field uncertainty')
        hall_measure_data = apply_field_unc(cfg_geom.systunc, hpg.sparse_field)
    else:
        hall_measure_data = hpg.sparse_field

    # Smear input values by noise
    #if cfg_params.noise is not None:
    #    for field in ['Br','Bphi','Bz']:
    #        noise_vec = np.random.normal(0,cfg_params.noise,hall_measure_data[field].shape[0])
    #        hall_measure_data.loc[:,field] += noise_vec
    #    hall_measure_data.eval('Bx = Br*cos(Phi)-Bphi*sin(Phi)', inplace=True)
    #    hall_measure_data.eval('By = Bphi*cos(Phi)+Br*sin(Phi)', inplace=True)
        
    print(hall_measure_data.head())
    print(hall_measure_data.columns)

    # repeat with any fields to add on?
    if cfg_params.cfg_calc_data is None:
        calc_data = None
    elif cfg_params.cfg_calc_data == "Calc_Bus":
        raise NotImplementedError('Oh no! On-the-fly calculations of bus bar contributions using helicalc is not yet supported.')
    else:
        # TODO currently not applying systematic uncertainties here
        # note we really only need the data path. we assume other conditions are equivalent to main data.
        cfg_calc_data = cfg_params.cfg_calc_data
        # use Hall probe generator
        input_calc_data = DataFrameMaker(cfg_calc_data.path, input_type='pkl').data_frame
        # THIS ISN'T DOING THE QUERY
        if not cfg_geom.do_selection:
            input_calc_data = input_calc_data.query(' and '.join(cfg_data.conditions))
        hpg_calc = HallProbeGenerator(input_calc_data, z_steps=cfg_geom.z_steps,
                                      r_steps=cfg_geom.r_steps, phi_steps=cfg_geom.phi_steps,
                                      x_steps=cfg_geom.x_steps, y_steps=cfg_geom.y_steps,
                                      interpolate=cfg_geom.interpolate, do2pi=cfg_geom.do2pi,
                                      do_selection=cfg_geom.do_selection)
        hall_measure_data_calc = hpg_calc.sparse_field
        # now grab calculated values
        br_calc_data = hall_measure_data_calc['Br'].values
        bphi_calc_data = hall_measure_data_calc['Bphi'].values
        bz_calc_data = hall_measure_data_calc['Bz'].values
        calc_data = {'br': br_calc_data, 'bphi': bphi_calc_data, 'bz': bz_calc_data}

    print("hall_measure_data")
    print(hall_measure_data)
    ff = FieldFitter(hall_measure_data, calc_data)
    if profile:
        ZZ, RR, PP, Bz, Br, Bphi = ff.fit(cfg_params, cfg_pickle, profile=profile)
        return ZZ, RR, PP, Bz, Br, Bphi
    else:
        ff.fit(cfg_params, cfg_pickle, iterative=iterative)

    saveunc = (cfg_geom.systunc is None) and not cfg_pickle.recreate and ('Unc' not in cfg_data.path)
    iscart = (cfg_geom.geom == 'cart')
    ff.merge_data_fit_res(saveunc,iscart)
    print(ff.input_data)

    if cfg_geom.systunc is not None: # True systematic --> save variant field and fit
        pkl.dump(ff.input_data, open(cfg_data.path.split('.')[0]+f'_{cfg_geom.systunc}.Mu2E.Fit.p', "wb"), pkl.HIGHEST_PROTOCOL)
    elif cfg_pickle.recreate is False: # True nominal --> save nominal field and fit
        #if cfg_params.noise is not None:
        #    pkl.dump(ff.input_data, open(cfg_data.path.split('.')[0]+f'_noise{cfg_params.noise.replace(".","p")}.Mu2E.Fit.p', "wb"), pkl.HIGHEST_PROTOCOL)
        #else:
        pkl.dump(ff.input_data, open(cfg_data.path.split('.')[0]+'.Mu2E.Fit.p', "wb"), pkl.HIGHEST_PROTOCOL)
    elif cfg_geom.geom == 'cart': #Special case --> save 'fitted' cartesian field
        if (cfg_pickle.load_name.endswith('Unc') or 'Rebar' in cfg_pickle.load_name) and 'pinn' not in cfg_pickle.load_name:
            pkl.dump(ff.input_data, open(cfg_data.path.split('.')[0]+f'_{cfg_pickle.load_name.split("_")[-1]}.Mu2E.Fit.p', "wb"), pkl.HIGHEST_PROTOCOL)
        else:
            pkl.dump(ff.input_data, open(cfg_data.path.split('.')[0]+'.Mu2E.Fit.p', "wb"), pkl.HIGHEST_PROTOCOL)

    if cfg_plot.df_fine is not None:
        df_fine = DataFrameMaker(cfg_plot.df_fine, input_type='pkl').data_frame
        df_fine = df_fine.sort_values(['Z', 'R', 'Phi'])
        print("df_fine")
        print(df_fine)
        # Need to manually define functional form
        ff_fine = FieldFitter(df_fine, None)
        from collections import namedtuple
        pickle_temp = namedtuple('pickle_temp', 'use_pickle save_pickle load_name save_name recreate')

        cfg_pickle_fine = pickle_temp(use_pickle=False, save_pickle=False,
                                      load_name='fine',
                                      save_name='fine', recreate=False)
        ff_fine.prep_fit_func(cfg_params, cfg_pickle_fine)
        model_fine = ff_fine.model()
        fit_fine = model_fine.eval(r=df_fine.R.values, z=df_fine.Z.values, phi=df_fine.Phi.values, x=df_fine.X.values, y=df_fine.Y.values, params=ff.params)
        df_fine.loc[:,'Br_fit']   = fit_fine[0:len(fit_fine)//3]
        df_fine.loc[:,'Bz_fit']   = fit_fine[len(fit_fine)//3:2*len(fit_fine)//3]
        df_fine.loc[:,'Bphi_fit'] = fit_fine[2*len(fit_fine)//3:]

        # If fit uncertainties were saved in main fit, compute also for fine grid
        if saveunc:
            fit_unc = custom_eval_unc(model=model_fine, r=df_fine.R.values, z=df_fine.Z.values, phi=df_fine.Phi.values, x=df_fine.X.values, y=df_fine.Y.values, params=ff.params)
            df_fine.loc[:,'Br_unc']   = fit_unc[0:len(fit_unc)//3]
            df_fine.loc[:,'Bz_unc']   = fit_unc[len(fit_unc)//3:2*len(fit_unc)//3]
            df_fine.loc[:,'Bphi_unc'] = fit_unc[2*len(fit_unc)//3:]
            if cfg_params.noise is not None:
                pkl.dump(df_fine, open(cfg_plot.df_fine.split('.')[0]+f'_noise{cfg_params.noise}.Mu2E.Fit.p', "wb"), pkl.HIGHEST_PROTOCOL)
            else:
                pkl.dump(df_fine, open(cfg_plot.df_fine.split('.')[0]+'.Mu2E.Fit.p', "wb"), pkl.HIGHEST_PROTOCOL)

        # 'Fit' to nominal with syst. params
        elif cfg_pickle.load_name.endswith('Unc') and cfg_pickle.recreate:
            pkl.dump(df_fine, open(cfg_plot.df_fine.split('.')[0]+f"_{cfg_pickle.load_name.split('_')[-1]}.Mu2E.Fit.p", "wb"), pkl.HIGHEST_PROTOCOL)

        # Fitting to variant DSCylFMSAll map, comparing against nominal DSCylFine (ex. fitting to PINN-subtracted systematic)
        # TODO make this more inclusive
        elif 'Unc' in cfg_data.path:
            unc = [p for p in cfg_data.path.split('_') if 'Unc' in p][0]
            pkl.dump(df_fine, open(cfg_plot.df_fine.split('.')[0]+f"_{unc}.Mu2E.Fit.p", "wb"), pkl.HIGHEST_PROTOCOL)

        # In all other cases, name of DSCylFine fit should match DSCylFine map?
        else:
            pkl.dump(df_fine, open(cfg_plot.df_fine.split('.')[0]+".Mu2E.Fit.p", "wb"), pkl.HIGHEST_PROTOCOL)

    else:
        df_fine = None
    if cfg_plot.plot_type != 'none':
        make_fit_plots(ff.input_data, cfg_data, cfg_geom, cfg_plot, name, aspect=aspect, parallel=parallel_plots, df_fine=df_fine)
        
    return hall_measure_data, ff

def custom_eval_unc(model, params, **kwargs):
    # Get variable params
    var_names = []
    for p in params.valuesdict().keys():
        if params[p].vary:
            var_names.append(p)
    nvarys = len(var_names)
    
    # ensure fjac and df2 are correct size if independent var updated by kwargs
    ndata = model.eval(params,**kwargs).size
    fjac = np.zeros((nvarys, ndata))
    df2 = np.zeros(ndata)
    if any(p.stderr is None for p in params.values()):
        print('Missing stderr for {p}, covariance is 0')
        return df2

    # find derivative by hand!
    pars = params.copy()
    for i in range(nvarys):
        pname = var_names[i]
        val0 = pars[pname].value
        dval = pars[pname].stderr/3.0

        pars[pname].value = val0 + dval
        res1 = model.eval(pars, **kwargs)

        pars[pname].value = val0 - dval
        res2 = model.eval(pars, **kwargs)

        pars[pname].value = val0
        fjac[i] = (res1 - res2) * 1.5 

    for i in range(nvarys):
        for j in range(nvarys):
            if i == j:
                df2 += fjac[i]*fjac[j]
            else:
                df2 += fjac[i]*fjac[j]*params[var_names[i]].correl[var_names[j]]

    prob = erf(1.0/np.sqrt(2))
    return np.sqrt(df2) * t.ppf((prob+1)/2.0, ndata-nvarys)

if __name__ == "__main__":
    pi = np.pi
    data_maker1 = DataFrameMaker('../datafiles/FieldMapData_1760_v5/Mu2e_DSmap', use_pickle=True)
    r_steps = [25, 225, 425, 625, 800]
    phi_steps = [(i/8.0)*np.pi for i in range(-7, 9)]
    z_steps = list(range(5021, 13021, 50))
    hpg = HallProbeGenerator(data_maker1.data_frame,
                             z_steps=z_steps, r_steps=r_steps, phi_steps=phi_steps)
