#! /usr/bin/env python

from __future__ import absolute_import
from mu2e.hallprober import field_map_analysis
from collections import namedtuple
import numpy as np
import pandas as pd
from mu2e import mu2e_ext_path
from mu2e.src.make_csv import make_csv
from six.moves import range
from sys import argv
import os

############################
# defining the cfg structs #
############################

cfg_data   = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom   = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps x_steps y_steps '
                        'systunc interpolate do2pi do_selection')
cfg_plot   = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir df_fine')
cfg_params = namedtuple('cfg_params', 'pitch1 ms_h1 ns_h1 pitch2 ms_h2 ns_h2 '
                        ' length1 ms_c1 ns_c1 length2 ms_c2 ns_c2 '
                        ' ks_dict bs_tuples bs_bounds loss version '
                        ' method ms_asym_max cfg_calc_data noise', defaults=('leastsq', -1, None, None))
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')

#################
# the data cfgs #
#################

path_DS_Mau13        = mu2e_ext_path+'Bmaps/Mu2e_DSMap_V13'

path_DS_only                = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_SolCalc_DS_Only.Mu2E.p'
path_all_coils              = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_SolCalc_All_Coils.Mu2E.p'
path_DS_helicalc_coils      = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils.Mu2E.p'
path_DS_helicalc_busbar     = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Busbars_Only.Mu2E.p'
path_DS_helicalc_raw        = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars.pkl'
path_DS_helicalc_all        = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars.Mu2E.p'
path_DS_helicalc_pinn       = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_PINN_Subtracted.Mu2E.p'
path_DSCartVal              = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars.Mu2E.p'
path_DSCartVal_pinn         = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars_PINN_Subtracted.Mu2E.p'

path_DS_helicalc_rebar      = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_Rebar.Mu2E.p'
path_DS_helicalc_rebarup    = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_RebarUpUnc.Mu2E.p'
path_DS_helicalc_rebardo    = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_RebarDownUnc.Mu2E.p'
path_DS_rebar_pinn          = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_Rebar_PINN_Subtracted.Mu2E.p'
path_DS_rebarup_pinn        = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_RebarUpUnc_PINN_Subtracted.Mu2E.p'
path_DS_rebardo_pinn        = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_RebarDownUnc_PINN_Subtracted.Mu2E.p'
path_DSCartVal_rebar        = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars_Rebar.Mu2E.p'
path_DSCartVal_rebar_pinn   = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars_Rebar_PINN_Subtracted.Mu2E.p'
path_DSCartVal_rebarup_pinn = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars_RebarUpUnc_PINN_Subtracted.Mu2E.p'
path_DSCartVal_rebardo_pinn = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars_RebarDownUnc_PINN_Subtracted.Mu2E.p'

path_DS_rebar_noise_pinn    = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_Rebar_noise0p3_PINN_Subtracted.Mu2E.p'
path_DSCartVal_rebar_noise_pinn = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars_Rebar_noise0p3_PINN_Subtracted.Mu2E.p'

cfg_data_DS_Mau13           = cfg_data('Mau13',    'DS', path_DS_Mau13,
                                       ('Z>4.200', 'Z<13.900', 'R!=0'))

cfg_data_DS_only            = cfg_data('helicalc', 'DS', path_DS_only,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_all_coils          = cfg_data('helicalc', 'DS', path_all_coils,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_coils  = cfg_data('helicalc', 'DS', path_DS_helicalc_coils,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_busbar = cfg_data('helicalc', 'DS', path_DS_helicalc_busbar,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_posunc = cfg_data('helicalc', 'DS', path_DS_helicalc_raw,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_all    = cfg_data('helicalc', 'DS', path_DS_helicalc_all,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_pinn   = cfg_data('helicalc', 'DS', path_DS_helicalc_pinn,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DSCartVal          = cfg_data('helicalc', 'DS', path_DSCartVal,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DSCartVal_pinn     = cfg_data('helicalc', 'DS', path_DSCartVal_pinn,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_rebar    = cfg_data('helicalc', 'DS', path_DS_helicalc_rebar,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_rebarup  = cfg_data('helicalc', 'DS', path_DS_helicalc_rebarup,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_rebardo  = cfg_data('helicalc', 'DS', path_DS_helicalc_rebardo,
                                       ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_rebar_pinn   = cfg_data('helicalc', 'DS', path_DS_rebar_pinn,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_rebarup_pinn   = cfg_data('helicalc', 'DS', path_DS_rebarup_pinn,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DS_helicalc_rebardo_pinn   = cfg_data('helicalc', 'DS', path_DS_rebardo_pinn,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DSCartVal_rebar          = cfg_data('helicalc', 'DS', path_DSCartVal_rebar,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DSCartVal_rebar_pinn     = cfg_data('helicalc', 'DS', path_DSCartVal_rebar_pinn,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DSCartVal_rebarup_pinn     = cfg_data('helicalc', 'DS', path_DSCartVal_rebarup_pinn,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DSCartVal_rebardo_pinn     = cfg_data('helicalc', 'DS', path_DSCartVal_rebardo_pinn,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DS_Rebar_noise_pinn   = cfg_data('helicalc', 'DS', path_DS_rebar_noise_pinn,
                                             ('Z>4.200', 'Z<13.900'))

cfg_data_DSCartVal_Rebar_noise_pinn     = cfg_data('helicalc', 'DS', path_DSCartVal_rebar_noise_pinn,
                                             ('Z>4.200', 'Z<13.900'))

#################
# the geom cfgs #
#################

pi8r_800mm = [0.05590169944, 0.16770509831, 0.33541019663, 0.55901699437, 0.78262379213]
pi4r_800mm = [0.03535533906, 0.1767766953, 0.35355339059, 0.53033008589, 0.81317279836]
pi2r_800mm = [0.025, 0.175, 0.375, 0.525, 0.800]
r_steps_800mm = (pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm,
                 pi2r_800mm, pi8r_800mm, pi4r_800mm, pi8r_800mm)

phi_steps_8 = (0, 0.463648, np.pi/4, 1.107149, np.pi/2, 2.034444, 3*np.pi/4, 2.677945)
phi_steps_helicalc=(0., 0.39269908, 0.78539816, 1.17809725, 1.57079633, 1.96349541, 2.35619449, 2.74889357)

z_steps_DS_long = [i/1000 for i in range(4221, 13921, 50)]

# Actual configs
cfg_geom_cyl_800mm_long         = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_8,
                                           x_steps=None, y_steps=None,
                                           systunc=None, interpolate=False,
                                           do2pi=False, do_selection=True)

# Note since do_selection is false, we don't actually use the steps to sparsify the field
# Phi steps are still relevant since they set the plotting granularity
cfg_geom_cyl_800mm_long_full       = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_helicalc,
                                              x_steps=None, y_steps=None,
                                              systunc=None, interpolate=False,
                                              do2pi=False, do_selection=False)

cfg_geom_cyl_800mm_long_full_LaserUnc = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_helicalc,
                                              x_steps=None, y_steps=None,
                                              systunc='LaserUnc', interpolate=False,
                                              do2pi=False, do_selection=False)

cfg_geom_cyl_800mm_long_full_MetUnc = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_helicalc,
                                              x_steps=None, y_steps=None,
                                              systunc='MetUnc', interpolate=False,
                                              do2pi=False, do_selection=False)

cfg_geom_cyl_800mm_long_full_CalibMagUnc = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_helicalc,
                                              x_steps=None, y_steps=None,
                                              systunc='CalibMagUnc', interpolate=False,
                                              do2pi=False, do_selection=False)

cfg_geom_cyl_800mm_long_full_CalibRotUnc = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_helicalc,
                                              x_steps=None, y_steps=None,
                                              systunc='CalibRotUnc', interpolate=False,
                                              do2pi=False, do_selection=False)

cfg_geom_cyl_800mm_long_full_TempUnc = cfg_geom('cyl', z_steps_DS_long, r_steps_800mm, phi_steps_helicalc,
                                                x_steps=None, y_steps=None,
                                                systunc='TempUnc', interpolate=False,
                                                do2pi=False, do_selection=False)

###### Cartesian geometry ######
#Full grid
xy_steps_cart = [i/1000 for i in range(-950,   950, 25)] 
z_steps_cart  = [i/1000 for i in range(4225, 13850, 25)]
cfg_geom_cart = cfg_geom('cart', z_steps=z_steps_cart, r_steps=None, phi_steps=None,
                         x_steps=xy_steps_cart, y_steps=xy_steps_cart, systunc=None,
                         interpolate=False, do2pi=False, do_selection=False)

#PINN subtracted
xy_steps_cart_pinn = [i/1000 for i in range(-800,   825, 25)]
z_steps_cart_pinn  = [i/1000 for i in range(4250, 13575, 25)]
cfg_geom_cart_pinn = cfg_geom('cart', z_steps=z_steps_cart_pinn, r_steps=None, phi_steps=None,
                              x_steps=xy_steps_cart_pinn, y_steps=xy_steps_cart_pinn, systunc=None,
                              interpolate=False, do2pi=False, do_selection=False)

#################
# the plot cfgs #
#################

df_fine_path         = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFine_Helicalc_All_Coils_All_Busbars.Mu2E.p'
df_fine_path_rebar   = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFine_Helicalc_All_Coils_All_Busbars_Rebar.Mu2E.p'
df_fine_path_rebarup = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFine_Helicalc_All_Coils_All_Busbars_RebarUpUnc.Mu2E.p'
df_fine_path_rebardo = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFine_Helicalc_All_Coils_All_Busbars_RebarDownUnc.Mu2E.p'
df_fine_path_pinn    = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFine_Helicalc_All_Coils_All_Busbars_PINN_Subtracted.Mu2E.p'
df_fine_path_noise_pinn = mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFine_Helicalc_All_Coils_All_Busbars_Rebar_noise0p3_PINN_Subtracted.Mu2E.p'

cfg_plot_mpl          = cfg_plot('mpl',        [-2, 2], 'html', None, None)
cfg_plot_mpl_nonuni   = cfg_plot('mpl_nonuni', [-2, 2], 'html', None, None)
cfg_plot_mpl_fine     = cfg_plot('mpl_nonuni', [-2, 2], 'html', None, df_fine_path)
cfg_plot_mpl_pinn     = cfg_plot('mpl_nonuni', [-2, 2], 'html', None, df_fine_path_pinn)
cfg_plot_mpl_rebar    = cfg_plot('mpl_nonuni', [-2, 2], 'html', None, df_fine_path_rebar)
cfg_plot_mpl_rebarup  = cfg_plot('mpl_nonuni', [-2, 2], 'html', None, df_fine_path_rebarup)
cfg_plot_mpl_rebardo  = cfg_plot('mpl_nonuni', [-2, 2], 'html', None, df_fine_path_rebardo)
cfg_plot_mpl_noise_pinn = cfg_plot('mpl_nonuni', [-2, 2], 'html', None, df_fine_path_noise_pinn)

cfg_plot_none         = cfg_plot('none', None, None, None, None)

###################
# the params cfgs #
###################

cfg_params_DS_Mau13_only_ks = cfg_params(pitch1=0, ms_h1=0, ns_h1=0,
                                         pitch2=0, ms_h2=0, ns_h2=0,
                                         length1=11.9, ms_c1=1, ns_c1=1,
                                         length2=0, ms_c2=0, ns_c2=0,
                                         ks_dict={'k1': 0,
                                                  'k2': -3,
                                                  'k3': 10000,
                                                  'k4': 0,
                                                  'k5': 0,
                                                  'k6': 0,
                                                  'k7': 0,},
                                         bs_tuples=None,
                                         bs_bounds=None,
                                         loss='linear',
                                         method='leastsq',
                                         version=1000)
#                                         cfg_calc_data=cfg_data_DS_helicalc_busbar)

cfg_params_DS_Mau13_new     = cfg_params(pitch1=0, ms_h1=0, ns_h1=0,
                                         pitch2=0, ms_h2=0, ns_h2=0,
                                         length1=11.3, ms_c1=70, ns_c1=6,
                                         length2=0, ms_c2=0, ns_c2=0,
                                         ks_dict={'k1': 0,
                                                  'k2': -3,
                                                  'k3': 10000,
                                                  'k4': 0,
                                                  'k5': 0,
                                                  'k6': 0,
                                                  'k7': 0,},
                                         bs_tuples=None,
                                         bs_bounds=None,
                                         loss='linear',
                                         method='leastsq',
                                         version=1005,
                                         noise=0.3)
#                                         cfg_calc_data=cfg_data_DS_helicalc_busbar)

###################
# the pickle cfgs #
###################

cfg_pickle_DS_only       = cfg_pickle(use_pickle=False, save_pickle=True,
                                      load_name='DS_only_V7',
                                      save_name='DS_only_V7', recreate=False)

cfg_pickle_DS_only_new   = cfg_pickle(use_pickle=False, save_pickle=True,
                                      load_name='DS_only_V8',
                                      save_name='DS_only_V8', recreate=False)

cfg_pickle_all_coils     = cfg_pickle(use_pickle=False, save_pickle=True,
                                      load_name='all_coils_V1',
                                      save_name='all_coils_V1', recreate=False)

cfg_pickle_all_coils_new = cfg_pickle(use_pickle=False, save_pickle=True,
                                      load_name='all_coils_V3',
                                      save_name='all_coils_V3', recreate=False)

cfg_pickle_Mau13_only_ks     = cfg_pickle(use_pickle=False, save_pickle=True,
                                          load_name='Mau13_only_ks',
                                          save_name='Mau13_only_ks', recreate=False)

cfg_pickle_Mau13_only_ks_rec = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='Mau13_only_ks',
                                          save_name='Mau13_only_ks', recreate=True)

# NOTE: for helicalc fits, we expect 'save_name' to correspond to the 'actual fit'
# e.g. helicalc_LaserUnc means either we did the LaserUnc fit, or loaded the LaserUnc
# params to fit the nominal field

cfg_pickle_helicalc_all      = cfg_pickle(use_pickle=False, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_all', recreate=False)
        
cfg_pickle_helicalc_all_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_all',
                                          save_name='helicalc_all', recreate=True)
        
cfg_pickle_helicalc_all_LaserUnc      = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_LaserUnc', recreate=False)
        
cfg_pickle_helicalc_all_LaserUnc_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_LaserUnc',
                                          save_name='helicalc_LaserUnc', recreate=True)
        
cfg_pickle_helicalc_all_MetUnc      = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_MetUnc', recreate=False)
        
cfg_pickle_helicalc_all_MetUnc_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_MetUnc',
                                          save_name='helicalc_MetUnc', recreate=True)
        
cfg_pickle_helicalc_all_CalibMagUnc      = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_CalibMagUnc', recreate=False)
        
cfg_pickle_helicalc_all_CalibMagUnc_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_CalibMagUnc',
                                          save_name='helicalc_CalibMagUnc', recreate=True)
        
cfg_pickle_helicalc_all_CalibRotUnc      = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_CalibRotUnc', recreate=False)
        
cfg_pickle_helicalc_all_CalibRotUnc_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_CalibRotUnc',
                                          save_name='helicalc_CalibRotUnc', recreate=True)
        
cfg_pickle_helicalc_all_TempUnc          = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_TempUnc', recreate=False)
        
cfg_pickle_helicalc_all_TempUnc_rec      = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_TempUnc',
                                          save_name='helicalc_TempUnc', recreate=True)

cfg_pickle_helicalc_all_Rebar            = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_Rebar', recreate=False)
        
cfg_pickle_helicalc_all_Rebar_rec        = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_Rebar',
                                          save_name='helicalc_Rebar', recreate=True)
        
cfg_pickle_helicalc_all_Rebar_noise      = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_all',
                                          save_name='helicalc_Rebar_noise', recreate=False)
        
cfg_pickle_helicalc_all_Rebar_noise_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_Rebar_noise',
                                          save_name='helicalc_Rebar_noise', recreate=True)
        
cfg_pickle_helicalc_all_RebarUpUnc        = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_Rebar',
                                          save_name='helicalc_RebarUpUnc', recreate=False)
        
cfg_pickle_helicalc_all_RebarUpUnc_rec    = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_RebarUpUnc',
                                          save_name='helicalc_RebarUpUnc', recreate=True)

cfg_pickle_helicalc_all_RebarDoUnc        = cfg_pickle(use_pickle=True, save_pickle=True,
                                          load_name='helicalc_Rebar',
                                          save_name='helicalc_RebarDoUnc', recreate=False)
        
cfg_pickle_helicalc_all_RebarDoUnc_rec    = cfg_pickle(use_pickle=True, save_pickle=False,
                                          load_name='helicalc_RebarDoUnc',
                                          save_name='helicalc_RebarDoUnc', recreate=True)

cfg_pickle_helicalc_pinn  = cfg_pickle(use_pickle=True, save_pickle=True,
                                       load_name='helicalc_pinn',
                                       save_name='helicalc_pinn', recreate=False)

cfg_pickle_helicalc_pinn_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                           load_name='helicalc_pinn',
                                           save_name='helicalc_pinn', recreate=True)

cfg_pickle_helicalc_Rebar_pinn  = cfg_pickle(use_pickle=True, save_pickle=True,
                                             load_name='helicalc_Rebar',
                                             save_name='helicalc_Rebar_pinn', recreate=False)

cfg_pickle_helicalc_Rebar_pinn_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                                 load_name='helicalc_Rebar_pinn',
                                                 save_name='helicalc_Rebar_pinn', recreate=True)

cfg_pickle_helicalc_RebarUpUnc_pinn  = cfg_pickle(use_pickle=True, save_pickle=True,
                                                  load_name='helicalc_RebarUpUnc',
                                                  save_name='helicalc_RebarUpUnc_pinn', recreate=False)

cfg_pickle_helicalc_RebarUpUnc_pinn_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                                      load_name='helicalc_RebarUpUnc_pinn',
                                                      save_name='helicalc_RebarUpUnc_pinn', recreate=True)

cfg_pickle_helicalc_RebarDownUnc_pinn  = cfg_pickle(use_pickle=True, save_pickle=True,
                                                    load_name='helicalc_RebarDoUnc',
                                                    save_name='helicalc_RebarDownUnc_pinn', recreate=False)

cfg_pickle_helicalc_RebarDownUnc_pinn_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                                        load_name='helicalc_RebarDownUnc_pinn',
                                                        save_name='helicalc_RebarDownUnc_pinn', recreate=True)

cfg_pickle_helicalc_Rebar_noise_pinn  = cfg_pickle(use_pickle=True, save_pickle=True,
                                                   load_name='helicalc_Rebar_noise',
                                                   save_name='helicalc_Rebar_noise_pinn', recreate=False)

cfg_pickle_helicalc_Rebar_noise_pinn_rec  = cfg_pickle(use_pickle=True, save_pickle=False,
                                                       load_name='helicalc_Rebar_noise_pinn',
                                                       save_name='helicalc_Rebar_noise_pinn', recreate=True)

#############
# main code #
#############

if __name__ == "__main__":
    if len(argv) != 1:
        
        '''
        # Fitting regular grid
        hmd, ff = field_map_analysis('fma_mau13_only_ks', cfg_data_DS_Mau13,
                                      cfg_geom_cyl_800mm_long, cfg_params_DS_Mau13_only_ks,
                                      cfg_pickle_Mau13_only_ks, cfg_plot_mpl)

        # Plotting regular grid
        hmd, ff = field_map_analysis('fma_mau13_only_ks', cfg_data_DS_Mau13,
                                      cfg_geom_cyl_800mm_long, cfg_params_DS_Mau13_only_ks,
                                      cfg_pickle_Mau13_only_ks_rec, cfg_plot_mpl)

        # Fitting helicalc nominal
        hmd, ff = field_map_analysis('fma_helicalc_all', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all, cfg_plot_mpl_fine)

        # Plotting helicalc nominal
        hmd, ff = field_map_analysis('fma_helicalc_all', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_rec, cfg_plot_mpl_fine)

        # Fitting helicalc LaserUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_LaserUnc', cfg_data_DS_helicalc_posunc,
                                     cfg_geom_cyl_800mm_long_full_LaserUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_LaserUnc, cfg_plot_mpl_nonuni)

        # Plotting helicalc LaserUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_LaserUnc', cfg_data_DS_helicalc_posunc,
                                     cfg_geom_cyl_800mm_long_full_LaserUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_LaserUnc_rec, cfg_plot_mpl_nonuni)

        # Plotting helicalc nominal vs LaserUnc fit
        hmd, ff = field_map_analysis('fma_helicalc_all_LaserUnc_nom', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_LaserUnc_rec, cfg_plot_mpl_fine)

        # Fitting helicalc MetUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_MetUnc', cfg_data_DS_helicalc_posunc,
                                     cfg_geom_cyl_800mm_long_full_MetUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_MetUnc, cfg_plot_mpl_nonuni)

        # Fitting helicalc MetUnc
        for itoy in range(10):
            cfg_pickle_toy = cfg_pickle_helicalc_all_MetUnc._replace(save_name = f'helicalc_MetUnc{itoy}')            
            cfg_geom_MetUnc_toy = cfg_geom_cyl_800mm_long_full_MetUnc._replace(systunc = f'MetUnc{itoy}')
            hmd, ff = field_map_analysis(f'fma_helicalc_all_MetUnc{itoy}', cfg_data_DS_helicalc_posunc,
                                         cfg_geom_MetUnc_toy, cfg_params_DS_Mau13_new,
                                         cfg_pickle_toy, cfg_plot_none)

        # Plotting helicalc MetUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_MetUnc', cfg_data_DS_helicalc_posunc,
                                     cfg_geom_cyl_800mm_long_full_MetUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_MetUnc_rec, cfg_plot_mpl_nonuni)

        # Plotting helicalc nominal vs MetUnc fit
        hmd, ff = field_map_analysis('fma_helicalc_all_MetUnc_nom', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_MetUnc_rec, cfg_plot_mpl_fine)

        # Fitting helicalc CalibMagUnc
        for itoy in range(10):
            cfg_pickle_toy = cfg_pickle_helicalc_all_CalibMagUnc._replace(save_name = f'helicalc_CalibMagUnc{itoy}')            
            cfg_geom_CalibMagUnc_toy = cfg_geom_cyl_800mm_long_full_CalibMagUnc._replace(systunc = f'CalibMagUnc{itoy}')
            hmd, ff = field_map_analysis(f'fma_helicalc_all_CalibMagUnc{itoy}', cfg_data_DS_helicalc_all,
                                         cfg_geom_CalibMagUnc_toy, cfg_params_DS_Mau13_new,
                                         cfg_pickle_toy, cfg_plot_none)

        # Plotting helicalc CalibMagUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_CalibMagUnc', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full_CalibMagUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_CalibMagUnc_rec, cfg_plot_mpl_nonuni)

        # Plotting helicalc nominal vs CalibMagUnc fit
        hmd, ff = field_map_analysis('fma_helicalc_all_CalibMagUnc_nom', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_CalibMagUnc_rec, cfg_plot_mpl_fine)

        # Fitting helicalc CalibRotUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_CalibRotUnc', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full_CalibRotUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_CalibRotUnc, cfg_plot_mpl_nonuni)

        # Fitting helicalc CalibRotUnc
        for itoy in range(10):
            cfg_pickle_toy = cfg_pickle_helicalc_all_CalibRotUnc._replace(save_name = f'helicalc_CalibRotUnc{itoy}')            
            cfg_geom_CalibRotUnc_toy = cfg_geom_cyl_800mm_long_full_CalibRotUnc._replace(systunc = f'CalibRotUnc{itoy}')            
            hmd, ff = field_map_analysis(f'fma_helicalc_all_CalibRotUnc{itoy}', cfg_data_DS_helicalc_all,
                                         cfg_geom_CalibRotUnc_toy, cfg_params_DS_Mau13_new,
                                         cfg_pickle_toy, cfg_plot_none)

        # Plotting helicalc CalibRotUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_CalibRotUnc', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full_CalibRotUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_CalibRotUnc_rec, cfg_plot_mpl_nonuni)

        # Plotting helicalc nominal vs CalibRotUnc fit
        hmd, ff = field_map_analysis('fma_helicalc_all_CalibRotUnc_nom', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_CalibRotUnc_rec, cfg_plot_mpl_fine)

        # Fitting helicalc TempUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_TempUnc', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full_TempUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_TempUnc, cfg_plot_mpl_nonuni)

        # Fitting helicalc TempUnc toys
        for itoy in range(10):
            cfg_pickle_toy = cfg_pickle_helicalc_all_TempUnc._replace(save_name = f'helicalc_TempUnc{itoy}')            
            cfg_geom_TempUnc_toy = cfg_geom_cyl_800mm_long_full_TempUnc._replace(systunc = f'TempUnc{itoy}')            
            hmd, ff = field_map_analysis(f'fma_helicalc_all_TempUnc{itoy}', cfg_data_DS_helicalc_all,
                                         cfg_geom_TempUnc_toy, cfg_params_DS_Mau13_new,
                                         cfg_pickle_toy, cfg_plot_none)

        # Plotting helicalc TempUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_TempUnc', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full_TempUnc, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_TempUnc_rec, cfg_plot_mpl_nonuni)

        # Plotting helicalc nominal vs TempUnc fit
        hmd, ff = field_map_analysis('fma_helicalc_all_TempUnc_nom', cfg_data_DS_helicalc_all,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_TempUnc_rec, cfg_plot_mpl_fine)

        # Fitting helicalc rebar nominal
        hmd, ff = field_map_analysis('fma_helicalc_all_Rebar', cfg_data_DS_helicalc_rebar,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_Rebar, cfg_plot_mpl_rebar)

        # Fitting helicalc rebar nominal w/ noise
        hmd, ff = field_map_analysis('fma_helicalc_all_Rebar_noise_fitmap', cfg_data_DS_helicalc_rebar,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_Rebar_noise_rec, cfg_plot_mpl_nonuni)

        # Fitting helicalc RebarUpUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_RebarUpUnc', cfg_data_DS_helicalc_rebarup,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_RebarUpUnc, cfg_plot_mpl_rebarup)

        # Plotting helicalc Rebar vs RebarUpUnc fit
        hmd, ff = field_map_analysis('fma_helicalc_all_RebarUpUnc_nom', cfg_data_DS_helicalc_rebar,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_RebarUpUnc_rec, cfg_plot_mpl_rebar)

        # Fitting helicalc RebarDoUnc
        hmd, ff = field_map_analysis('fma_helicalc_all_RebarDoUnc', cfg_data_DS_helicalc_rebardo,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_RebarDoUnc, cfg_plot_mpl_rebardo)

        # Plotting helicalc nominal vs RebarDoUnc fit
        hmd, ff = field_map_analysis('fma_helicalc_all_RebarDoUnc_nom', cfg_data_DS_helicalc_rebar,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_RebarDoUnc_rec, cfg_plot_mpl_rebar)

        # Evaluating nominal field on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart', cfg_data_DSCartVal,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_rec, cfg_plot_mpl)

        # Fitting PINN-corrected map on fit grid
        hmd, ff = field_map_analysis('fma_helicalc_pinn', cfg_data_DS_helicalc_pinn,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_pinn, cfg_plot_mpl_pinn)

        # Evaluating PINN-corrected map on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_pinn', cfg_data_DSCartVal_pinn,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_pinn_rec, cfg_plot_mpl)

        # Evaluating nominal field on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_LaserUnc', cfg_data_DSCartVal,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_LaserUnc_rec, cfg_plot_mpl)

        # Evaluating nominal field on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_MetUnc', cfg_data_DSCartVal,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_MetUnc_rec, cfg_plot_mpl)

        # Evaluating nominal field on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_CalibMagUnc', cfg_data_DSCartVal,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_CalibMagUnc_rec, cfg_plot_mpl)

        # Evaluating nominal field on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_CalibRotUnc', cfg_data_DSCartVal,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_CalibRotUnc_rec, cfg_plot_mpl)

        # Evaluating nominal field on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_TempUnc', cfg_data_DSCartVal,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_TempUnc_rec, cfg_plot_mpl)

        # Evaluating rebar fit on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_Rebar_noise', cfg_data_DSCartVal_Rebar,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_Rebar_noise_rec, cfg_plot_none)

        # Evaluating rebarup fit on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_RebarUpUnc', cfg_data_DSCartVal_Rebar,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_RebarUpUnc_rec, cfg_plot_mpl)

        # Evaluating rebardown fit on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_RebarDownUnc', cfg_data_DSCartVal_Rebar,
                                     cfg_geom_cart, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_all_RebarDoUnc_rec, cfg_plot_mpl)

        # Fitting PINN-corrected rebar map on fit grid
        hmd, ff = field_map_analysis('fma_helicalc_Rebar_pinn', cfg_data_DS_helicalc_Rebar_pinn,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_Rebar_pinn, cfg_plot_mpl_nonuni)

        # Fitting PINN-corrected rebarup map on fit grid
        hmd, ff = field_map_analysis('fma_helicalc_RebarUpUnc_pinn', cfg_data_DS_helicalc_rebarup_pinn,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_RebarUpUnc_pinn, cfg_plot_mpl_nonuni)

        # Fitting PINN-corrected rebardo map on fit grid
        hmd, ff = field_map_analysis('fma_helicalc_RebarDownUnc_pinn', cfg_data_DS_helicalc_rebardo_pinn,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_RebarDownUnc_pinn, cfg_plot_mpl_nonuni)

        # Evaluating PINN-corrected rebar map on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_Rebar_pinn', cfg_data_DSCartVal_Rebar_pinn,
                                     cfg_geom_cart_pinn, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_Rebar_pinn_rec, cfg_plot_none)

        # Evaluating PINN-corrected rebarup map on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_RebarUpUnc_pinn', cfg_data_DSCartVal_rebarup_pinn,
                                     cfg_geom_cart_pinn, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_RebarUpUnc_pinn_rec, cfg_plot_none)

        # Evaluating PINN-corrected rebardo map on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_RebarDownUnc_pinn', cfg_data_DSCartVal_rebardo_pinn,
                                     cfg_geom_cart_pinn, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_RebarDownUnc_pinn_rec, cfg_plot_none)

        # Fitting PINN-corrected rebar map w/ noise on fit grid
        hmd, ff = field_map_analysis('fma_helicalc_noise_Rebar_pinn_test', cfg_data_DS_Rebar_noise_pinn,
                                     cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_Rebar_noise_pinn, cfg_plot_none)#cfg_plot_mpl_noise_pinn

        # Evaluating PINN-corrected rebar map w/ noise on cartesian validation grid
        hmd, ff = field_map_analysis('fma_valid_cart_Rebar_noise_pinn', cfg_data_DSCartVal_Rebar_noise_pinn,
                                     cfg_geom_cart_pinn, cfg_params_DS_Mau13_new,
                                     cfg_pickle_helicalc_Rebar_noise_pinn_rec, cfg_plot_none)


        for unc in ['MetUnc','CalibMagUnc','CalibRotUnc','TempUnc']:
            for itoy in range(10):

                path_DS_helicalc_pinn_unc = mu2e_ext_path+f'Bmaps/ensembles/{unc}/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars_{unc}{itoy}_PINN_Subtracted.Mu2E.p'
            
                cfg_data_DS_helicalc_pinn_unc   = cfg_data('helicalc', 'DS', path_DS_helicalc_pinn_unc,
                                                           ('Z>4.200', 'Z<13.900'))
            
                cfg_pickle_helicalc_pinn_unc  = cfg_pickle(use_pickle=True, save_pickle=True,
                                                           load_name='helicalc_pinn',
                                                           save_name=f'helicalc_pinn_{unc}{itoy}', recreate=False)

                # Fitting PINN-corrected map on fit grid
                hmd, ff = field_map_analysis(f'fma_helicalc_pinn_{unc}{itoy}', cfg_data_DS_helicalc_pinn_unc,
                                             cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                             cfg_pickle_helicalc_pinn_unc, cfg_plot_none)

                path_DSCartVal_pinn_unc = mu2e_ext_path+f'Bmaps/ensembles/{unc}/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars_{unc}{itoy}_PINN_Subtracted.Mu2E.p'
            
                cfg_data_DSCartVal_pinn_unc     = cfg_data('helicalc', 'DS', path_DSCartVal_pinn_unc,
                                                           ('Z>4.200', 'Z<13.900'))
            
                cfg_pickle_helicalc_pinn_unc_rec = cfg_pickle(use_pickle=True, save_pickle=False,
                                                              load_name=f'helicalc_pinn_{unc}{itoy}',
                                                              save_name=f'helicalc_pinn_{unc}{itoy}', recreate=True)

                # Evaluating PINN-corrected map on cartesian validation grid
                hmd, ff = field_map_analysis(f'fma_valid_cart_pinn_{unc}{itoy}', cfg_data_DSCartVal_pinn_unc,
                                             cfg_geom_cart_pinn, cfg_params_DS_Mau13_new,
                                             cfg_pickle_helicalc_pinn_unc_rec, cfg_plot_none)

        m_vals = [1,2,3,4,5,6,7,8,9,10,20,30]
        for im in range(len(m_vals)):
            m = m_vals[im]
            m0 = m_vals[im-1]
            
            cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(ms_c1 = m)
            cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(ms_c1 = m)

            bootstrap = False
            if os.path.exists('../data/fit_params/DS_only_m{m0}_results_correl.p'):
                bootstrap = True
                print(f'Using bootstrap from {m0} to initialize {m} for old fit')

            cfg_pickle_DS_only       = cfg_pickle(use_pickle=bootstrap, save_pickle=True,
                                                  load_name=f'DS_only_m{m0}',
                                                  save_name=f'DS_only_m{m}', recreate=False)
            
            # Fitting DS only, old function
            hmd, ff = field_map_analysis(f'fma_DS_only_m{m}', cfg_data_DS_only,
                                         cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_only_ks,
                                         cfg_pickle_DS_only, cfg_plot_mpl_nonuni)

            bootstrap_new = False
            if os.path.exists(f'../data/fit_params/DS_only_m{m0}_new_results_correl.p'):
                bootstrap_new = True
                print(f'Using bootstrap from {m0} to initialize {m} for new fit')

            cfg_pickle_DS_only_new   = cfg_pickle(use_pickle=bootstrap_new, save_pickle=True,
                                                  load_name=f'DS_only_m{m0}_new',
                                                  save_name=f'DS_only_m{m}_new', recreate=False)

            # Fitting DS only, new function
            hmd, ff = field_map_analysis(f'fma_DS_only_m{m}_new', cfg_data_DS_only,
                                         cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                         cfg_pickle_DS_only_new, cfg_plot_mpl_nonuni)


        for n in [1,2,3,4]:
            for m in [10,20,30,40]:

                cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(ms_c1 = m)
                cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(ns_c1 = n)
                cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(ms_c1 = m)
                cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(ns_c1 = n)
                
                cfg_pickle_all_coils       = cfg_pickle(use_pickle=False, save_pickle=True,
                                                        load_name=f'All_coils_m{m}_n{n}',
                                                        save_name=f'All_coils_m{m}_n{n}', recreate=False)
            
                # Fitting DS+TS+PS, old function
                hmd, ff = field_map_analysis(f'fma_all_coils_m{m}_n{n}', cfg_data_all_coils,
                                             cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_only_ks,
                                             cfg_pickle_all_coils, cfg_plot_mpl_nonuni)
                
                cfg_pickle_all_coils_new   = cfg_pickle(use_pickle=False, save_pickle=True,
                                                        load_name=f'All_coils_m{m}_n{n}_new',
                                                        save_name=f'All_coils_m{m}_n{n}_new', recreate=False)

                # Fitting DS+TS+PS, hybrid function
                hmd, ff = field_map_analysis(f'fma_all_coils_m{m}_n{n}_new', cfg_data_all_coils,
                                             cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                             cfg_pickle_all_coils_new, cfg_plot_mpl_nonuni)

        for m in [30,40,50]:
            for n in [3,4,5]:

                cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(ms_c1 = m)
                cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(ns_c1 = n)
                cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(ms_c1 = m)
                cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(ns_c1 = n)

                bootstrap = False
                if os.path.exists(f'../data/fit_params/helicalc_all_m{m}_n{n-1}_results_correl.p'):
                    bootstrap = True
                    print(f'Using bootstrap from m{m}, n{n-1} to initialize m{m}, n{n} for old fit')

                cfg_pickle_helicalc_all      = cfg_pickle(use_pickle=bootstrap, save_pickle=True,
                                                          load_name=f'helicalc_all_m{m}_n{n-1}',
                                                          save_name=f'helicalc_all_m{m}_n{n}', recreate=False)
        
                # Fitting helicalc all, old function
                hmd, ff = field_map_analysis(f'fma_helicalc_all_m{m}_n{n}', cfg_data_DS_helicalc_all,
                                             cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_only_ks,
                                             cfg_pickle_helicalc_all, cfg_plot_mpl_nonuni)

                bootstrap = False
                if os.path.exists(f'../data/fit_params/helicalc_all_m{m}_n{n-1}_new_results_correl.p'):
                    bootstrap = True
                    print(f'Using bootstrap from m{m}, n{n-1} to initialize m{m}, n{n} for new fit')

                cfg_pickle_helicalc_all_new  = cfg_pickle(use_pickle=bootstrap, save_pickle=True,
                                                          load_name=f'helicalc_all_m{m}_n{n-1}_new',
                                                          save_name=f'helicalc_all_m{m}_n{n}_new', recreate=False)
        
                # Fitting helicalc all, new function
                hmd, ff = field_map_analysis(f'fma_helicalc_all_m{m}_n{n}_new', cfg_data_DS_helicalc_all,
                                             cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                             cfg_pickle_helicalc_all_new, cfg_plot_mpl_nonuni)

        for L in [11.3,11.6]:
            for m in [40,50,60]:
                for n in [4,5,6]:
                    
                    cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(ms_c1 = m)
                    cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(ns_c1 = n)
                    cfg_params_DS_Mau13_new     = cfg_params_DS_Mau13_new._replace(length1 = L)
                    
                    #cfg_pickle_helicalc_coils_new  = cfg_pickle(use_pickle=False, save_pickle=True,
                    #                                            load_name=f'helicalc_coils_m{m}_n{n}_L{str(L).replace(".","p")}_new',
                    #                                            save_name=f'helicalc_coils_m{m}_n{n}_L{str(L).replace(".","p")}_new', recreate=False)
        
                    # Fitting helicalc coils, new function
                    #hmd, ff = field_map_analysis(f'fma_helicalc_coils_m{m}_n{n}_L{str(L).replace(".","p")}_new', cfg_data_DS_helicalc_coils,
                    #                             cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                    #                             cfg_pickle_helicalc_coils_new, cfg_plot_mpl_nonuni)

                    bootstrap = False
                    if os.path.exists(f'../data/fit_params/helicalc_all_m{m}_n{n-1}_L{str(L).replace(".","p")}_new_noise_results_correl.p'):
                        bootstrap = True
                        print(f'Using bootstrap from m{m}, n{n-1} to initialize m{m}, n{n} for new fit w/ noise')

                    cfg_pickle_helicalc_all_new  = cfg_pickle(use_pickle=bootstrap, save_pickle=True,
                                                              load_name=f'helicalc_all_m{m}_n{n-1}_L{str(L).replace(".","p")}_new_noise',
                                                              save_name=f'helicalc_all_m{m}_n{n}_L{str(L).replace(".","p")}_new_noise', recreate=False)
        
                    # Fitting helicalc full field, new function
                    hmd, ff = field_map_analysis(f'fma_helicalc_all_m{m}_n{n}_L{str(L).replace(".","p")}_new_noise', cfg_data_DS_helicalc_all,
                                                 cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_new,
                                                 cfg_pickle_helicalc_all_new, cfg_plot_none)

                    #cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(ms_c1 = m)
                    #cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(ns_c1 = n)
                    #cfg_params_DS_Mau13_only_ks = cfg_params_DS_Mau13_only_ks._replace(length1 = L)

                    #bootstrap = False
                    #if os.path.exists(f'../data/fit_params/helicalc_all_m{m}_n{n-1}_L{str(L).replace(".","p")}_results_correl.p'):
                    #    bootstrap = True
                    #    print(f'Using bootstrap from m{m}, n{n-1} to initialize m{m}, n{n} for old fit')

                    #cfg_pickle_helicalc_all  = cfg_pickle(use_pickle=bootstrap, save_pickle=True,
                    #                                      load_name=f'helicalc_all_m{m}_n{n-1}_L{str(L).replace(".","p")}',
                    #                                      save_name=f'helicalc_all_m{m}_n{n}_L{str(L).replace(".","p")}', recreate=False)
        
                    # Fitting helicalc full field, new function
                    #hmd, ff = field_map_analysis(f'fma_helicalc_all_m{m}_n{n}_L{str(L).replace(".","p")}', cfg_data_DS_helicalc_all,
                    #                             cfg_geom_cyl_800mm_long_full, cfg_params_DS_Mau13_only_ks,
                    #                             cfg_pickle_helicalc_all, cfg_plot_mpl_nonuni)
        '''
    else:
        pass
