from collections import namedtuple

cfg_data   = namedtuple('cfg_data', 'datatype magnet path conditions')
cfg_geom   = namedtuple('cfg_geom', 'geom z_steps r_steps phi_steps x_steps y_steps '
                        'systunc interpolate do2pi do_selection')
cfg_plot   = namedtuple('cfg_plot', 'plot_type zlims save_loc sub_dir df_fine')
cfg_params = namedtuple('cfg_params', 'pitch1 ms_h1 ns_h1 pitch2 ms_h2 ns_h2 '
                        ' length1 ms_c1 ns_c1 length2 ms_c2 ns_c2 '
                        ' ks_dict bs_tuples bs_bounds loss version '
                        ' method ms_asym_max cfg_calc_data noise', defaults=('leastsq', -1, None, None))
cfg_pickle = namedtuple('cfg_pickle', 'use_pickle save_pickle load_name save_name recreate')
