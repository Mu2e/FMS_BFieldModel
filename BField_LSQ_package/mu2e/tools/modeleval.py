#! /usr/bin/env python

import math
import numpy as np
import pandas as pd
from mu2e import mu2e_ext_path

# would like something like this...put configs in mu2e not scripts?
# `import hallprobesim_redux.py`
# instead do the hacky (and unsafe) exec command
exec(open(mu2e_ext_path+"../Mu2E/scripts/FieldFitting/hallprobesim_redux.py").read())

# "fast" is only ~0.02 ms faster right now...may be fine to get rid of
def get_mag_field_function(param_name,units=('m','G'),fastcart=False,input_ff=None):
    # first, recreate model if none is input
    if type(input_ff) == type(None):
        config_pickle_Opt = cfg_pickle(use_pickle=True, save_pickle=False,
                                       load_name=param_name, save_name=param_name, recreate=True)
        hmd,ff = field_map_analysis('fma_ds_cyl_test', cfg_data_DS_Mau13,
                                    cfg_geom_cyl_800mm_long, cfg_params_DS_Mau13,
                                    config_pickle_Opt, cfg_plot_none)

    # # clean things up
    # dels = ["cfg_","z_steps","path_","pi2","pi4","pi8","piall","r_steps","phi_steps","hmd"]
    # for name in dir():
    #     if any([n in name for n in dels]):
    #         del globals()[name]
    # del dels

    if fastcart:
        def mag_calc_func(x,y,z,cart=True):
            '''
            Returns:
            single bfield tuple in Tesla or Gauss in cartesian coords.
            '''
            x = np.array([x])
            y = np.array([y])
            z = np.array([z])

            r = np.sqrt(x**2+y**2)
            phi = np.arctan2(y,x)

            if units[0] == 'mm':
                r = r/1000.
                z = z/1000.

            if units[1] == 'T':
                br,bz,bphi = np.nan_to_num(ff.result.eval(r=r,phi=phi,z=z))/1e4
            else:
                br,bz,bphi = np.nan_to_num(ff.result.eval(r=r,phi=phi,z=z))

            bx = br*math.cos(phi)-bphi*math.sin(phi)
            by = br*math.sin(phi)+bphi*math.cos(phi)

            return (bx,by,bz)
    else:
        def mag_calc_func(a,b,z,cart=True):
            '''
            Returns:
            cart==True: (bx,by,bz)
            cart==False: (br,bphi,bz)
            output: bfield in Tesla or Gauss
            '''
            if type(a) == pd.core.series.Series:
                a = a.values
                b = b.values
                z = z.values
            if (type(a) != np.array):
                a = np.array([a]).flatten()
                b = np.array([b]).flatten()
                z = np.array([z]).flatten()

            if cart:
                r = np.sqrt(a**2+b**2)
                phi = np.arctan2(b,a)
            else:
                r = a
                phi = b

            if units[0] == 'mm':
                r = r/1000.
                z = z/1000.

            br,bz,bphi = (ff.result.eval(r=r,phi=phi,z=z)).reshape((3,-1))
            br = np.nan_to_num(br)
            bz = np.nan_to_num(bz)
            bphi = np.nan_to_num(bphi)
            if units[1] == "T":
                br = br/1e4
                bphi = bphi/1e4
                bz = bz/1e4
            if cart:
                bx = br*np.cos(phi)-bphi*np.sin(phi)
                by = br*np.sin(phi)+bphi*np.cos(phi)
                try:
                    bx = float(bx)
                    by = float(by)
                    bz = float(bz)
                except:
                    pass
                return (bx,by,bz)
            else:
                try:
                    br = float(br)
                    bphi = float(bphi)
                    bz = float(bz)
                except:
                    pass
                return (br,bphi,bz)

    return mag_calc_func
