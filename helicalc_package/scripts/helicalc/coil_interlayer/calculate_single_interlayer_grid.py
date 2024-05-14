import sys
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from helicalc import helicalc_dir, helicalc_data
from helicalc.busbar import ArcIntegrator3D
from helicalc.tools import (
    generate_cartesian_grid_df,
    generate_cylindrical_grid_df,
    add_points_for_J
)
from helicalc.constants import dxyz_arc_bar_dict, TSd_grid, DS_grid, DS_FMS_cyl_grid, DS_FMS_cyl_grid_SP, DS_cyl_grid_fine, DSCartVal_grid
from helicalc.solenoid_geom_funcs import load_all_geoms

# data
datadir = helicalc_data+'Bmaps/helicalc_partial/'

# load straight bus bars, dump all other geometries
paramname = 'Mu2e_V13'
# paramname = 'Mu2e_V13_altDS11'
version = paramname.replace('Mu2e_V', '')
df_dict = load_all_geoms(version=version, return_dict=True)
df_interlayer = df_dict['interlayers']

# assume same chunk size for everything, for now
N_per_chunk = 10000

regions = {'TSd': TSd_grid, 'DS': DS_grid, 'DSCylFMS': DS_FMS_cyl_grid,
           'DSCylFMSAll': [DS_FMS_cyl_grid, DS_FMS_cyl_grid_SP], 'DSCylFine': DS_cyl_grid_fine, 'DSCartVal': DSCartVal_grid}

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["DS"(default), "TSd", "DSCylFMS", "DSCylFMSAll", "DSCylFine"]')
    parser.add_argument('-C', '--Coil',
                        help='Coil number [56-62, 66], default '+
                        'is 56 (DS-1 interlayer connect).')
    parser.add_argument('-D', '--Device',
                        help='Which GPU to use? [0 (default), 1, 2, 3].')
    parser.add_argument('-j', '--Jacobian',
                        help='Include points for calculating '+
                        'the Jacobian of the field? "n"(default)/"y"')
    parser.add_argument('-d', '--dxyz_Jacobian',
                        help='What step size (in m) to use for points used in '+
                        'the Jacobian calculation? e.g. "0.001" (default)')
    parser.add_argument('-t', '--Testing',
                        help='Calculate using small subset of field points '+
                        ' (N=100000)? "y"/"n"(default).')
    parser.add_argument('-i', '--infile', help='pickle file with coordinate grid')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Region is None:
        args.Region = 'DS'
    else:
        args.Region = args.Region.strip()
    reg = args.Region
    if args.Coil is None:
        args.Coil = 1
    else:
        args.Coil = int(args.Coil.strip())
    df_cond = df_interlayer.query(f'`cond N`=={args.Coil}').iloc[0]
    # kludge to add "_il" to "cond N"
    #(to not confuse with other arc segment column names)
    df_cond['cond N'] = f'{int(df_cond["cond N"])}_il'

    R = df_cond.R0
    # pick correct integration grid based on which SC cross section
    if df_cond['T'] < 7e-3:
        ind_dxyz = 2
    else:
        ind_dxyz = 1
    # grab integration grid and adjust for R
    dxyz = dxyz_arc_bar_dict[ind_dxyz]
    dxyz[2] = dxyz[2] / R
    # pick correct GPU
    if args.Device is None:
        args.Device = 0
    else:
        args.Device = int(args.Device.strip())
    if args.Jacobian is None:
        args.Jacobian = False
    else:
        if args.Jacobian.strip() == 'y':
            args.Jacobian = True
        else:
            args.Jacobian = False
    if args.dxyz_Jacobian is None:
        args.dxyz_Jacobian = 0.001
    else:
        args.dxyz_Jacobian = float(args.dxyz_Jacobian)
    if args.Testing is None:
        args.Testing = False
    else:
        args.Testing = args.Testing.strip() == 'y'
    # set up base directory/name
    # if args.Testing:
    #     base_name = f'Bmaps/helicalc_partial/tests/{paramname}.{reg}_region.'+\
    #                  'test-helicalc.'
    # else:
    #     base_name = f'Bmaps/helicalc_partial/{paramname}.{reg}_region.'+\
    #                  'standard-helicalc.'
    # print configs
    print(f'Region: {reg}')
    # redirect stdout to log file
    dt = datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M%S')
    log_file = open(datadir+f"logs/{dt}_calculate_{reg}_"+
                    "region_interlayer_connect.log", "w")
    old_stdout = sys.stdout
    sys.stdout = log_file
    # find correct chunk size
    N_calc = N_per_chunk
    # create grid
    if reg in ['DSCylFMS', 'DSCylFMSAll', 'DSCylFine']:
        df = generate_cylindrical_grid_df(regions[reg], dec_round=9)
    elif 'Unc' in reg:
        if args.infile is None:
            print("For Unc type region, provide pickle with shifted measurement grid")
            exit()
        df = pd.read_pickle(args.infile)
        df = df[['X','Y','Z','HP']]
    else:
        df = generate_cartesian_grid_df(regions[reg])
    if args.Testing:
        df = df.iloc[:100000].copy()
    # add extra points for Jacobian?
    if args.Jacobian:
        df = add_points_for_J(df, dxyz=args.dxyz_Jacobian)
        suff = '_Jacobian'
    else:
        suff = ''
    if args.Testing:
        base_name = f'Bmaps/helicalc_partial/tests/{paramname}.{reg}_region.'+\
                     f'test-helicalc{suff}.'
    else:
        base_name = f'Bmaps/helicalc_partial/{paramname}.{reg}_region.'+\
                     f'standard-helicalc{suff}.'
    # initialize conductor
    myArc = ArcIntegrator3D(df_cond, dxyz=dxyz, dev=args.Device)
    # integrate!
    myArc.integrate_grid(df, N_batch=N_calc, tqdm=tqdm)
    # save!
    myArc.save_grid_calc(savetype='pkl', savename=base_name+
                         f'coil_{args.Coil}_interlayer',
                         all_cols=False)
