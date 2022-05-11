import sys
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from helicalc import helicalc_dir, helicalc_data
from helicalc.busbar import StraightIntegrator3D
from helicalc.tools import generate_cartesian_grid_df
from helicalc.constants import dxyz_straight_bar_dict, TSd_grid, DS_grid
from helicalc.solenoid_geom_funcs import load_all_geoms

# data
datadir = helicalc_data+'Bmaps/helicalc_partial/'

# load straight bus bars, dump all other geometries
df_dict = load_all_geoms(return_dict=True)
df_str = df_dict['straights']

# assume same chunk size for everything, for now
N_per_chunk = 10000

regions = {'TSd': TSd_grid, 'DS': DS_grid,}

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["DS"(default), "TSd"]')
    parser.add_argument('-C', '--Conductor',
                        help='Conductor number [12, 25-67], default is 12 '+
                        '(from Gap DS7-8 to over DS-2).')
    parser.add_argument('-D', '--Device',
                        help='Which GPU to use? [0 (default), 1, 2, 3].')
    parser.add_argument('-t', '--Testing',
                        help='Calculate using small subset of field points '+
                        ' (N=100000)? "y"/"n"(default).')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Region is None:
        args.Region = 'DS'
    else:
        args.Region = args.Region.strip()
    reg = args.Region
    if args.Conductor is None:
        args.Conductor = 12
    else:
        args.Conductor = int(args.Conductor.strip())
    df_cond = df_str.query(f'`cond N`=={args.Conductor}').iloc[0]
    # pick correct integration grid based on which SC cross section
    if df_cond['T'] < 7e-3:
        ind_dxyz = 2
    else:
        ind_dxyz = 1
    if args.Device is None:
        args.Device = 0
    else:
        args.Device = int(args.Device.strip())
    if args.Testing is None:
        args.Testing = False
    else:
        args.Testing = args.Testing.strip() == 'y'
    # set up base directory/name
    if args.Testing:
        base_name = f'Bmaps/helicalc_partial/tests/Mau13.{reg}_region.'+\
                     'test-busbar.'
    else:
        base_name = f'Bmaps/helicalc_partial/Mau13.{reg}_region.'+\
                     'standard-busbar.'
    # print configs
    print(f'Region: {reg}')
    # redirect stdout to log file
    dt = datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M%S')
    log_file = open(datadir+f"logs/{dt}_calculate_{reg}_"+
                    "region_busbar_straight.log", "w")
    old_stdout = sys.stdout
    sys.stdout = log_file
    # find correct chunk size
    N_calc = N_per_chunk
    # set up grid
    df = generate_cartesian_grid_df(regions[reg])
    if args.Testing:
        df = df.iloc[:100000].copy()
    # initialize conductor
    myStraight = StraightIntegrator3D(df_cond,
                                      dxyz=dxyz_straight_bar_dict[ind_dxyz],
                                      dev=args.Device)
    # integrate!
    myStraight.integrate_grid(df, N_batch=N_calc, tqdm=tqdm)
    # save!
    myStraight.save_grid_calc(savetype='pkl', savename=base_name+
                              f'cond_N_{args.Conductor}_straight',
                              all_cols=False)
