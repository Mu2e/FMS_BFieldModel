import sys
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from helicalc import helicalc_dir, helicalc_data
from helicalc.coil import CoilIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.tools import generate_cartesian_grid_df
from helicalc.constants import dxyz_dict, TSd_grid, DS_grid

# data
datadir = helicalc_data+'Bmaps/helicalc_partial/'

# load coils
paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13'

geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
# load chunk data
chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
df_chunks = pd.read_csv(chunk_file)

regions = {'TSd': TSd_grid, 'DS': DS_grid,}

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["DS"(default), "TSd"]')
    parser.add_argument('-C', '--Coil',
                        help='Coil number [56-66], default is 56 (DS-1).')
    parser.add_argument('-L', '--Layer',
                        help='Coil layer [1 (default), 2,...]. Do not enter layer number > total number of layers in that coil.')
    parser.add_argument('-D', '--Device',
                        help='Which GPU to use? [0 (default), 1, 2, 3].')
    parser.add_argument('-t', '--Testing',
                        help='Calculate using small subset of field points (N=1000)?'+
                        '"y"/"n"(default). If yes (y), region defaults to DS.')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Region is None:
        args.Region = 'DS'
    else:
        args.Region = args.Region.strip()
    reg = args.Region
    if args.Coil is None:
        args.Coil = 56
    else:
        args.Coil = int(args.Coil.strip())
    if args.Layer is None:
        args.Layer = 1
    else:
        args.Layer = int(args.Layer.strip())
    df_coil = geom_df.query(f'Coil_Num=={args.Coil}').iloc[0]
    N_layers = df_coil.N_layers
    if args.Layer > N_layers:
        raise ValueError(f'(Layer={args.Layer}) > (N_layers={N_layers}). Please enter a valid layer.')
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
        #reg = 'DS'
        base_name = f'Bmaps/helicalc_partial/tests/Mau13.{reg}_region.test-helicalc.'
    else:
        base_name = f'Bmaps/helicalc_partial/Mau13.{reg}_region.standard-helicalc.'
    # print configs
    print(f'Region: {reg}')
    # redirect stdout to log file
    dt = datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M%S')
    log_file = open(datadir+f"logs/{dt}_calculate_{reg}_region.log", "w")
    old_stdout = sys.stdout
    sys.stdout = log_file
    # find correct chunk size
    N_calc = df_chunks.query(f'Nt_Ri == {df_coil.Nt_Ri}').iloc[0].N_field_points
    # set up grid
    df = generate_cartesian_grid_df(regions[reg])
    if args.Testing:
        df = df.iloc[:1000].copy()
    #df_ = df.iloc[:10000]
    # initialize coil
    myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz],
                            layer=args.Layer, dev=args.Device,
                            interlayer_connect=True)
    # integrate!
    myCoil.integrate_grid(df, N_batch=N_calc, tqdm=tqdm)
    #myCoil.integrate_grid(df_, N_batch=N_calc, tqdm=tqdm)
    # save!
    myCoil.save_grid_calc(savetype='pkl', savename=base_name+
                          f'coil_{args.Coil}_layer_{args.Layer}',
                          all_helicalc_cols=False)
