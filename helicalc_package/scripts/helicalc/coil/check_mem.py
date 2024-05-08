import sys
import argparse
import numpy as np
from helicalc import helicalc_dir, helicalc_data
from helicalc.coil import CoilIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.tools import generate_cartesian_grid_df
from helicalc.constants import DS_grid, dxyz_dict

paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13'

geom_df_mu2e = read_solenoid_geom_combined(paramdir,paramname)

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--Coil',
                        help='Coil number [56-66], default is 56 (DS-1).')
    parser.add_argument('-L', '--Layer',
                        help='Coil layer [1 (default), 2,...]. Do not enter layer number > total number of layers in that coil.')
    parser.add_argument('-N', '--Number',
                        help='Number of field points (in hundreds), default is 10=100.')
    parser.add_argument('-dxyz', '--dxyz',
                        help='Which formula to use for step size: [1 (default), 2]. 1 is for multilayer DS coils, 2 is for single layser DS coils.')
    parser.add_argument('-D', '--Device',
                        help='Which GPU to use? [0 (default), 1, 2, 3].')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Coil is None:
        args.Coil = 56
    else:
        args.Coil = int(args.Coil.strip())
    if args.Layer is None:
        args.Layer = 1
    else:
        args.Layer = int(args.Layer.strip())
    df_coil = geom_df_mu2e.query(f'Coil_Num=={args.Coil}').iloc[0]
    N_layers = df_coil.N_layers
    if args.Layer > N_layers:
        raise ValueError(f'(Layer={args.Layer}) > (N_layers={N_layers}). Please enter a valid layer.')
    if args.Number is None:
        N = 100
    else:
        N = 10 * int(args.Number.strip())
    # load N test points
    df = generate_cartesian_grid_df(DS_grid).iloc[:N]
    if args.dxyz is None:
        i = 0
    else:
        i = int(args.dxyz.strip())
    dxyz = dxyz_dict[i]
    if args.Device is None:
        args.Device = 0
    else:
        args.Device = int(args.Device.strip())
    # print configs
    # print(f'Coil: {args.Coil}, Layer: {args.Layer}')
    # print(f'Number of test points: {N}')
    # print(f'Step size config: {i}, dxyz = {dxyz}')
    # load coil and try to integrate
    try:
        myCoil = CoilIntegrator(df_coil, dxyz=dxyz, layer=args.Layer, dev=args.Device)
        Bx, By, Bz = myCoil.integrate_vec_v2(x0_vec=df.X, y0_vec=df.Y, z0_vec=df.Z)
        # print(f'Success: Bx[:10]={Bx[:10]}, By[:10]={By[:10]}, Bz[:10]={Bz[:10]}')
    except:
        # print('Not enough memory!')
        sys.exit(1)

    # if through try, then exit with 0
    sys.exit(0)
