import sys
from time import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from helicalc import helicalc_dir, helicalc_data
from helicalc.solcalc import *
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.tools import (
    generate_cartesian_grid_df,
    generate_cylindrical_grid_df,
    add_points_for_J
)
from helicalc.constants import (
    PS_grid,
    TSu_grid,
    TSd_grid,
    DS_grid,
    PStoDumpArea_grid,
    ProtonDumpArea_grid,
    DS_cyl2d_grid_5mm,
    DS_FMS_cyl_grid,
    DS_FMS_cyl_grid_SP,
    DS_cyl_grid_fine,
    DSCartVal_grid
)

# paramdir = '/home/ckampa/coding/helicalc/dev/params/'
paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13'
# paramname = 'Mu2e_V13_altDS11'
datadir = helicalc_data+'Bmaps/SolCalc_partial/'

regions = {'PS': PS_grid, 'TSu': TSu_grid, 'TSd': TSd_grid, 'DS': DS_grid,
           'PStoDumpArea': PStoDumpArea_grid,
           'ProtonDumpArea': ProtonDumpArea_grid,
           'DSCyl2D': DS_cyl2d_grid_5mm,
           'DSCylFMS': DS_FMS_cyl_grid,
           'DSCylFMSAll': [DS_FMS_cyl_grid, DS_FMS_cyl_grid_SP],
           'DSCylFine': DS_cyl_grid_fine,
           'DSCartVal': DSCartVal_grid}

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["PS"(default), "TSu", "TSd", "DS", "PStoDumpArea"'+
                        ', "ProtonDumpArea", "DSCyl2D", "DSCylFMS", "DSCylFMSAll", "DSCylFine"]')
    parser.add_argument('-j', '--Jacobian',
                        help='Include points for calculating '+
                        'the Jacobian of the field? "n"(default)/"y"')
    parser.add_argument('-d', '--dxyz_Jacobian',
                        help='What step size (in m) to use for points used in '+
                        'the Jacobian calculation? e.g. "0.001" (default)')
    parser.add_argument('-t', '--Testing',
                        help='Calculate using small subset of coils?'+
                        '"y"(default)/"n"')
    parser.add_argument('-i', '--infile', help='pickle file with coordinate grid')
    parser.add_argument('-u', '--Unused',
                        help='Unused argument.')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Region is None:
        args.Region = 'PS'
    else:
        args.Region = args.Region.strip()
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
        args.Testing = 'n'
    else:
        args.Testing = args.Testing.strip()
    reg = args.Region
    # print configs
    print(f'Region: {reg}')
    print(f'Testing on subset of coils? {args.Testing}\n')
    # redirect stdout to log file
    dt = datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M%S')
    log_file = open(datadir+f"logs/{dt}_calculate_{reg}_region.log", "w")
    old_stdout = sys.stdout
    sys.stdout = log_file
    # print configs in file
    print(f'Region: {reg}')
    print(f'Testing on subset of coils? {args.Testing}\n')
    # step size for integrator
    drz = np.array([5e-3, 1e-2])
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
    # add extra points for Jacobian?
    if args.Jacobian:
        df = add_points_for_J(df, dxyz=args.dxyz_Jacobian)
        suff = '_Jacobian'
    else:
        suff = ''
    # define base save name
    base_name = f'{paramname}.SolCalc.{reg}_region.standard{suff}'
    # load geometry
    geom_df_mu2e = read_solenoid_geom_combined(paramdir,paramname)
    # TESTING (only a few coils)
    if args.Testing == 'y':
        geom_df_mu2e = geom_df_mu2e.iloc[5:8]
    # loop through all coils
    N_coils = len(geom_df_mu2e)
    for i in range(N_coils):
    # for geom in geom_df_mu2e.itertuples():
        j = int(round(geom_df_mu2e.iloc[i].Coil_Num))
        # print coil number to screen for reference
        print(f'Calculating coil {i+1}/'+f'{N_coils}', file=old_stdout)
        # instantiate integrator
        mySolCalc = SolCalcIntegrator(geom_df_mu2e.iloc[i], drz=drz)
        # integrate on grid (and update the grid df)
        df = mySolCalc.integrate_grid(df)
        # save single coil results
        mySolCalc.save_grid_calc(savetype='pkl',
                                 savename=datadir+base_name+f'.coil_{j}',
                                 all_solcalc_cols=False)

    # save df with all coils
    i0 = int(round(geom_df_mu2e.iloc[0].Coil_Num))
    i1 = int(round(geom_df_mu2e.iloc[-1].Coil_Num))
    mySolCalc.save_grid_calc(savetype='pkl',
                         savename=datadir+base_name+f'.coils_{i0}-{i1}',
                         all_solcalc_cols=True)
    # close log file
    log_file.close()
