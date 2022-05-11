import sys
from time import time
from datetime import datetime
import argparse
import numpy as np
from helicalc import helicalc_dir, helicalc_data
from helicalc.solcalc import *
from helicalc.geometry import read_solenoid_geom_combined
from helicalc.tools import generate_cartesian_grid_df
from helicalc.constants import (
    PS_grid,
    TSu_grid,
    TSd_grid,
    DS_grid,
    PStoDumpArea_grid,
    ProtonDumpArea_grid,
    DS_cyl2d_grid_5mm
)

# paramdir = '/home/ckampa/coding/helicalc/dev/params/'
paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13'
datadir = helicalc_data+'Bmaps/SolCalc_partial/'

regions = {'PS': PS_grid, 'TSu': TSu_grid, 'TSd': TSd_grid, 'DS': DS_grid,
           'PStoDumpArea': PStoDumpArea_grid,
           'ProtonDumpArea': ProtonDumpArea_grid,
           'DSCyl2D': DS_cyl2d_grid_5mm}

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["PS"(default), "TSu", "TSd", "DS", "PStoDumpArea"'+
                        ', "ProtonDumpArea", "DSCyl2D"]')
    parser.add_argument('-t', '--Testing',
                        help='Calculate using small subset of coils?'+
                        '"y"(default)/"n"')
    parser.add_argument('-u', '--Unused',
                        help='Unused argument.')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Region is None:
        args.Region = 'PS'
    else:
        args.Region = args.Region.strip()
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
    df = generate_cartesian_grid_df(regions[reg])
    # define base save name
    base_name = f'Mau13.SolCalc.{reg}_region.standard'
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
