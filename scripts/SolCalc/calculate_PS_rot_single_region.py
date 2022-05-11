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
    ProtonDumpArea_grid
)

# paramdir = '/home/ckampa/coding/helicalc/dev/params/'
paramdir = helicalc_dir + 'dev/params/'
# old version
#paramname = 'Mu2e_PS3_rot_16mrad'
#paramname = 'Mu2e_PS23_rot_16mrad'

# all 3

# individual rotations
# paramname = 'Mu2e_PS123_rot_16mrad'
# datadir = helicalc_data+'Bmaps/SolCalc_partial/PS3_rot_16mrad/'
# base_coils = 'PS3_16mrad'

# rotate coldmass
# 16mrad
# paramname = 'Mu2e_PS_coldmass_rot_16mrad'
# datadir = helicalc_data+'Bmaps/SolCalc_partial/PS_coldmass_rot_16mrad/'
# base_coils = 'PS_coldmass_16mrad'
# 23mrad
# paramname = 'Mu2e_PS_coldmass_rot_23mrad'
# datadir = helicalc_data+'Bmaps/SolCalc_partial/PS_coldmass_rot_23mrad/'
# base_coils = 'PS_coldmass_23mrad'
# 7mrad
paramname = 'Mu2e_PS_coldmass_rot_7mrad'
datadir = helicalc_data+'Bmaps/SolCalc_partial/PS_coldmass_rot_7mrad/'
base_coils = 'PS_coldmass_7mrad'

regions = {'PS': PS_grid, 'TSu': TSu_grid, 'TSd': TSd_grid, 'DS': DS_grid,
           'PStoDumpArea': PStoDumpArea_grid,
           'ProtonDumpArea': ProtonDumpArea_grid}

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["PS"(default), "TSu", "TSd", "DS", "PStoDumpArea"'+
                        ', "ProtonDumpArea"]')
    parser.add_argument('-c', '--Coils',
                        help='Which coils to calculate? '+
                        '["1,2,3" (default), "1,2", "1,3", "2,3", 1", "2", "3"]')
    # parser.add_argument('-t', '--Testing',
    #                     help='Calculate using small subset of coils?'+
    #                     '"y"(default)/"n"')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Region is None:
        args.Region = 'PS'
    else:
        args.Region = args.Region.strip()
    if args.Coils is None:
        args.Coils = '1,2,3'
    else:
        args.Coils = args.Coils.strip()
    # if args.Testing is None:
    #     args.Testing = 'n'
    # else:
    #     args.Testing = args.Testing.strip()
    reg = args.Region
    coils = [int(i) for i in args.Coils.split(',')]
    # print configs
    print(f'Region: {reg}')
    # print(f'Testing on subset of coils? {args.Testing}\n')
    # redirect stdout to log file
    dt = datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M%S')
    log_file = open(datadir+f"logs/{dt}_calculate_{reg}_region.log", "w")
    old_stdout = sys.stdout
    sys.stdout = log_file
    # print configs in file
    # print(f'Region: {reg}')
    # print(f'Testing on subset of coils? {args.Testing}\n')
    # step size for integrator
    drz = np.array([5e-3, 1e-2])
    # create grid
    df = generate_cartesian_grid_df(regions[reg])
    # define base save name
    base_name = f'Mau13.SolCalc.{reg}_region.{base_coils}'
    # load geometry
    geom_df_mu2e = read_solenoid_geom_combined(paramdir,paramname)
    # which coils
    #if args.Coils != '2-3':
    #    geom_df_mu2e = geom_df_mu2e.query(f'Coil_Num == {args.Coils}')
    geom_df_mu2e = geom_df_mu2e[np.isin(geom_df_mu2e.Coil_Num, coils)]
    print(f'{len(geom_df_mu2e)} Coils to Calculate', file=old_stdout)
    # TESTING (only a few coils)
    # if args.Testing == 'y':
    #     geom_df_mu2e = geom_df_mu2e.iloc[5:8]
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
    # i0 = int(round(geom_df_mu2e.iloc[0].Coil_Num))
    # i1 = int(round(geom_df_mu2e.iloc[-1].Coil_Num))
    # mySolCalc.save_grid_calc(savetype='pkl',
    #                      savename=datadir+base_name+f'.coils_{i0}-{i1}',
    #                      all_solcalc_cols=True)
    # close log file
    log_file.close()
