import os
import time
import argparse
import pandas as pd
import pickle as pkl
from scipy.interpolate import griddata

from mu2e import mu2e_ext_path
from hallprobesim_redux import *
from mu2e.fieldfitter_redux2 import FieldFitter

# for use in emtracks LHelix momentum analysis

def trackerfit(param_load_name='Mau13', savename='Bmaps/Mau13_standard_tracker_fit_df.p', tracker=True):
    starttime = time.time()
    # print('Loading Validation Grid Points')
    query = "(X <= 0.8) & (X >= -0.8) & (Y <= 0.8) & (Y >= -0.8)"
    if tracker:
        query += "& (Z >= 8.25) & (Z <= 11.75)"
    df_Mu2e = pd.read_pickle(mu2e_ext_path+'Bmaps/Mu2e_DSMap_V13.p')
    df_Mu2e = df_Mu2e.query(query)
    # grab X=Y=0 values for later
    df_00 = df_Mu2e.query('(X == 0) & (Y ==0)')
    df_Mu2e = df_Mu2e.query('(R >= 25e-3)')
    # df_Mu2e = df_Mu2e.query("(R >= 25e-3) & (R <= 0.8) & (Z >= 8.25) & (Z <= 11.75)")
    # df_Mu2e = df_Mu2e.query("(R >= 25e-3) & (R <= 0.8) & (Z >= 4.2) & (Z <= 13.9)")
    # df_Mu2e = df_Mu2e.query("(R >= 25e-3)")
    # if sample is not None:
    #     df_Mu2e = df_Mu2e.sample(sample)

    cfg_pickle_Mau13_recreate = cfg_pickle(use_pickle=True, save_pickle=False, load_name=param_load_name,save_name=param_load_name, recreate=True)

    # print('Initializing FieldFitter')
    ff = FieldFitter(df_Mu2e)
    # print('Running Fit')
    ff.fit(cfg_params_DS_Mau13, cfg_pickle_Mau13_recreate)
    # print('Merging data fit residuals')
    ff.merge_data_fit_res()
    # print('Saving to pickle')
    ff.input_data.to_pickle(mu2e_ext_path+savename)
    # print('Further manipulations')
    df = pd.read_pickle(mu2e_ext_path+savename)
    # print('Interpolating X=Y=0')
    df2 = df.query('(X<=.05) & (X>=-.05) & (Y<=.05) &(Y>=-.05)')
    zs = df2.Z.unique()
    xs = np.zeros_like(zs)
    ys = np.zeros_like(zs)
    test_grid = np.array([xs,ys,zs]).T
    Br_fit = griddata(df2[['X','Y','Z']].values, df2['Br_fit'], test_grid, method='linear')
    Bphi_fit = griddata(df2[['X','Y','Z']].values, df2['Bphi_fit'], test_grid, method='linear')
    Bz_fit = griddata(df2[['X','Y','Z']].values, df2['Bz_fit'], test_grid, method='linear')
    df_00['Br_fit'] = Br_fit
    df_00['Bz_fit'] = Bz_fit
    df_00['Bphi_fit'] = Bphi_fit
    df = pd.concat([df, df_00], ignore_index=True)
    # resort
    df.sort_values(by=['X','Y','Z'])
    # print('Calculating Bx, By')
    df.eval('Bx_fit = Br_fit * cos(Phi) - Bphi_fit*sin(Phi)', inplace=True)
    df.eval('By_fit = Br_fit * sin(Phi) + Bphi_fit*cos(Phi)', inplace=True)
    df.to_pickle(mu2e_ext_path+savename)
    # print(f'Completed in {(time.time()-starttime):.1f} seconds')


if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_name', help='Parameter load name (from Mu2E package)')
    parser.add_argument('-s', '--save_name', help='Output dataframe save name (mu2e_ext_path already included)')
    parser.add_argument('-t', '--tracker', help='Include tracker query (yes[default]/no)')
    # parser.add_argument('-N', '--num_sample', help='Number of samples to calculate for (default no limit)')
    args = parser.parse_args()
    # fill defaults where needed
    if args.param_name is None:
        args.param_name = 'Mau13'
    if args.save_name is None:
        args.save_name = 'Bmaps/Mau13_standard_tracker_fit_df.p'
    if args.tracker is None:
        args.tracker = True
    else:
        if args.tracker == 'yes':
            args.tracker = True
        else:
            args.tracker = False
    trackerfit(args.param_name, args.save_name, args.tracker)
