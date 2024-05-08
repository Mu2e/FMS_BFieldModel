import os
import time
import argparse
import pandas as pd
import pickle as pkl

from mu2e import mu2e_ext_path
from hallprobesim_redux import *
from mu2e.fieldfitter_redux2 import FieldFitter

def validatefit(param_load_name='Mau13', sample=None, savename='Bmaps/Mau13_standard_fit_df.p'):
    starttime = time.time()
    print('Loading Validation Grid Points')
    df_Mu2e = pd.read_pickle(mu2e_ext_path+'Bmaps/Mu2e_DSMap_V13.p')
    df_Mu2e = df_Mu2e.query("(R >= 25e-3) & (R <= 0.8) & (Z >= 4.2) & (Z <= 13.9)")
    # df_Mu2e = df_Mu2e.query("(R >= 25e-3)")
    if sample is not None:
        df_Mu2e = df_Mu2e.sample(sample)

    cfg_pickle_Mau13_recreate = cfg_pickle(use_pickle=True, save_pickle=False, load_name=param_load_name,save_name=param_load_name, recreate=True)

    print('Initializing FieldFitter')
    ff = FieldFitter(df_Mu2e)
    print('Running Fit')
    ff.fit(cfg_params_DS_Mau13, cfg_pickle_Mau13_recreate)
    print('Merging data fit residuals')
    ff.merge_data_fit_res()
    print('Saving to pickle')
    ff.input_data.to_pickle(mu2e_ext_path+savename)
    print(f'Completed in {(time.time()-starttime):.1f} seconds')


if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_name', help='Parameter load name (from Mu2E package)')
    parser.add_argument('-s', '--save_name', help='Output dataframe save name (mu2e_ext_path already included)')
    parser.add_argument('-N', '--num_sample', help='Number of samples to calculate for (default no limit)')
    args = parser.parse_args()
    # fill defaults where needed
    if args.param_name is None:
        args.param_name = 'Mau13'
    if args.save_name is None:
        args.save_name = 'Bmaps/Mau13_standard_fit_df.p'
    if not args.num_sample is None:
        args.num_sample = int(args.num_sample)
    # run validation
    validatefit(args.param_name, args.num_sample, args.save_name)
