import numpy as np
import pandas as pd
import pickle as pkl
import argparse

def gen_scale_factor_df(run=0, N_experiments=50, sigma=1e-4):
    N_probes = 5
    scale_factors = 1. + np.random.normal(loc=0., scale=sigma, size=(N_experiments, N_probes))
    df = pd.DataFrame(scale_factors, columns=[f'hp_{n}_sf' for n in range(N_probes)])
    df.loc[:, 'run'] = run
    df.loc[:, 'experiment'] = df.index
    df.loc[:,'id'] = [f'run{row.run:03d}_exp{row.experiment:03d}' for row in df.itertuples()]
    return df

# save function (pickle)
def save_df(df, outputfile):
    df.to_pickle(outputfile)

# main function (with arg parsing)
if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', help='Run number (0, 1, ...)')
    parser.add_argument('-N', '--num_exp', help='Number of experiments (def. 50)')
    parser.add_argument('-f', '--file', help='Output pickle file name.')
    args = parser.parse_args()
    # fill defaults where needed
    if args.run is None:
        args.run = 0
    else:
        args.run = int(args.run)
    if args.num_exp is None:
        args.num_exp = 50
    else:
        args.num_exp = int(args.num_exp)
    if args.file is None:
        args.file = f'/home/ckampa/data/pickles/Mu2E/ensemble_random_scale_factor/run{args.run:03d}_random_sfs_init_conds_df.p'

    df = gen_scale_factor_df(run=args.run, N_experiments=args.num_exp, sigma=1e-4)
    save_df(df, args.file)
