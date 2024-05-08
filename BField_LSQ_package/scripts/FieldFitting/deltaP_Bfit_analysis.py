import re
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

from mu2e import mu2e_ext_path
from emtracks.particle import trajectory_solver
from emtracks.mapinterp import get_df_interp_func

# ESTRIDE = 10 # testing (3 cm)
ESTRIDE = 1 # real (3 mm)
step=100 # 100 default


def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# analyze a given track given Bfield func
def analyze_particle_momentum(particle_num, Bfunc, name, outdir):
    # load track (pickle)
    fname = outdir+name+f'.{particle_num:03d}.pkl.nom.pkl'
    e = trajectory_solver.from_pickle(fname)
    # analyze
    e.B_func = Bfunc
    e.mom_LHelix = None
    e.analyze_trajectory_LHelix(step=step, stride=1)
    return e.mom_LHelix
    # mom = e.mom_LHelix
    # return mom

# driving function
def run_analysis(Bfunc, outname, name="run_04", outdir="/home/ckampa/data/pickles/distortions/linear_gradient/run_04/", N_lim=None):
    print("Start of Fit Residual Track Analysis")
    print("--------------------------------------------")
    print(f"Directory: {outdir},\nFilename: {name}")
    start = time.time()
    # df_run = pd.read_pickle(outdir+'MC_sample_plus_reco.pkl')
    df_run = pd.read_pickle(outdir+'MC_sample.pkl')
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    base = outdir+name
    particle_nums = [int(f[f.index('.')+1:f.index('.')+1+3]) for f in sorted(os.listdir(outdir)) if "nom" in f]
    if N_lim is not None:
        particle_nums = particle_nums[:int(N_lim)]
        df_run = df_run.iloc[:int(N_lim)].copy()
        df_run.reset_index(drop=True, inplace=True)
    N = len(df_run)
    # run through each track file in parallel
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(analyze_particle_momentum)(num, Bfunc, name=name, outdir=outdir) for num in tqdm(particle_nums, file=sys.stdout, desc='particle #'))
    # reformat output tuple and save to dataframe with MC info
    moms = reco_tuples
    df_run.loc[:, 'mom_reco'] = moms
    df_run.eval('deltaP = p_mc - mom_reco', inplace=True)
    df_run.to_pickle(outname)
    # save dataframe to pickle
    df_run.to_pickle(outname)
    # print runtime info
    stop = time.time()
    print("Calculations Complete")
    print(f"Runtime: {stop-start} s, {(stop-start)/60.} min, {(stop-start)/60./60.} hr")
    print(f"Speed: {(stop-start) / (N)} s / trajectory")
    return

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--map_name', help='Bmap filename (mu2e_ext_path/Bmaps/ already included)')
    parser.add_argument('-p', '--pkl_dir', help='Output dataframe pickle location (mu2e_ext_path/pickles/Bfit_CE_reco/ already included)')
    parser.add_argument('-f', '--use_fit', help='Use fit values? If "yes" (default), uses columns ["Bx_fit", "By_fit", "Bz_fit"], else uses columns ["Bx", "By", "Bz"].')
    args = parser.parse_args()
    # fill defaults where needed
    if args.map_name is None:
        args.map_name = mu2e_ext_path+'Bmaps/Mu2e_DSMap_V13.p'
    else:
        args.map_name = mu2e_ext_path+'Bmaps/'+args.map_name
    if args.pkl_dir is None:
        args.pkl_dir = mu2e_ext_path+'pickles/Bfit_CE_reco/default/'
    else:
        args.pkl_dir = mu2e_ext_path+'pickles/Bfit_CE_reco/'+args.pkl_dir+'/'
    if args.use_fit is None:
        args.use_fit = True
    else:
        args.use_fit = args.use_fit == 'yes'
    # short Bfield name
    index = [i.start() for i in re.finditer('/', args.map_name)][-1]
    Bshort = args.map_name[index+1:-9]
    outname = args.pkl_dir + Bshort + '_deltaP.p'
    # check pickle output directory and create if doesn't exist
    check_dir(args.pkl_dir)
    # get df func
    print(f'Using Bmap File: {args.map_name}')
    if args.use_fit:
        Blabels = ['Bx_fit', 'By_fit', 'Bz_fit']
    else:
        Blabels = ['Bx', 'By', 'Bz']
    Bfunc = get_df_interp_func(filename=args.map_name, gauss=False, Blabels=Blabels)
    # run analysis
    # N_lim = 64 # testing
    N_lim = None
    run_analysis(Bfunc, outname, name="run_04", outdir="/home/ckampa/data/pickles/distortions/linear_gradient/run_04/", N_lim=N_lim)
