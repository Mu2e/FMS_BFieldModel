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


def get_residuals_interp_func(mapfile):
    df_fit = pd.read_pickle(mapfile)
    # calculate |B| and all residuals
    df_fit.eval('B = (Bx**2+By**2+Bz**2)**(1/2)', inplace=True)
    df_fit.eval('B_fit = (Bx_fit**2+By_fit**2+Bz_fit**2)**(1/2)', inplace=True)
    df_fit.eval('B_res = B - B_fit', inplace=True)
    df_fit.eval('Bx_res = Bx - Bx_fit', inplace=True)
    df_fit.eval('By_res = By - By_fit', inplace=True)
    df_fit.eval('Bz_res = Bz - Bz_fit', inplace=True)
    # setup interp functions
    xyz_res_func = get_df_interp_func(df=df_fit, Blabels=['Bx_res','By_res','Bz_res'])
    mag_res_func = get_df_interp_func(df=df_fit, Blabels=['B_res','By_res','Bz_res'])
    return xyz_res_func, mag_res_func

def track_residual(particle_num, xyz_res_func, mag_res_func, Rrange=None, name='run_04', outdir='/home/ckampa/data/pickles/distortions/linear_gradient/run_04/'):
    # tracker radius range is about 40cm < r < 70cm
    query = 'z >= 8.41 & z <= 11.66' # tracker query
    fname = outdir+name+f'.{particle_num:03d}.pkl.nom.pkl'
    e = trajectory_solver.from_pickle(fname)
    if not Rrange is None:
        query += f'& r >= {Rrange[0]} & r <= {Rrange[1]}'
        e.dataframe.eval('r = (x**2+y**2)**(1/2)', inplace=True)
    xyz_track = e.dataframe.query(query)[['x','y','z']].values[::ESTRIDE]
    xyz_res_track = np.array([xyz_res_func(xyz) for xyz in xyz_track])
    mag_res_track = np.array([mag_res_func(xyz) for xyz in xyz_track])[:,0].reshape(-1, 1)
    return np.concatenate([xyz_res_track, mag_res_track], axis=1)

# driving function
def run_analysis(xyz_res_func, mag_res_func, outname, Rrange=None, name="run_04", outdir="/home/ckampa/data/pickles/distortions/linear_gradient/run_04/", N_lim=None):
    print("Start of Fit Residual Track Analysis")
    print("--------------------------------------------")
    print(f"Directory: {outdir},\nFilename: {name}")
    start = time.time()
    num_cpu = multiprocessing.cpu_count()
    print(f"CPU Cores: {num_cpu}")
    base = outdir+name
    particle_nums = [int(f[f.index('.')+1:f.index('.')+1+3]) for f in sorted(os.listdir(outdir)) if "nom" in f]
    if N_lim is not None:
        particle_nums = particle_nums[:int(N_lim)]
    N = len(particle_nums)
    # run through each track file in parallel
    reco_tuples = Parallel(n_jobs=num_cpu)(delayed(track_residual)(num, xyz_res_func, mag_res_func, Rrange=Rrange, name=name, outdir=outdir) for num in tqdm(particle_nums, file=sys.stdout, desc='particle #'))
    # split output to each residual piece
    Bx_ress = np.concatenate([i[:,0] for i in reco_tuples])#.flatten()
    By_ress = np.concatenate([i[:,1] for i in reco_tuples])#.flatten()
    Bz_ress = np.concatenate([i[:,2] for i in reco_tuples])#.flatten()
    B_ress = np.concatenate([i[:,3] for i in reco_tuples])#.flatten()
    # print(Bx_ress.shape, By_ress.shape, Bz_ress.shape, B_ress.shape)
    results_dict = {'Bx_res':Bx_ress, 'By_res':By_ress, 'Bz_res':Bz_ress, 'B_res':B_ress}
    # make dataframe and save to pickle
    df_run = pd.DataFrame(results_dict)
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
    args = parser.parse_args()
    # fill defaults where needed
    if args.map_name is None:
        args.map_name = mu2e_ext_path+'Bmaps/Mu2e_DSMap_V13.p'
    else:
        args.map_name = mu2e_ext_path+'Bmaps/'+args.map_name

    # make new file names and create Rcut
    outname = args.map_name[:-2]+'_track_res.p'
    outname_Rcut = args.map_name[:-2]+'_track_res_Rcut.p'
    Rcut = [0.4, 0.7]
    # load residual functions
    xyz_res_func, mag_res_func = get_residuals_interp_func(args.map_name)
    # run analysis with and without Rcut
    # N_lim = 64 # testing
    N_lim = None
    run_analysis(xyz_res_func, mag_res_func, outname, Rrange=None, N_lim=N_lim)
    run_analysis(xyz_res_func, mag_res_func, outname_Rcut, Rrange=Rcut, N_lim=N_lim)

    # print(f"x=y=0, z=9")
    # print(f"xyz_res = {xyz_res_func([0,0,9])}")
    # print(f"mag_res = {mag_res_func([0,0,9])[0]}")
