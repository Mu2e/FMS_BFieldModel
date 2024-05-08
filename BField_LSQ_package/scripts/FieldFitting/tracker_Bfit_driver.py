import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

RUN_SCRIPT = '/home/ckampa/coding/Mu2E/scripts/FieldFitting/tracker_Bfit.py'

def run_fit_script(p_flag, s_flag, t_flag='no'):
    os.system(f'python {RUN_SCRIPT} -p {p_flag} -s {s_flag} -t {t_flag}')

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', '--paramname', help='Fit parameter loadname (from base param directory)')
    parser.add_argument('-md', '--mapdir', help='Output Bfield map directory (from mu2e_ext_path)')
    parser.add_argument('-r', '--run_num', help='Which Run number?')
    parser.add_argument('-i', '--experiment_i', help='First experiment number (default "000")')
    parser.add_argument('-f', '--experiment_f', help='Last experiment number')

    args = parser.parse_args()
    # fill defaults where needed
    if args.paramname is None:
        # args.paramname = 'Mau13'
        args.paramname = 'ensemble_random_scale_factor/Mau13'
    if args.mapdir is None:
        args.mapdir = 'Bmaps/ensemble_random_scale_factor/'
    if args.run_num is None:
        args.run_num = '000'
    else:
        args.run_num = f'{int(args.run_num):03d}'
    if args.experiment_i is None:
        args.experiment_i = '000'
    else:
        args.experiment_i = f'{int(args.experiment_i):03d}'
    if args.experiment_f is None:
        args.experiment_f = '000'
    else:
        args.experiment_f = f'{int(args.experiment_f):03d}'

    param_name_list = [f'{args.paramname}_run{args.run_num}_exp{experiment:03d}' for experiment in np.arange(int(args.experiment_i), int(args.experiment_f)+1)]

    output_name_list = [f'{args.mapdir}Mau13_run{args.run_num}_exp{experiment:03d}_fit_df.p' for experiment in np.arange(int(args.experiment_i), int(args.experiment_f)+1)]

    # for i, names in tqdm(enumerate(zip(param_name_list, output_name_list)), total=len(param_name_list), desc=f"Experiment \#"):
    #     param_name, output_name = names
    #     os.system(f'python {RUN_SCRIPT} -p {param_name} -s {output_name} -t no')

    # num_cpu = multiprocessing.cpu_count()
    num_cpu = 17 # artificial limit to prevent memory overload

    Parallel(n_jobs=num_cpu)(delayed(run_fit_script)(p_flag=names[0], s_flag=names[1]) for names in tqdm(zip(param_name_list, output_name_list), file=sys.stdout, desc='experiment #', total=len(param_name_list)))
