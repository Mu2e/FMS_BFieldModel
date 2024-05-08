import sys
import os
import argparse
from mu2e import mu2e_ext_path

RUN_SCRIPT = '/home/ckampa/coding/Mu2E/scripts/FieldFitting/deltaP_Bfit_analysis.py'

def run_fit_script(m_flag, p_flag):
    os.system(f'python {RUN_SCRIPT} -m {m_flag} -p {p_flag}')

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-md', '--mapdir', help='Input/Output Bfield map directory to parse through (from mu2e_ext_path/Bmaps/)')
    args = parser.parse_args()
    # fill defaults where needed
    if args.mapdir is None:
        args.mapdir = 'ensemble_random_scale_factor/'
    else:
        args.mapdir = args.mapdir+'/'
    # pick pickle dir to match map dir
    p_dir = args.mapdir[:-1]

    files = os.listdir(mu2e_ext_path+'Bmaps/'+args.mapdir)
    files_run = []
    for f in files:
        if f[-8:] == 'fit_df.p':
            files_run.append(f)

    for f in files_run:
        run_fit_script(args.mapdir+f, p_dir)
