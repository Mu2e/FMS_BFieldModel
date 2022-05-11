import subprocess
#import argparse
import numpy as np
import pandas as pd
from helicalc import helicalc_dir, helicalc_data
from helicalc.geometry import read_solenoid_geom_combined

paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13'

if __name__=='__main__':
    # load DS coils
    df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
    #df.reset_index(drop=True, inplace=True)
    print(df)
    # loop through df, only doing unique Nt_Ri
    done = []
    for row in df.itertuples():
        if not row.Nt_Ri in done:
            print(row)
            if len(done) < 1:
                append = 'n'
            else:
                append = 'y'
            _ = subprocess.run(f'python find_max_N.py -C {row.Coil_Num}'+
                               f' -L 1 -dxyz {row.dxyz} -D 3'+
                               ' -f Bmaps/aux/batch_N_helicalc_03-16-22.txt'+
                               f' -A {append}', shell=True,
                               capture_output=False)
            done.append(row.Nt_Ri)
