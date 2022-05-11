import subprocess
import argparse
from helicalc import helicalc_dir, helicalc_data
#from helicalc.coil import CoilIntegrator
from helicalc.geometry import read_solenoid_geom_combined
# from helicalc.tools import generate_cartesian_grid_df
# from helicalc.constants import (
#     PS_grid,
#     TSu_grid,
#     TSd_grid,
#     DS_grid,
#     PStoDumpArea_grid,
#     ProtonDumpArea_grid
# )

paramdir = helicalc_dir + 'dev/params/'
paramname = 'Mu2e_V13'
#datadir = helicalc_data+'Bmaps/aux/helicalc/'

geom_df_mu2e = read_solenoid_geom_combined(paramdir,paramname)

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--Coil',
                        help='Coil number [56-66], default is 56 (DS-1).')
    parser.add_argument('-L', '--Layer',
                        help='Coil layer [1 (default), 2,...]. Do not enter layer number > total number of layers in that coil.')
    parser.add_argument('-dxyz', '--dxyz',
                        help='Which formula to use for step size: [1 (default), 2]. 1 is for multilayer DS coils, 2 is for single layser DS coils.')
    parser.add_argument('-D', '--Device',
                        help='Which GPU to use? [0 (default), 1, 2, 3].')
    parser.add_argument('-f', '--File', help='Where to save results? Starting point is "helicalc_data".')
    parser.add_argument('-A', '--Append', help='Append to file? y/n (default). No (n) overwrites file.')
    args = parser.parse_args()
    # fill defaults where needed
    if args.Coil is None:
        args.Coil = 56
    else:
        args.Coil = int(args.Coil.strip())
    if args.Layer is None:
        args.Layer = 1
    else:
        args.Layer = int(args.Layer.strip())
    if args.dxyz is None:
        args.dxyz = 1
    else:
        args.dxyz = int(args.dxyz.strip())
    if args.Device is None:
        args.Device = 0
    else:
        args.Device = int(args.Device.strip())
    if args.Append is None:
        args.Append = False
    else:
        args.Append = args.Append.strip() == 'y'
    # run until return code is 1
    rcode = 0
    # coarse scan up, steps of 1000
    N = 100
    while(rcode==0):
        compproc = subprocess.run(f'python check_mem.py -C {args.Coil}'+
                                  f' -L {args.Layer} -N {N} -dxyz {args.dxyz}'+
                                  f' -D {args.Device}', shell=True,
                                  capture_output=False)
        rcode = compproc.returncode
        N += 100
    # fine scan down, steps of 500
    while(rcode==1):
        N -= 50
        compproc = subprocess.run(f'python check_mem.py -C {args.Coil}'+
                                  f' -L {args.Layer} -N {N} -dxyz {args.dxyz}'+
                                  f' -D {args.Device}', shell=True,
                                  capture_output=False)
        rcode = compproc.returncode
    # finer scan up, steps of 100
    while(rcode==0):
        N += 10
        compproc = subprocess.run(f'python check_mem.py -C {args.Coil}'+
                                  f' -L {args.Layer} -N {N} -dxyz {args.dxyz}'+
                                  f' -D {args.Device}', shell=True,
                                  capture_output=False)
        rcode = compproc.returncode
    # finest scan down, steps of 10
    while(rcode==1):
        N -= 1
        compproc = subprocess.run(f'python check_mem.py -C {args.Coil}'+
                                  f' -L {args.Layer} -N {N} -dxyz {args.dxyz}'+
                                  f' -D {args.Device}', shell=True,
                                  capture_output=False)
        rcode = compproc.returncode
    # final N is out of range, so N max - 1
    # N -= 1
    df_coil = geom_df_mu2e.query(f'Coil_Num=={args.Coil}').iloc[0]
    #Nt_Ri = f'{df_coil.N_turns * df_coil.Ri:d}'
    Nt_Ri = int(df_coil.N_turns * df_coil.Ri)
    print(f'N_turns * Ri = {Nt_Ri}')
    print(f'Maximum N: {N}, or number of field points = {N*10}')
    # write to file
    if args.File is not None:
        args.File = args.File.strip()
        if args.Append:
            oflag = 'a'
            headerline = ''
        else:
            oflag = 'w'
            #headerline = 'N_turns * Ri, N_field_points\n'
            headerline = 'Nt_Ri,N_field_points\n'
        with open(helicalc_data+args.File, oflag) as ofile:
            ofile.write(headerline)
            ofile.write(f'{Nt_Ri},{N*10}\n')
    # # test -- can we get the value from the other script?
    # i = subprocess.run('python check_mem.py -C 56 -L 1 -N 1 -dxyz 1 -D 0',
    #                    shell=True, capture_output=True)

    # print(f'Finished memory check script, which returned: {i.returncode}')
