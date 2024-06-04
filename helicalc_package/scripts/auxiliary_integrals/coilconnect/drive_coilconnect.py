import subprocess
import argparse
import numpy as np
from helicalc.solenoid_geom_funcs import load_all_geoms

# load straight bus bars, dump all other geometries
paramname = 'Mu2e_V13'
version = paramname.replace('Mu2e_V', '')
df_dict = load_all_geoms(version=version, return_dict=True)
df_coilconnect = df_dict['coilconnect']
# FIXME! This line is leftover from radial current (2 per coil). Clean this up.
#unique_coils = np.unique([s.strip('out').strip('in') for s
#                         in df_busbarconnect['cond N'].values])
unique_coils = df_coilconnect['cond N'].values
N_cond = len(unique_coils)
# last GPU will get fewer conductors
# we have 11 scripts to run, so this split is more sensible (3, 3, 3, 2)
N_cond_per_GPU = int(N_cond // 4) + 1

# evenly split, with remaining bars to GPU 3
coilconnect_GPU_dict = {0: range(0, N_cond_per_GPU),
                        1: range(N_cond_per_GPU, 2*N_cond_per_GPU),
                        2: range(2*N_cond_per_GPU, 3*N_cond_per_GPU),
                        3: range(3*N_cond_per_GPU, N_cond)}

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["DS"(default), "TSd", "DSCylFMS", "DSCylFMSAll", "DSCylFine", "DSCartVal"]')
    parser.add_argument('-D', '--Device',
                        help='Which GPU (i.e. which coils/layers) to use? '+
                        '[0 (default), 1, 2, 3].')
    parser.add_argument('-j', '--Jacobian',
                        help='Include points for calculating '+
                        'the Jacobian of the field? "n"(default)/"y"')
    parser.add_argument('-d', '--dxyz_Jacobian',
                        help='What step size (in m) to use for points used in '+
                        'the Jacobian calculation? e.g. "0.001" (default)')
    parser.add_argument('-t', '--Testing',
                        help='Calculate using small subset of field points '+
                        '(N=10000)? "y"/"n"(default).')
    args = parser.parse_args()
    # fill defaults if necessary
    if args.Region is None:
        args.Region = 'DS'
    else:
        args.Region = args.Region.strip()
    reg = args.Region
    if args.Device is None:
        args.Device = 0
    else:
        args.Device = int(args.Device.strip())
    Dev = args.Device
    if args.Jacobian is None:
        args.Jacobian = 'n'
    else:
        args.Jacobian = args.Jacobian.strip()
    Jac = args.Jacobian
    if args.dxyz_Jacobian is None:
        args.dxyz_Jacobian = '0.001'
    else:
        args.dxyz_Jacobian = args.dxyz_Jacobian.strip()
    dxyz = args.dxyz_Jacobian
    if args.Testing is None:
        args.Testing = 'n'
    else:
        args.Testing = args.Testing.strip()
    Test = args.Testing

    print(f'Running on GPU: {Dev}')
    for i in coilconnect_GPU_dict[Dev]:
        cn = unique_coils[i]
        print(f'Calculating {i}: Coil_Num={cn}')
        _ = subprocess.run(f'python calculate_single_coilconnect_grid.py'+
                           f' -r {reg} -C {cn} -D {Dev} -j {Jac} -d {dxyz} -t {Test}',
                           shell=True, capture_output=False)
