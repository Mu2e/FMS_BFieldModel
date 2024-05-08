import subprocess
import argparse
from helicalc.constants import dxyz_dict, TSd_grid, DS_grid, helicalc_GPU_dict

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--Region',
                        help='Which region of Mu2e to calculate? '+
                        '["DS"(default), "TSd", "DSCylFMS", "DSCylFMSAll"]')
    parser.add_argument('-D', '--Device',
                        help='Which GPU (i.e. which coils/layers) to use? [0 (default), 1, 2, 3].')
    parser.add_argument('-j', '--Jacobian',
                        help='Include points for calculating '+
                        'the Jacobian of the field? "n"(default)/"y"')
    parser.add_argument('-d', '--dxyz_Jacobian',
                        help='What step size (in m) to use for points used in '+
                        'the Jacobian calculation? e.g. "0.001" (default)')
    parser.add_argument('-t', '--Testing',
                        help='Calculate using small subset of field points (N=1000)?'+
                        '"y"/"n"(default). If yes (y), region defaults to DS.')
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
    # if Test == 'y':
    #     reg = 'DS'

    print(f'Running on GPU: {Dev}')
    for info in helicalc_GPU_dict[Dev]:
        print(f'Calculating: {info}')
        _ = subprocess.run(f'python calculate_single_coil_grid.py -r {reg} -C {info["coil"]}'+
                           f' -L {info["layer"]} -D {Dev} -j {Jac} -d {dxyz} -t {Test}', shell=True,
                           capture_output=False)
