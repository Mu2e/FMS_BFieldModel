'''
Check for data directory and necessary PSOff file.
In the future, we may consider running the PSOff calculation during
package installation, assuming only a small number of points are needed.
'''
import os
# import subprocess

def run_cfgs():
    # check if conda config exists
    base_dir = os.path.dirname(__file__)+'/../'
    print('Base source directory: '+base_dir)
    # check for data symlink
    print('Checking for "data" directory in base directory...')
    if not 'data' in os.listdir(base_dir):
        raise OSError('Symbolic link to data directory not found. Please create this link and install this package again.')
    else:
        print('Found "data/"')
    # check directory structure
    print('Checking for "data/Bmaps/" directory in base directory...')
    if not 'Bmaps' in os.listdir(base_dir+'data/'):
        # raise OSError('data/Bmaps/ does not exist. Creating subdirectory now...')
        print('data/Bmaps/ does not exist. Creating subdirectory now...')
        os.mkdir(base_dir+'data/Bmaps/')
        print('Subdirectory successfully created!')
    else:
        print('Found "data/Bmaps/"')
    # check for PSOff directory
    print('Checking for "data/Bmaps/aux/" directory in base directory...')
    if not 'aux' in os.listdir(base_dir+'data/Bmaps'):
        # raise OSError('data/Bmaps/ does not exist. Creating subdirectory now...')
        print('data/Bmaps/aux/ does not exist. Creating subdirectory now...')
        os.mkdir(base_dir+'data/Bmaps/aux/')
        print('Subdirectory successfully created!')
    else:
        print('Found "data/Bmaps/aux/"')
    # check for PSOff file
    print('Checking for "PSOff" file for SolCalc_GUI...')
    if not 'Mau13.SolCalc.PS_region.standard.PSoff.pkl' in os.listdir(base_dir+'data/Bmaps/aux/'):
        raise OSError('Could not find "PSOff" file:\n'+
                      '"data/Bmaps/aux/Mau13.SolCalc.PS_region.standard.PSoff.pkl"\n'+
                      'Please copy the file and install package again.')
    else:
        print('Found "data/Bmaps/aux/Mau13.SolCalc.PS_region.standard.PSoff.pkl"')

if __name__=='__main__':
    run_cfgs()
