import subprocess
#import pandas as pd
#from tqdm import tqdm
#from helicalc import helicalc_dir, helicalc_data
#from helicalc.coil import CoilIntegrator
#from helicalc.geometry import read_solenoid_geom_combined
#from helicalc.tools import generate_cartesian_grid_df
from helicalc.constants import dxyz_dict, TSd_grid, DS_grid, helicalc_GPU_dict

# data
#datadir = helicalc_data+'Bmaps/helicalc_partial/'

# load coils
#paramdir = helicalc_dir + 'dev/params/'
#paramname = 'Mu2e_V13'

#geom_df = read_solenoid_geom_combined(paramdir,paramname).iloc[55:].copy()
# load chunk data
#chunk_file = helicalc_data+'Bmaps/aux/batch_N_helicalc_03-16-22.txt'
#df_chunks = pd.read_csv(chunk_file)

#regions = {'TSd': TSd_grid, 'DS': DS_grid,}
#reg = 'DS'

if __name__=='__main__':
    #df = generate_cartesian_grid_df(DS_grid)
    #df_ = df.iloc[:1000]
    # loop through all coil layers
    for Dev in helicalc_GPU_dict:
        print(f'Running on GPU: {Dev}')
        for info in helicalc_GPU_dict[Dev]:
            print(f'Calculating: {info}')
            _ = subprocess.run(f'python calculate_single_coil_grid.py -r DS -C {info["coil"]}'+
                               f' -L {info["layer"]} -D {Dev} -t y', shell=True,
                               capture_output=False)


            # df_coil = geom_df.query(f'Coil_Num=={info["coil"]}').iloc[0]
            # N_calc = df_chunks.query(f'Nt_Ri == {df_coil.Nt_Ri}').iloc[0].N_field_points
            # myCoil = CoilIntegrator(df_coil, dxyz=dxyz_dict[df_coil.dxyz],
            #                         layer=info['layer'], dev=Dev,
            #                         interlayer_connect=True)
            # myCoil.integrate_grid(df_, N_batch=N_calc, tqdm=tqdm)
            # myCoil.save_grid_calc(savetype='pkl', savename=f'Bmaps/helicalc_partial/tests/'+
            #               f'Mau13.{reg}_region.test-helicalc.'+
            #               f'coil_{info["coil"]}_layer_{info["layer"]}',
            #               all_helicalc_cols=False)
