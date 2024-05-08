import sys
import numpy as np
import pandas as pd
from helicalc.coil import CoilIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from tqdm import tqdm

# output info
output_dir = '/home/ckampa/data/pickles/helicalc/testing/'
# save_name = 'Helicalc_v00_xz_plane.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine_helicity.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine_helicity_noadj_32bit.pkl'
# save_name = 'Helicalc_v00_xz_plane_fine_helicity_noadj.pkl'
## coil only map
#save_name = 'Helicalc_v00_xz_plane_fine_helicity_helonly.pkl'
save_name = 'Helicalc_v01_xz_plane_coarse_helicity_helonly.pkl'

# "truth" (Mau13-(PS+TS)) dataframe
# df_true = pd.read_pickle('/home/shared_data/Bmaps/Mau13/subtracted/Mau13_1.00xDS_0.00xPS-TS_DSMap.p')
df_true = pd.read_pickle('/home/ckampa/data/Bmaps/Mau13/DSMap_helical_windings_only.p')
# create grid of test points
## QUICK
# xs = [0.]
# ys = [0.]
# zs = df_true.Z.unique()[::200]
#######
xs = [-0.8, -0.4, 0., 0.4, 0.8]
ys = [0.]
zs = df_true.Z.unique()[::25]
X, Y, Z = np.meshgrid(xs,ys,zs,indexing='ij')
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# load Mau13 coils
# geom_df = read_solenoid_geom_combined('../dev/params/', 'DS_V13_adjusted')
geom_df = read_solenoid_geom_combined('../../dev/params/', 'DS_V13')

# would be much faster to loop through field points first, then coils
def Mu2e_DS_B(x=0., y=0., z=10.571):
    x -= 3.904
    Bs = []
    names = []
    for i in range(len(geom_df)):
        geom_coil = geom_df.iloc[i]
        for j in range(1, int(geom_coil.N_layers+1)):
            # coarse
            # CoilIG = CoilIntegrator(geom_coil, dxyz=np.array([2e-3,2e-3, 2e-3/geom_coil.Ri]), layer=j)
            # fine
            #CoilIG = CoilIntegrator(geom_coil, dxyz=np.array([1e-3,1e-3, 1e-3/geom_coil.Ri]), layer=j, dev=3)
            # coarse (3-14-22)
            CoilIG = CoilIntegrator(geom_coil, dxyz=np.array([2e-3,1e-3, 5e-3/geom_coil.Ri]), layer=j, dev=3)
            Bs.append(CoilIG.integrate(x0=x, y0=y, z0=z))
            names.append(f'Coil_{geom_coil.Coil_Num}_Layer_{j}')
    Bs = np.array(Bs)
    return np.sum(Bs, axis=0)# , Bs, names

if __name__ == '__main__':
    # loop through points
    Bs = []
    B_trues = []
    for x,y,z in tqdm(zip(X,Y,Z), total=len(X), file=sys.stdout, desc='Field Point'):
        Bs.append(Mu2e_DS_B(x=x, y=y, z=z))
        B_trues.append(df_true.query(f'X=={x} & Y=={y} & Z=={z}')[['Bx','By','Bz']].values[0])

    Bs = 1e4*np.array(Bs) # scale to Gauss for comparison
    B_trues = np.array(B_trues)

    df_run = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z, 'Bx_Mau13':B_trues[:,0],
                           'By_Mau13':B_trues[:,1], 'Bz_Mau13':B_trues[:,2],
                           'Bx':Bs[:,0], 'By':Bs[:,1], 'Bz':Bs[:,2]})
    df_run.eval('B = (Bx**2+By**2+Bz**2)**(1/2)', inplace=True)
    df_run.eval('B_Mau13 = (Bx_Mau13**2+By_Mau13**2+Bz_Mau13**2)**(1/2)', inplace=True)
    df_run.eval('dBx = Bx-Bx_Mau13', inplace=True)
    df_run.eval('dBy = By-By_Mau13', inplace=True)
    df_run.eval('dBz = Bz-Bz_Mau13', inplace=True)
    df_run.eval('dB = B-B_Mau13', inplace=True)

    df_run.to_pickle(output_dir+save_name)
