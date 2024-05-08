import sys
import numpy as np
import pandas as pd
from helicalc.coil import CoilIntegrator
from helicalc.geometry import read_solenoid_geom_combined
from tqdm import tqdm

# output info
output_dir = '/home/ckampa/data/pickles/helicalc/testing/'
save_name = 'Loop_Current_Test_Stream.pkl'

# Opera with SOLENOID geometry dataframe
df_sol = pd.read_pickle('/home/ckampa/data/Bmaps/loop_current/solenoid_field_map.pkl')
# Opera with BR20 brick geometry dataframe
df_br = pd.read_pickle('/home/ckampa/data/Bmaps/loop_current/brick_field_map.pkl')

# get small area near conductor
x0 = 1.06055
R = 0.1 # 0.9
z0 = 0.
L = 0.1
map0 = (df_sol.X >= x0-R) & (df_sol.X <= x0+R) & (df_sol.Z >= z0-L) & (df_sol.Z <= z0+L) & (df_sol.Y == 0.)
# query brick and solenoid into calculation df
# stride = 4 # 16 # 8 # 4
# map0 = np.isin(df_sol.X, df_sol.X.unique()[::stride]) & np.isin(df_sol.Z, df_sol.Z.unique()[::stride]) & (df_sol.Y == 0.)

# results df, start with SOLENOID df
df = df_sol[map0].copy()
df.sort_values(by=['X', 'Y', 'Z'], inplace=True)
df.reset_index(drop=True, inplace=True)
# add BR20 calculations
df['Bx_br'] = df_br[map0]['Bx'].values
df['By_br'] = df_br[map0]['By'].values
df['Bz_br'] = df_br[map0]['Bz'].values

# create grid of test points
## QUICK
# xs = [0.]
# ys = [0.]
# zs = df_true.Z.unique()[::200]
#######
'''
xs = [-0.8, -0.4, 0., 0.4, 0.8]
ys = [0.]
zs = df_true.Z.unique()[::25]
X, Y, Z = np.meshgrid(xs,ys,zs,indexing='ij')
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
'''

# load Mau13 coils
# geom_df = read_solenoid_geom_combined('../dev/params/', 'DS_V13_adjusted')
geom_df = read_solenoid_geom_combined('/home/ckampa/coding/helicalc/dev/params/', 'current_loop')
geom_df['pitch_bar'] = 0 # pitch is zero for a ring
geom_df['z'] = -0.0002 # fix slight offset
geom_coil = geom_df.iloc[0]

# CoilIG = CoilIntegrator(geom_coil, dxyz=np.array([1e-3, 1e-3, 1e-3/geom_coil.Ri]), layer=1)
CoilIG = CoilIntegrator(geom_coil, dxyz=np.array([5e-4, 5e-4, 1e-4/geom_coil.Ri]), layer=1)
B_calcs = []
for row in tqdm(df.itertuples(), total=len(df)):
    B_calcs.append(CoilIG.integrate(row.X, row.Y, row.Z))

B_calcs = np.array(B_calcs)
df['Bx_calc'] = B_calcs[:,0]
df['By_calc'] = B_calcs[:,1]
df['Bz_calc'] = B_calcs[:,2]

# calculate |B|
df.eval('B = (Bx**2 + By**2 + Bz**2)**(1/2)', inplace=True)
df.eval('B_br = (Bx_br**2 + By_br**2 + Bz_br**2)**(1/2)', inplace=True)
df.eval('B_calc = (Bx_calc**2 + By_calc**2 + Bz_calc**2)**(1/2)', inplace=True)
# calculate residuals
# "dBx_BR_SOL" means Bx_br - Bx_sol, for example
# SOL - BR
df.eval('dBx_SOL_BR = Bx - Bx_br', inplace=True)
df.eval('dBy_SOL_BR = By - By_br', inplace=True)
df.eval('dBz_SOL_BR = Bz - Bz_br', inplace=True)
df.eval('dB_SOL_BR = B - B_br', inplace=True)
# HEL - SOL
df.eval('dBx_HEL_SOL = Bx_calc - Bx', inplace=True)
df.eval('dBy_HEL_SOL = By_calc - By', inplace=True)
df.eval('dBz_HEL_SOL = Bz_calc - Bz', inplace=True)
df.eval('dB_HEL_SOL = B_calc - B', inplace=True)
# HEL - BR
df.eval('dBx_HEL_BR = Bx_calc - Bx_br', inplace=True)
df.eval('dBy_HEL_BR = By_calc - By_br', inplace=True)
df.eval('dBz_HEL_BR = Bz_calc - Bz_br', inplace=True)
df.eval('dB_HEL_BR = B_calc - B_br', inplace=True)

df.to_pickle(output_dir+save_name)

'''
def Mu2e_DS_B(x=0., y=0., z=10.571):
    x -= 3.904
    Bs = []
    names = []
    for i in range(len(geom_df)):
        geom_coil = geom_df.iloc[i]
        for j in range(1, int(geom_coil.N_layers+1)):
            # CoilIG = CoilIntegrator(geom_coil, dxyz=np.array([2e-3,2e-3, 2e-3/geom_coil.Ri]), layer=j)
            CoilIG = CoilIntegrator(geom_coil, dxyz=np.array([1e-3,1e-3, 1e-3/geom_coil.Ri]), layer=j)
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
'''
