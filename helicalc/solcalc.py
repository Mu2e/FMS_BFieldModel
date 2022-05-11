from sys import getsizeof
from time import time
import numpy as np
import pandas as pd
import torch as tc
from scipy.special import ellipk, ellipe
#from scipy.integrate import dblquad, tplquad
from scipy.spatial.transform import Rotation
#from tqdm.notebook import tqdm
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

# imports from this library
from helicalc.tools import get_gpu_memory_map
from helicalc.integrate import trapz_2d
from helicalc.constants import MAXMEM, mu0

# move these functions elsewhere?
def k2(r, z, a):
    return (4*a*r) / ((a+r)**2 + z**2)

def Br_Bz_integrands(r, z, a, j):
    # shared elements
    cutoff = 1e-5
    k2_ = k2(r, z, a)
    K = ellipk(k2_)
    E = ellipe(k2_)
    prefac = (mu0 * j) / (2 * np.pi * ((a + r)**2 + z**2)**(1/2))
    denom = ((a-r)**2 + z**2)
    # individual calculations
    if r < cutoff:
        if hasattr(z, '__iter__'):
            Br = np.zeros_like(z)
        else:
            Br = 0
    else:
        Br = prefac * z/r * (-K + (a**2 + r**2 + z**2)/denom * E)
    Bz = prefac * (K + (a**2 - r**2 - z**2)/denom * E)
    return Br, Bz

def Br_Bz_integrands_vec(r, z, a, j):
    # shared elements
    cutoff = 1e-5
    k2_ = k2(r, z, a)
    K = ellipk(k2_)
    E = ellipe(k2_)
    # calculate
    prefac = (mu0 * j) / (2 * np.pi * ((a + r)**2 + z**2)**(1/2))
    denom = ((a-r)**2 + z**2)
    # mask
    m = (r*np.ones_like(K)) < cutoff
    Br = np.zeros_like(K)
    Br[~m] = prefac[~m] * (z+np.zeros_like(K))[~m]/(r+np.zeros_like(K))[~m] * (-K[~m] + (a**2 + r**2 + z**2)[~m]/denom[~m] * E[~m])
    Bz = prefac * (K + (a**2 - r**2 - z**2)/denom * E)
    return Br, Bz


# main integrator class
class SolCalcIntegrator(object):
    def __init__(self, geom_coil, drz, use_basic_geom=True, layer=1, int_func=np.trapz, lib=np, dev=3):
        if lib is tc:
            # set device
            tc.cuda.set_device(dev)
            self.start_dev_mem = get_gpu_memory_map()[dev]
            self.mem_err_expected = False
            raise NotImplementedError('GPU implementation is not completed and will fail.')
        else:
            #print('ERROR! NOT IMPLEMENTED!')
            pass
        self.use_basic_geom = use_basic_geom
        if use_basic_geom:
            self.r0 = geom_coil.Ri
            self.r1 = geom_coil.Ro
            self.z0 = -geom_coil.L/2
            self.z1 = geom_coil.L/2
            self.I_tot = geom_coil.I_turn*geom_coil.N_turns*geom_coil.N_layers
        else:
            # set r0 and r1 based on layer
            self.r0 = geom_coil.rho0_a + (layer-1)*(geom_coil.h_cable + 2*geom_coil.t_ci + geom_coil.t_il)
            self.r1 = self.r0 + geom_coil.h_sc
            dz = geom_coil.t_ci + geom_coil.t_gi + (geom_coil.w_cable - geom_coil.w_sc)/2
            self.z0 = -geom_coil.L/2 + dz
            self.z1 = geom_coil.L/2 - dz
            self.I_tot = geom_coil.I_turn*geom_coil.N_turns
        # calculate currrent density
        self.area = (self.r1-self.r0)*(self.z1-self.z0)
        self.j_tot = self.I_tot/self.area
        # limits of integration
        r_lims = [self.r0, self.r1]
        # z limits (in coil coordinates)
        z_lims = [-geom_coil.L/2, geom_coil.L/2]
        # combine limits
        rz_lims = [r_lims, z_lims]
        # define base R, Z grid to calculate integrands on
        if lib is tc:
            # only once ellipe & ellipk in torch
#             self.rs = tc.linspace(rz_lims[0][0], rz_lims[0][1], abs(int((rz_lims[0][1]-rz_lims[0][0])/drz[0] + 1))).cuda()
#             self.zs = tc.linspace(rz_lims[1][0], rz_lims[1][1], abs(int((rz_lims[1][1]-rz_lims[1][0])/drz[1] + 1))).cuda()
            self.rs = np.linspace(rz_lims[0][0], rz_lims[0][1], abs(int((rz_lims[0][1]-rz_lims[0][0])/drz[0] + 1)))
            self.zs = np.linspace(rz_lims[1][0], rz_lims[1][1], abs(int((rz_lims[1][1]-rz_lims[1][0])/drz[1] + 1)))
        else:
            self.rs = np.linspace(rz_lims[0][0], rz_lims[0][1], abs(int((rz_lims[0][1]-rz_lims[0][0])/drz[0] + 1)))
            self.zs = np.linspace(rz_lims[1][0], rz_lims[1][1], abs(int((rz_lims[1][1]-rz_lims[1][0])/drz[1] + 1)))
        # estimate memory / limit of number of field points to calulate at once
        # FIXME!
        # set R, Z grid
        # only once ellipe & ellipk in torch
        ## self.R, self.Z = lib.meshgrid(self.rs, self.zs, indexing='xy')
        self.R, self.Z = np.meshgrid(self.rs, self.zs, indexing='ij')
        # add extra things to self
        self.geom_coil = geom_coil
        self.int_func = int_func
        self.layer = layer
        self.lib = lib
        self.dev = dev
        # rotations!
        self.XYZ_rot = geom_coil[[f'rot{i:d}' for i in [0,1,2]]].values
        self.XYZ_rot_rad = np.radians(self.XYZ_rot)
        self.mu2e_to_coil = Rotation.from_euler('XYZ', -self.XYZ_rot_rad)
        self.coil_to_mu2e = self.mu2e_to_coil.inv()
        self.xc, self.yc, self.zc = geom_coil[['x','y','z']].values
        # saved results
        self.x_calcs = []
        self.y_calcs = []
        self.z_calcs = []
        # test lists before final rotations
        self.Br_calcs_prerot = []
        self.Bz_calcs_prerot = []
        # lists we actually want (after rotations)
        self.Bx_calcs = []
        self.By_calcs = []
        self.Bz_calcs = []
        # determine actual memory footprint
        sizes = []
        if lib is tc:
            for o in [self.rs, self.zs, self.R, self.Z]:
                try:
                    sizes.append(getsizeof(o.storage())*1e-6)
                except:
                    sizes.append(o.nbytes*1e-6)
        else:
            for o in [self.rs, self.zs, self.R, self.Z]:
                sizes.append(o.nbytes*1e-6)
        self.getsizeof_init_mb = np.array(sizes)

    def integrate_single(self, x0=0, y0=0, z0=0, debug=False):
        # append to lists
        self.x_calcs.append(x0)
        self.y_calcs.append(y0)
        self.z_calcs.append(z0)
        # rotate & translate based on geom
        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        r0 = (x0**2+y0**2)**(1/2)
        phi0 = np.arctan2(y0, x0)
        # TEST
        if debug:
            print(x0, y0, z0)
            print(r0)
            print(phi0)
        # calculate integrands
        Br_integrand_calcs, Bz_integrand_calcs = Br_Bz_integrands(r0, self.Z+z0, self.R, self.j_tot)
        # save integrands (testing)
        self.Br_int = Br_integrand_calcs
        self.Bz_int = Bz_integrand_calcs
        # integrate Br
        if self.lib is tc:
            Br = trapz_2d(self.rs, self.zs, Br_integrand_calcs, self.int_func).item()
            Bz = trapz_2d(self.rs, self.zs, Bz_integrand_calcs, self.int_func).item()
        else:
            Br = trapz_2d(self.rs, self.zs, Br_integrand_calcs, self.int_func)
            Bz = trapz_2d(self.rs, self.zs, Bz_integrand_calcs, self.int_func)
        # save results in coil's coordinates
        self.Br_calcs_prerot.append(Br)
        self.Bz_calcs_prerot.append(Bz)
        # calculate cartesian
        Bx = Br * np.cos(phi0)
        By = Br * np.sin(phi0)
        # rotate back to Mu2e coordinates
        B_vec = np.array([Bx, By, Bz])
        B_vec_rot = self.coil_to_mu2e.apply(B_vec)
        self.Bx_calcs.append(B_vec_rot[0])
        self.By_calcs.append(B_vec_rot[1])
        self.Bz_calcs.append(B_vec_rot[2])
        return B_vec_rot

    def integrate_vec(self, x0=np.array([0]), y0=np.array([0]), z0=np.array([0]), debug=False):
        # append to lists
        self.x_calcs.append(x0)
        self.y_calcs.append(y0)
        self.z_calcs.append(z0)
        # rotate & translate based on geom
        x0, y0, z0 = self.mu2e_to_coil.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]).T).T
        r0 = (x0**2+y0**2)**(1/2)
        phi0 = np.arctan2(y0, x0)
        # TEST
        if debug:
            print(x0, y0, z0)
            print(r0)
            print(phi0)
        # calculate integrands
        Br_integrand_calcs, Bz_integrand_calcs = Br_Bz_integrands_vec(r0[:,None,None],
                                                                      self.Z[None,...]+z0[:,None,None],
                                                                      self.R[None,...], self.j_tot)
        # save integrands (testing)
        self.Br_int = Br_integrand_calcs
        self.Bz_int = Bz_integrand_calcs
        # integrate Br
        if self.lib is tc:
            Br = trapz_2d(self.rs, self.zs, Br_integrand_calcs, self.int_func).item()
            Bz = trapz_2d(self.rs, self.zs, Bz_integrand_calcs, self.int_func).item()
        else:
            Br = trapz_2d(self.rs, self.zs, Br_integrand_calcs, self.int_func)
            Bz = trapz_2d(self.rs, self.zs, Bz_integrand_calcs, self.int_func)
        # save results in coil's coordinates
        self.Br_calcs_prerot.append(Br)
        self.Bz_calcs_prerot.append(Bz)
        # calculate cartesian
        Bx = Br * np.cos(phi0)
        By = Br * np.sin(phi0)
        # rotate back to Mu2e coordinates
        B_vec = np.array([Bx, By, Bz]).T
        B_vec_rot = self.coil_to_mu2e.apply(B_vec).T
        self.Bx_calcs.append(B_vec_rot[0])
        self.By_calcs.append(B_vec_rot[1])
        self.Bz_calcs.append(B_vec_rot[2])
        return B_vec_rot

    def integrate_grid(self, df, N_proc=None, OPTIMAL=None, tqdm=tqdm, verbose=False):
        # initial time
        t0 = time()
        i = int(round(self.geom_coil.Coil_Num))
        print(f'Coil {i}: grid with {len(df):E} points')
        # add dataframe to object
        self.df = df.copy() # passed in dataframe with columns X, Y, Z [m]
        # defaults if necessary
        if N_proc is None:
            N_proc = 32 # or num.cpu for long calculation (warn others)
        if OPTIMAL is None:
            # should tune this
            #OPTIMAL = 5000000
            OPTIMAL = 500000
        # calculate best chunk size
        N_per_chunk = int(OPTIMAL/self.R.size)+1
        N_chunks = int(len(df)/N_per_chunk) + 1
        # let user know chunk size
        print(f'Chunk size: {N_per_chunk}, Number of chunks: {N_chunks}')
        # generate padded arrays corresponding to chunk size and number of chunks
        vals_list = []
        for col in ['X', 'Y', 'Z']:
            vals = np.zeros(N_per_chunk * N_chunks)
            vals[:len(self.df)] = self.df[col]
            vals = vals.reshape((N_chunks, N_per_chunk))
            vals_list.append(vals)
        # parallel calculations
        num_cpu = multiprocessing.cpu_count()
        num_cpu = N_proc
        if verbose:
            print(f"CPU Cores available: {num_cpu}")
            print(f'Using {num_cpu} cores')
        # calculate chunks
        output_tuples = Parallel(n_jobs=num_cpu)\
        (delayed(self.integrate_vec)(x_, y_, z_) for x_, y_, z_ in tqdm(zip(*vals_list), desc='Chunk #', total=len(vals_list[0])))
        # sort and store results
        Bxs = []
        Bys = []
        Bzs = []
        for t in output_tuples:
            Bxs.append(t[0])
            Bys.append(t[1])
            Bzs.append(t[2])
        Bxs = np.array(Bxs).flatten()[:len(self.df)]
        Bys = np.array(Bys).flatten()[:len(self.df)]
        Bzs = np.array(Bzs).flatten()[:len(self.df)]
        xs = vals_list[0].flatten()[:len(self.df)]
        ys = vals_list[1].flatten()[:len(self.df)]
        zs = vals_list[2].flatten()[:len(self.df)]

        self.df.loc[:, f'Bx_solcalc_{i}'] = Bxs
        self.df.loc[:, f'By_solcalc_{i}'] = Bys
        self.df.loc[:, f'Bz_solcalc_{i}'] = Bzs

        # final time, report total time
        tf = time()
        print(f'Calculation time: {(tf - t0):0.2f} s\n')

        return self.df

    def save_grid_calc(self, savetype='pkl', savename='data/Mau13.PS_region.standard.coil0', all_solcalc_cols=False):
        # determine which columns to save
        i = int(round(self.geom_coil.Coil_Num))
        cols = ['X', 'Y', 'Z']
        for col in self.df.columns:
            if all_solcalc_cols:
                if 'solcalc' in col:
                    cols.append(col)
            else:
                if f'solcalc_{i}' in col:
                    cols.append(col)
        # save
        df_to_save = self.df[cols]
        if savetype == 'pkl':
            df_to_save.to_pickle(f'{savename}.{savetype}')
        else:
            raise NotImplementedError('Allowed savetype: ["pkl"]')

# see if this works outside class defn.
def integrate_grid(SolCalc, df, N_proc=None, OPTIMAL=None, tqdm=tqdm, verbose=False):
    i = round(SolCalc.geom_coil.Coil_Num)
    print(f'Coil {i}: grid with {len(df):E} points')
    ## add dataframe to object
    ##df = df.copy() # passed in dataframe with columns X, Y, Z [m]
    # defaults if necessary
    if N_proc is None:
        N_proc = 32 # or num.cpu for long calculation (warn others)
    if OPTIMAL is None:
        # should tune this
        #OPTIMAL = 5000000
        OPTIMAL = 500000
    # calculate best chunk size
    N_per_chunk = int(OPTIMAL/SolCalc.R.size)+1
    N_chunks = int(len(df)/N_per_chunk) + 1
    # let user know chunk size
    print(f'Chunk size: {N_per_chunk}, Number of chunks: {N_chunks}')
    # generate padded arrays corresponding to chunk size and number of chunks
    vals_list = []
    for col in ['X', 'Y', 'Z']:
        vals = np.zeros(N_per_chunk * N_chunks)
        vals[:len(df)] = df[col]
        vals = vals.reshape((N_chunks, N_per_chunk))
        vals_list.append(vals)
    # parallel calculations
    num_cpu = multiprocessing.cpu_count()
    num_cpu = N_proc
    if verbose:
        print(f"CPU Cores available: {num_cpu}")
        print(f'Using {num_cpu} cores')
    # calculate chunks
    output_tuples = Parallel(n_jobs=num_cpu)\
    (delayed(SolCalc.integrate_vec)(x_, y_, z_) for x_, y_, z_ in tqdm(zip(*vals_list), desc='Chunk #', total=len(vals_list[0])))
    # sort and store results
    Bxs = []
    Bys = []
    Bzs = []
    for t in output_tuples:
        Bxs.append(t[0])
        Bys.append(t[1])
        Bzs.append(t[2])
    Bxs = np.array(Bxs).flatten()[:len(df)]
    Bys = np.array(Bys).flatten()[:len(df)]
    Bzs = np.array(Bzs).flatten()[:len(df)]
    xs = vals_list[0].flatten()[:len(df)]
    ys = vals_list[1].flatten()[:len(df)]
    zs = vals_list[2].flatten()[:len(df)]

    df.loc[:, f'Bx_solcalc_{i}'] = Bxs
    df.loc[:, f'By_solcalc_{i}'] = Bys
    df.loc[:, f'Bz_solcalc_{i}'] = Bzs

    return df


if __name__=='__main__':
    from helicalc import helicalc_dir
    from helicalc.geometry import read_solenoid_geom_combined
    geom_df_mu2e = read_solenoid_geom_combined(helicalc_dir+'dev/params/','Mu2e_V13')
    i = 0 # 1st coil in the PS
    #i = 5 # coil in the TS
    # testing grid
    # df_PS = pd.read_pickle('/home/ckampa/coding/mu2e_utils/BField_plot/data/PSMap_no-offset.pkl')
    # step size
    drz = np.array([5e-3, 1e-2])
    # set up integrator
    mySolCalc = SolCalcIntegrator(geom_df_mu2e.iloc[i], drz=drz)
    print(f'Sample Calc @ x,y,z = (3.904, 0, -6.8): Bx, By, Bz = {mySolCalc.integrate_single(3.904, 0, -6.8)}')
    # grid does not work from this file
    # integrate on PS map
    # should take ~10 seconds
    # class method
    #mySolCalc.integrate_grid(df_PS)
    # global method
    #df = integrate_grid(mySolCalc, df_PS)
