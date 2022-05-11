from sys import getsizeof
from time import time
import numpy as np
import torch as tc
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from helicalc import helicalc_data
from .integrate import *
from .constants import *
from .tools import *


## STRAIGHT BARS (Longitudinal, Tangential)
class StraightIntegrator3D(object):
    def __init__(self, geom_df, dxyz, int_func=tc.trapz, lib=tc, dev=0):
        # library setup
        self.lib = lib
        self.int_func = int_func
        self.dev = dev
        # GPU setup
        self.start_dev_mem = get_gpu_memory_map()[dev]
        self.mem_err_expected = False
        if lib is tc:
            tc.cuda.set_device(dev)
        # store relevant parameters
        self.geom_df = geom_df
        self.L = geom_df.length
        self.I = geom_df.I
        self.I_flow = geom_df.I_flow
        self.W = geom_df.W
        self.T = geom_df['T']
        self.j = self.I / (self.W*self.T)
        self.mu_fac = mu_0 * self.j / (4*np.pi)
        self.dxyz = dxyz
        # local origin -- depends on "I_flow"
        if np.isclose(self.I_flow, 0.):
            self.xc = geom_df.x0
            self.yc = geom_df.y0
            self.zc = geom_df.z0
        else:
            self.xc = geom_df.x1
            self.yc = geom_df.y1
            self.zc = geom_df.z0
        # integration steps
        # use correct library
        if lib is tc:
            self.xps = lib.linspace(-self.W/2, self.W/2, abs(int(self.W/self.dxyz[0] + 1))).cuda()
            self.yps = lib.linspace(-self.T/2, self.T/2, abs(int(self.T/self.dxyz[1] + 1))).cuda()
            self.zps = lib.linspace(0, self.L, abs(int(self.L/self.dxyz[2] + 1))).cuda()
        else:
            self.xps = lib.linspace(-self.W/2, self.W/2, abs(int(self.W/self.dxyz[0] + 1)))
            self.yps = lib.linspace(-self.T/2, self.T/2, abs(int(self.T/self.dxyz[1] + 1)))
            self.zps = lib.linspace(0, self.L, abs(int(self.L/self.dxyz[2] + 1)))
        self.XP, self.YP, self.ZP = lib.meshgrid(self.xps, self.yps, self.zps, indexing='ij')
        # rotation
        self.euler2 = geom_df[['Phi2', 'theta2', 'psi2']].values
        self.rot = Rotation.from_euler('zyz', self.euler2[::-1], degrees=True)
        self.inv_rot = self.rot.inv()

    def integrate(self, x0, y0, z0):
        # attempt 1
        ###x0, y0, z0 = self.rot.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        # attempt 2
        x0, y0, z0 = self.inv_rot.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        RX = rx_str(x0, self.XP)
        RY = ry_str(y0, self.YP)
        RZ = rz_str(z0, self.ZP)
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        result = []
        for integrand_func in [straight_integrand_Bx, straight_integrand_By]:
            integrand_xyz = self.mu_fac * integrand_func(RX, RY, R2_32)
            result.append(trapz_3d(self.xps, self.yps, self.zps, integrand_xyz, self.int_func).item())
        B_vec = np.array(result+[0.])
        # rotate vector back to mu2e coordinates
        # attempt 1
        ###B_vec_rot = self.inv_rot.apply(B_vec)
        # attempt 2
        B_vec_rot = self.rot.apply(B_vec)
        self.last_result_norot = B_vec
        self.last_result = B_vec_rot
        return self.last_result

    def integrate_vec(self, x0_vec, y0_vec, z0_vec):
        # attempt 1
        # x0_vec, y0_vec, z0_vec = self.rot.apply(np.array([x0_vec-self.xc, y0_vec-self.yc, z0_vec-self.zc]).T).T
        # attempt 2
        x0_vec, y0_vec, z0_vec = self.inv_rot.apply(np.array([x0_vec-self.xc, y0_vec-self.yc, z0_vec-self.zc]).T).T
        if self.lib is tc:
            x0_vec = tc.from_numpy(x0_vec).cuda()
            y0_vec = tc.from_numpy(y0_vec).cuda()
            z0_vec = tc.from_numpy(z0_vec).cuda()
        RX = rx_str(x0_vec[:,None,None,None], self.XP[None,...])
        RY = ry_str(y0_vec[:,None,None,None], self.YP[None,...])
        RZ = rz_str(z0_vec[:,None,None,None], self.ZP[None,...])
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        result = []
        for integrand_func in [straight_integrand_Bx, straight_integrand_By]:
            integrand_xyz = self.mu_fac * integrand_func(RX, RY, R2_32)
            result.append(trapz_3d(self.xps, self.yps, self.zps, integrand_xyz, self.int_func))
        if self.lib is tc:
            B_vecs = tc.stack(result+[tc.zeros_like(x0_vec)]).cpu()
        else:
            B_vecs = np.array(result+[np.zeros_like(x0_vec)])
        # rotate vector back to mu2e coordinates
        # attempt 1
        # B_vecs_rot = self.inv_rot.apply(B_vecs.T).T
        # attempt 2
        B_vecs_rot = self.rot.apply(B_vecs.T).T
        self.last_result_norot = B_vecs
        self.last_result = B_vecs_rot
        return self.last_result

    def integrate_grid(self, df, N_batch=10000, tqdm=tqdm):
        # initial time
        t0 = time()
        # print info about bus bar
        i = int(round(self.geom_df['cond N']))
        print(f'Straight Bus Bar {i}: grid with {len(df):E} points')
        # add dataframe to object
        self.df = df.copy()
        # calculate number of chunks
        N_per_chunk = N_batch
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
        # loop through chunks and save results
        Bxs = []
        Bys = []
        Bzs = []
        for x_, y_, z_ in tqdm(zip(*vals_list), desc='Chunk #', total=len(vals_list[0])):
            Bx_, By_, Bz_ = self.integrate_vec(x_, y_, z_)
            Bxs.append(Bx_)
            Bys.append(By_)
            Bzs.append(Bz_)

        Bxs = np.array(Bxs).flatten()[:len(self.df)]
        Bys = np.array(Bys).flatten()[:len(self.df)]
        Bzs = np.array(Bzs).flatten()[:len(self.df)]
        xs = vals_list[0].flatten()[:len(self.df)]
        ys = vals_list[1].flatten()[:len(self.df)]
        zs = vals_list[2].flatten()[:len(self.df)]

        self.df.loc[:, f'Bx_bus_str_cn_{i}'] = Bxs
        self.df.loc[:, f'By_bus_str_cn_{i}'] = Bys
        self.df.loc[:, f'Bz_bus_str_cn_{i}'] = Bzs

        # final time, report total time
        tf = time()
        print(f'Calculation time: {(tf - t0):0.2f} s\n')

        return self.df

    def save_grid_calc(self, savetype='pkl', savename=f'Bmaps/helicalc_partial/Mau13.DS_region.standard-busbar.cond_N_57_straight_TEST', all_cols=False):
        # determine which columns to save
        i = int(round(self.geom_df['cond N']))
        cols = ['X', 'Y', 'Z']
        for col in self.df.columns:
            if all_cols:
                if 'bus_str' in col:
                    cols.append(col)
            else:
                if f'bus_str_cn_{i}' in col:
                    cols.append(col)
        # save
        df_to_save = self.df[cols]
        if savetype == 'pkl':
            df_to_save.to_pickle(f'{helicalc_data}{savename}.{savetype}')
        else:
            raise NotImplementedError('Allowed savetype: ["pkl"]')


## CIRCLE ARC BARS
class ArcIntegrator3D(object):
    def __init__(self, geom_df, dxyz, int_func=tc.trapz, lib=tc, dev=0):
    #def __init__(self, rho, I, euler2, xc, yc, zc, phi0=0, phif=2*np.pi, W=1e-2, T=2e-2, dxyz=np.array([1e-3, 1e-3, 1e-3])):
        # library setup
        self.lib = lib
        self.int_func = int_func
        self.dev = dev
        # GPU setup
        self.start_dev_mem = get_gpu_memory_map()[dev]
        self.mem_err_expected = False
        if lib is tc:
            tc.cuda.set_device(dev)
        # only works if geom_df is a series
        if 'R_curve' in geom_df.index:
            self.rho = geom_df.R_curve # different setup for transfer line arcs
        else:
            self.rho = geom_df.R0 # does not work for last 4 arcs (to transfer line)
        self.geom_df = geom_df
        self.I = geom_df.I
        self.W = geom_df.W
        self.T = geom_df['T']
        self.j = self.I / (self.W*self.T)
        self.mu_fac = mu_0 * self.j / (4*np.pi)
        self.phi0 = 0.
        self.phif = np.radians(geom_df.dphi)
        self.dxyz = dxyz
        # local origin
        self.xc = geom_df.x0
        self.yc = geom_df.y0
        self.zc = geom_df.z0
        # integration steps
        # use correct library
        if lib is tc:
            self.xps = lib.linspace(-self.W/2, self.W/2, abs(int(self.W/self.dxyz[0] + 1))).cuda()
            self.yps = lib.linspace(-self.T/2, self.T/2, abs(int(self.T/self.dxyz[1] + 1))).cuda()
            self.phis = lib.linspace(self.phi0, self.phif, abs(int((self.phif-self.phi0)/self.dxyz[2] + 1))).cuda()
        else:
            self.xps = lib.linspace(-self.W/2, self.W/2, abs(int(self.W/self.dxyz[0] + 1)))
            self.yps = lib.linspace(-self.T/2, self.T/2, abs(int(self.T/self.dxyz[1] + 1)))
            self.phis = lib.linspace(self.phi0, self.phif, abs(int((self.phif-self.phi0)/self.dxyz[2] + 1)))
        self.XP, self.YP, self.PHI = lib.meshgrid(self.xps, self.yps, self.phis, indexing='ij')
        # extra calculations
        self.SINPHI = lib.sin(self.PHI)
        self.COSPHI = lib.cos(self.PHI)
        self.RHOP = self.rho - self.YP
        # rotations
        self.euler2 = geom_df[['Phi2', 'theta2', 'psi2']].values
        self.rot = Rotation.from_euler('zyz', self.euler2[::-1], degrees=True)
        self.inv_rot = self.rot.inv()

    def integrate(self, x0, y0, z0):
        x0, y0, z0 = self.inv_rot.apply(np.array([x0-self.xc, y0-self.yc, z0-self.zc]))
        RX = rx_arc(x0, self.XP)
        RY = ry_arc(y0, self.rho, self.RHOP, self.COSPHI)
        RZ = rz_arc(z0, self.RHOP, self.SINPHI)
        self.RX = RX
        self.RY = RY
        self.RZ = RZ
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        self.R2_32 = R2_32
        result = []
        # int_func must have params (x, y, z, x0, y0, z0)
        for integrand_func in [arc_integrand_Bx, arc_integrand_By, arc_integrand_Bz]:
            integrand_xyz = self.mu_fac * integrand_func(self.RHOP, self.SINPHI, self.COSPHI, RX, RY, RZ, R2_32)
            result.append(trapz_3d(self.xps, self.yps, self.phis, integrand_xyz, self.int_func).item())
#             result.append(np.trapz(np.trapz(np.trapz(integrand_xyz, axis=-1, x=self.phis), axis=-1, x=self.yps), axis=-1, x=self.xps))
        B_vec = np.array(result)
        self.last_result_norot = B_vec
        self.last_result = self.rot.apply(B_vec)
        return self.last_result

    def integrate_vec(self, x0_vec, y0_vec, z0_vec):
        x0_vec, y0_vec, z0_vec = self.inv_rot.apply(np.array([x0_vec-self.xc, y0_vec-self.yc, z0_vec-self.zc]).T).T
        if self.lib is tc:
            x0_vec = tc.from_numpy(x0_vec).cuda()
            y0_vec = tc.from_numpy(y0_vec).cuda()
            z0_vec = tc.from_numpy(z0_vec).cuda()
        RX = rx_arc(x0_vec[:,None,None,None], self.XP[None,...])
        RY = ry_arc(y0_vec[:,None,None,None], self.rho, self.RHOP[None,...], self.COSPHI[None,...])
        RZ = rz_arc(z0_vec[:,None,None,None], self.RHOP[None,...], self.SINPHI[None,...])
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        result = []
        for integrand_func in [arc_integrand_Bx, arc_integrand_By, arc_integrand_Bz]:
            integrand_xyz = self.mu_fac * integrand_func(self.RHOP, self.SINPHI, self.COSPHI, RX, RY, RZ, R2_32)
            result.append(trapz_3d(self.xps, self.yps, self.phis, integrand_xyz, self.int_func))
        # coil
        if self.lib is tc:
            B_vecs = tc.stack(result).cpu()
        else:
            B_vecs = np.array(result)
        # rotate vector back to mu2e coordinates
        B_vecs_rot = self.rot.apply(B_vecs.T).T
        self.last_result_norot = B_vecs
        self.last_result = B_vecs_rot
        return self.last_result

    def integrate_grid(self, df, N_batch=10000, tqdm=tqdm):
        # initial time
        t0 = time()
        # print info about bus bar
        try:
            i = int(round(self.geom_df['cond N']))
        except:
            i = self.geom_df['cond N']
        print(f'Arc Bus Bar {i}: grid with {len(df):E} points')
        # add dataframe to object
        self.df = df.copy()
        # calculate number of chunks
        N_per_chunk = N_batch
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
        # loop through chunks and save results
        Bxs = []
        Bys = []
        Bzs = []
        for x_, y_, z_ in tqdm(zip(*vals_list), desc='Chunk #', total=len(vals_list[0])):
            Bx_, By_, Bz_ = self.integrate_vec(x_, y_, z_)
            Bxs.append(Bx_)
            Bys.append(By_)
            Bzs.append(Bz_)

        Bxs = np.array(Bxs).flatten()[:len(self.df)]
        Bys = np.array(Bys).flatten()[:len(self.df)]
        Bzs = np.array(Bzs).flatten()[:len(self.df)]
        xs = vals_list[0].flatten()[:len(self.df)]
        ys = vals_list[1].flatten()[:len(self.df)]
        zs = vals_list[2].flatten()[:len(self.df)]

        self.df.loc[:, f'Bx_bus_arc_cn_{i}'] = Bxs
        self.df.loc[:, f'By_bus_arc_cn_{i}'] = Bys
        self.df.loc[:, f'Bz_bus_arc_cn_{i}'] = Bzs

        # final time, report total time
        tf = time()
        print(f'Calculation time: {(tf - t0):0.2f} s\n')

        return self.df

    def save_grid_calc(self, savetype='pkl', savename=f'Bmaps/helicalc_partial/Mau13.DS_region.standard-busbar.cond_N_57_arc_TEST', all_cols=False):
        # determine which columns to save
        try:
            i = int(round(self.geom_df['cond N']))
        except:
            i = self.geom_df['cond N']
        cols = ['X', 'Y', 'Z']
        for col in self.df.columns:
            if all_cols:
                if 'bus_arc' in col:
                    cols.append(col)
            else:
                if f'bus_arc_cn_{i}' in col:
                    cols.append(col)
        # save
        df_to_save = self.df[cols]
        if savetype == 'pkl':
            df_to_save.to_pickle(f'{helicalc_data}{savename}.{savetype}')
        else:
            raise NotImplementedError('Allowed savetype: ["pkl"]')
