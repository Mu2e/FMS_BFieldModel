from sys import getsizeof
from time import time
import numpy as np
import torch as tc
from tqdm import tqdm
from helicalc import helicalc_data
from .integrate import *
from .constants import *
from .tools import *

class RadialStraightIntegrator1D(object):
    def __init__(self, geom_df, dr, int_func=tc.trapz, lib=tc, dev=0):
        # library setup
        self.lib = lib
        self.int_func = int_func
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
        # note mu_fac handles current direction!!
        self.mu_fac = self.I_flow * mu_0 * self.I / (4*np.pi)
        self.dr = dr
        # where is the axis?
        self.x_axis_offset = geom_df.x_axis_offset
        self.y_axis_offset = geom_df.y_axis_offset
        # start point (close radius)
        self.xc = geom_df.x0 - self.x_axis_offset
        self.yc = geom_df.y0 - self.y_axis_offset
        self.zc = geom_df.z0
        # initial radius and phi
        self.rhoc = (self.xc**2 + self.yc**2)**(1/2)
        self.phic = np.arctan2(self.yc, self.xc)
        self.COSPHIC = np.cos(self.phic)
        self.SINPHIC = np.sin(self.phic)
        # integration steps
        # use correct library
        if lib is tc:
            self.rhos = lib.linspace(self.rhoc, self.rhoc + self.L, abs(int(self.L/self.dr + 1))).cuda()
        else:
            self.rhos = lib.linspace(self.rhoc, self.rhoc + self.L, abs(int(self.L/self.dr + 1)))

    def integrate(self, x0, y0, z0):
        # translate point
        x0 = x0 - self.x_axis_offset
        y0 = y0 - self.y_axis_offset
        # calculate r vec
        RX = rx_r1d(self.rhos, self.COSPHIC, x0)
        RY = ry_r1d(self.rhos, self.SINPHIC, y0)
        RZ = rz_r1d(self.rhos, self.zc, z0)
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        # get angular information of field point
        transv0 = (x0**2 + y0**2)**(1/2)
        if transv0 < 1e-6:
            sin0 = 0.
            cos0 = 1.
        else:
            sin0 = y0 / transv0
            cos0 = x0 / transv0
        RPHI = -RX*sin0 + RY*cos0
        result = []
        for integrand_func in [radial_integrand_Bphi, radial_integrand_Bz]:
            integrand_xyz = self.mu_fac * integrand_func(RPHI, RZ, R2_32)
            result.append(trapz_1d(self.rhos, integrand_xyz, self.int_func).item())
        # r, phi, z
        B_vec_cyl = np.array([0.]+result)
        # go back to cartesian
        B_vec_cart = np.array([B_vec_cyl[0]*cos0 - B_vec_cyl[1]*sin0, B_vec_cyl[0]*sin0 + B_vec_cyl[1]*cos0, B_vec_cyl[2]])
        self.B_vec_cyl = B_vec_cyl
        self.B_vec_cart = B_vec_cart
        return self.B_vec_cart

    def integrate_vec(self, x0_vec, y0_vec, z0_vec):
        # translate point
        x0_vec = x0_vec - self.x_axis_offset
        y0_vec = y0_vec - self.y_axis_offset
        if self.lib is tc:
            x0_vec = tc.from_numpy(x0_vec).cuda()
            y0_vec = tc.from_numpy(y0_vec).cuda()
            z0_vec = tc.from_numpy(z0_vec).cuda()
        # calculate r vec
        RX = rx_r1d(self.rhos[None,...], self.COSPHIC, x0_vec[:, None])
        RY = ry_r1d(self.rhos[None,...], self.SINPHIC, y0_vec[:, None])
        RZ = rz_r1d(self.rhos[None,...], self.zc, z0_vec[:, None])
        R2_32 = (RX**2 + RY**2 + RZ**2)**(3/2)
        # get angular information of field point
        transv0 = (x0_vec**2 + y0_vec**2)**(1/2)
        sin0 = self.lib.zeros_like(x0_vec)
        cos0 = self.lib.ones_like(x0_vec)
        m = transv0 >= 1e-6
        sin0[m] = y0_vec[m] / transv0[m]
        cos0[m] = x0_vec[m] / transv0[m]
        RPHI = -RX*sin0[:,None] + RY*cos0[:,None]
        result = []
        for integrand_func in [radial_integrand_Bphi, radial_integrand_Bz]:
            integrand_xyz = self.mu_fac * integrand_func(RPHI, RZ, R2_32)
            result.append(trapz_1d(self.rhos, integrand_xyz, self.int_func))
        # r, phi, z
        if self.lib is tc:
            B_vecs_cyl = tc.stack([tc.zeros_like(x0_vec)]+result).cpu()
            sin0 = sin0.cpu()
            cos0 = cos0.cpu()
            # go back to cartesian
            B_vecs_cart = tc.stack([B_vecs_cyl[0,:]*cos0 - B_vecs_cyl[1,:]*sin0, B_vecs_cyl[0,:]*sin0 + B_vecs_cyl[1,:]*cos0, B_vecs_cyl[2,:]]).numpy()
            B_vecs_cyl = B_vecs_cyl.numpy()
        else:
            B_vecs_cyl = np.array([np.zeros_like(x0_vec)]+result)
            # go back to cartesian
            B_vecs_cart = np.array([B_vecs_cyl[0,:]*cos0 - B_vecs_cyl[1,:]*sin0, B_vecs_cyl[0,:]*sin0 + B_vecs_cyl[1,:]*cos0, B_vecs_cyl[2,:]])
        self.B_vecs_cyl = B_vecs_cyl
        self.B_vecs_cart = B_vecs_cart
        return self.B_vecs_cart

    def integrate_grid(self, df, N_batch=10000, tqdm=tqdm, verbose=1):
        # info about conductor
        i = self.geom_df['Coil_Num']
        if verbose > 0:
            # initial time
            t0 = time()
            print(f'Radial Current Coil {i}: grid with {len(df):E} points')
        # add dataframe to object
        self.df = df.copy()
        # calculate number of chunks
        N_per_chunk = N_batch
        N_chunks = int(len(df)/N_per_chunk) + 1
        if verbose > 0:
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
        if (tqdm is not None) and (verbose > 0):
            for x_, y_, z_ in tqdm(zip(*vals_list), desc='Chunk #', total=len(vals_list[0])):
                Bx_, By_, Bz_ = self.integrate_vec(x_, y_, z_)
                Bxs.append(Bx_)
                Bys.append(By_)
                Bzs.append(Bz_)
        else:
            for x_, y_, z_ in zip(*vals_list):
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

        self.df.loc[:, f'Bx_radial_coil_{i}'] = Bxs
        self.df.loc[:, f'By_radial_coil_{i}'] = Bys
        self.df.loc[:, f'Bz_radial_coil_{i}'] = Bzs

        if verbose > 0:
            # final time, report total time
            tf = time()
            print(f'Calculation time: {(tf - t0):0.2f} s\n')

        return self.df

    def save_grid_calc(self, savetype='pkl', savename=f'Bmaps/helicalc_partial/Mu2e_V13.DS_region.auxiliary.Coil_Num_0out_radial_TEST', all_cols=False):
        # determine which columns to save
        i = self.geom_df['Coil_Num']
        cols = ['X', 'Y', 'Z']
        for col in self.df.columns:
            if all_cols:
                if 'radial_coil' in col:
                    cols.append(col)
            else:
                if f'radial_coil_{i}' in col:
                    cols.append(col)
        # check for Hall probe label
        if "HP" in self.df.columns:
            cols.append("HP")
        # save
        df_to_save = self.df[cols]
        if savetype == 'pkl':
            df_to_save.to_pickle(f'{helicalc_data}{savename}.{savetype}')
        else:
            raise NotImplementedError('Allowed savetype: ["pkl"]')
