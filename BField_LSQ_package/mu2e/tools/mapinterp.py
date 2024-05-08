#! /usr/bin/env python

import numpy as np
from scipy.interpolate import griddata

from mu2e import mu2e_ext_path
from mu2e.dataframeprod import DataFrameMaker


def get_df_interp_func(df=None,filename="datafiles/Mau13/Mu2e_DSMap_V13", method="linear",input_type='pkl',ck=True, gauss=True):
    '''
    This factory function will return an interpolating function for any field map. An input x,y,z will output the corresponding Bx,By,Bz or Br,Bphi,Bz. Will decide later if linear interpolation is good enough.
    '''
    # load dataframe if not passed in
    if df is None:
        df = DataFrameMaker(mu2e_ext_path+"datafiles/Mau13/Mu2e_DSMap_V13", input_type=input_type).data_frame
    if not gauss:
        df["Bx"] = df["Bx"] / 1e4
        df["By"] = df["By"] / 1e4
        df["Bz"] = df["Bz"] / 1e4

    xs = df.X.unique()
    ys = df.Y.unique()
    zs = df.Z.unique()

    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    dz = zs[1]-zs[0]

    lx = len(xs)
    ly = len(ys)
    lz = len(zs)

    df_np = df[["X","Y","Z","Bx","By","Bz"]].values

    def get_cube(x,y,z):
        a_x, a_y, a_z = len(xs[xs <= x]) - 1, len(ys[ys <= y]) - 1, len(zs[zs <= z]) - 1
        corner_a = (ly * lz) * a_x + (lz) * a_y + a_z
        corner_b = corner_a + lz
        corner_c = corner_a + ly * lz
        corner_d = corner_a + ly * lz + lz
        index_list = [corner_a,corner_a+1,corner_b,corner_b+1,
        corner_c,corner_c+1,corner_d,corner_d+1]
        return df_np[index_list]

    def interp(x,y,z,cart=True):
        cube = get_cube(x,y,z)
        bx = griddata((cube[:,0],cube[:,1],cube[:,2]),cube[:,3], (x,y,z), method=method).item()
        by = griddata((cube[:,0],cube[:,1],cube[:,2]),cube[:,4], (x,y,z), method=method).item()
        bz = griddata((cube[:,0],cube[:,1],cube[:,2]),cube[:,5], (x,y,z), method=method).item()
        return bx,by,bz

    def ck_interp_single(xd,yd,zd,ff):
        c00 = ff[0,0,0]*(1 - xd) + ff[1,0,0] * xd
        c01 = ff[0,0,1]*(1 - xd) + ff[1,0,1] * xd
        c10 = ff[0,1,0]*(1 - xd) + ff[1,1,0] * xd
        c11 = ff[0,1,1]*(1 - xd) + ff[1,1,1] * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        return c0 * (1 - zd) + c1 * zd


    # def ck_interp(x,y,z,cart=True):
    def ck_interp(p_vec):
        cube = get_cube(*p_vec)

        xx = np.unique(cube[:,0])
        yy = np.unique(cube[:,1])
        zz = np.unique(cube[:,2])

        bxs_grid = cube[:,3].reshape((2,2,2))
        bys_grid = cube[:,4].reshape((2,2,2))
        bzs_grid = cube[:,5].reshape((2,2,2))

        xd = (p_vec[0]-xx[0])/(xx[1]-xx[0])
        yd = (p_vec[1]-yy[0])/(yy[1]-yy[0])
        zd = (p_vec[2]-zz[0])/(zz[1]-zz[0])

        bx = ck_interp_single(xd,yd,zd, bxs_grid)
        by = ck_interp_single(xd,yd,zd, bys_grid)
        bz = ck_interp_single(xd,yd,zd, bzs_grid)

        return bx,by,bz


    if ck:
        return ck_interp
    else:
        return interp
