import numpy as np
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go

from helicalc import helicalc_dir
from helicalc.geometry import read_solenoid_geom_combined

def cylinder(r, h, xc=0, yc=0, zc=0, pitch=0., yaw=0., roll=0., nt=100, nv=50, flip_angles=False):
    # generate grid of theta, z (or v for vertical)
    theta = np.linspace(2*np.pi+np.pi/4, 0+np.pi/4, nt)
    v = np.linspace(-h/2, h/2, nv)
    # make 2D grid for plotting later on
    TH, VV = np.meshgrid(theta, v)
    # cylindrical --> cartesian
    x = (r*np.cos(TH)).flatten()
    y = (r*np.sin(TH)).flatten()
    z = (VV).flatten()
    pos = np.array([x,y,z])
    # apply any coil rotations, w.r.t. cylinder center
    rot_angles = np.array([pitch, yaw, roll])
    # set to true or false depending on rotation sign definitions
    if flip_angles:
        rot_angles *= -1
    rot = Rotation.from_euler('XYZ', rot_angles, degrees=True)
    pos_rot = rot.apply(pos.T).T
    # get rotated x, y, z and translate to correct location
    x = pos_rot[0] + xc
    y = pos_rot[1] + yc
    z = pos_rot[2] + zc
    # reshape to 2D arrays for surface plotting
    x = x.reshape((len(v), len(theta)))
    y = y.reshape((len(v), len(theta)))
    z = z.reshape((len(v), len(theta)))
    return x, y, z

def get_cylinder_inner_surface_xyz(df, coil_num):
    c = df.query(f'Coil_Num == {coil_num}').iloc[0]
    x, y, z = cylinder(c.Ro, c.L, xc=c.x, yc=c.y, zc=c.z,
                       pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                       nt=360, nv=3, flip_angles=False)
    return x,y,z

def get_cylinder_outer_surface_xyz(df, coil_num):
    c = df.query(f'Coil_Num == {coil_num}').iloc[0]
    x, y, z = cylinder(c.Ro, c.L, xc=c.x, yc=c.y, zc=c.z,
                       pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                       nt=360, nv=3, flip_angles=False)
    return x,y,z

def get_thick_cylinder_surface_xyz(df, coil_num):
    c = df.query(f'Coil_Num == {coil_num}').iloc[0]
    # inner shell
    xi, yi, zi = cylinder(c.Ri, c.L, xc=c.x, yc=c.y, zc=c.z,
                          pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                          nt=360, nv=3, flip_angles=False)
    # outer shell
    xo, yo, zo = cylinder(c.Ro, c.L, xc=c.x, yc=c.y, zc=c.z,
                          pitch=c.rot0, yaw=c.rot1, roll=c.rot2,
                          nt=360, nv=3, flip_angles=False)
    # combine inner and outer layers
    x = np.concatenate([xo, xi[::-1], xo[:1]], axis=0)
    y = np.concatenate([yo, yi[::-1], yo[:1]], axis=0)
    z = np.concatenate([zo, zi[::-1], zo[:1]], axis=0)
    return x,y,z

def get_thick_cylinders_padded(df, coil_nums):
    xs = []
    ys = []
    zs = []
    cs = []
    for cn in coil_nums:
        x_, y_, z_ = get_thick_cylinder_surface_xyz(df, cn)
        # pad x_, y_, z_
        x_ = np.insert(np.insert(x_, 0, x_[0], axis=0), -1, x_[-1], axis=0)
        y_ = np.insert(np.insert(y_, 0, y_[0], axis=0), -1, y_[-1], axis=0)
        z_ = np.insert(np.insert(z_, 0, z_[0], axis=0), -1, z_[-1], axis=0)
        # add color value array and set padding to 0 (to turn transparent)
        cs_ = np.ones_like(x_)
        cs_[0, :] = 0
        cs_[-1, :] = 0
        # append to lists
        xs.append(x_)
        ys.append(y_)
        zs.append(z_)
        cs.append(cs_)
    # concatenate to np arrays and return
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    cs = np.concatenate(cs)
    return xs, ys, zs, cs

if __name__=='__main__':
    # load geometry
    # paramdir = '/home/ckampa/coding/helicalc/dev/params/'
    paramdir = helicalc_dir+'dev/params/'
    paramfile = 'Mu2e_V13'
    df_PS_nom = read_solenoid_geom_combined(paramdir, paramfile).iloc[:3]
    # test get thick cylinder
    #x, y, z = get_thick_cylinder_surface_xyz(df_PS_nom, 1)
    #print(x, y, z)
    # test thick cylinders with padding
    xs, ys, zs, cs = get_thick_cylinders_padded(df_PS_nom, [1, 2, 3])
    print(xs, ys, zs, cs)
