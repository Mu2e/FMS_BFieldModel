import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from helicalc import helicalc_dir
from .geometry import read_solenoid_geom_combined

# Loading dataframes with geometry information
def load_all_geoms(version=14, return_dict=True):
    # files
    geom_dir = helicalc_dir+'dev/params/'
    # old file
    #coils_file = geom_dir + f'Mu2e_Coils_Conductors_V{version}.pkl'
    interlayer_file = geom_dir+f'Mu2e_V{version}_coil_interlayer.txt'
    straight_file = geom_dir + f'Mu2e_Straight_Bars_V{version}.csv'
    arc_file = geom_dir + f'Mu2e_Arc_Bars_V{version}.csv'
    arc_transfer_file = geom_dir + f'Mu2e_Arc_Transfer_Bars_V{version}.csv'
    # load dataframes
    #df_coils = pd.read_pickle(coils_file)
    # read coils and add which solenoid system each coil is in
    df_coils = read_solenoid_geom_combined(helicalc_dir+'dev/params/',
                                           f'Mu2e_V{version}')
    # move this? or don't hard code?
    # FIXME!
    df_coils['Solenoid'] = 'TS'
    df_coils.loc[0:2, 'Solenoid'] = 'PS'
    df_coils.loc[55:, 'Solenoid'] = 'DS'
    df_coils['Solenoid_full'] = df_coils['Solenoid']
    df_coils.loc[3:5, 'Solenoid_full'] = 'TS1'
    df_coils.loc[6:23, 'Solenoid_full'] = 'TS2'
    df_coils.loc[24:31, 'Solenoid_full'] = 'TS3'
    df_coils.loc[32:49, 'Solenoid_full'] = 'TS4'
    df_coils.loc[50:54, 'Solenoid_full'] = 'TS5'
    # coil interlayer connection
    df_interlayer = pd.read_csv(interlayer_file)
    df_str = pd.read_csv(straight_file)
    df_arc = pd.read_csv(arc_file)
    df_arc_tr = pd.read_csv(arc_transfer_file)
    if return_dict:
        df_dict = {'coils': df_coils, 'interlayers': df_interlayer,
                   'straights': df_str, 'arcs': df_arc,
                   'arcs_transfer': df_arc_tr}
        return df_dict
    else:
        return df_coils, df_interlayer, df_str, df_arc, df_arc_tr

# creating other relevant dictionaries
def make_conductor_dicts(df_dict):
    # dictionary with relevant Monte Carlo generation function
    # based on geometry
    generate_dict = {'coils': gen_helical_coil_allsurface_points,
                     'interlayers': gen_arc_allsurface_points,
                     'straights': gen_straight_allsurface_points,
                     'arcs': gen_arc_allsurface_points,
                     'arcs_transfer': gen_arc_allsurface_points}
    # make a lookup dictionary for which condctor type corresponds to each
    # conductor number
    conductor_dict = {}
    id_column_dict = {}
    for key, df in df_dict.items():
        if key not in ['coils', 'interlayers']:
            for v in df['cond N'].values:
                conductor_dict[str(v)] = key
                id_column_dict[str(v)] = 'cond N'
        elif key == 'coils':
            for v in df['Coil_Num'].values:
                # check if in DS
                if v>=56:
                    conductor_dict[str(v)+'_c'] = key
                    id_column_dict[str(v)+'_c'] = 'Coil_Num'
        elif key == 'interlayers':
            for v in df['cond N'].values:
                conductor_dict[str(v)+'_il'] = key
                id_column_dict[str(v)+'_il'] = 'cond N'
        else:
            raise ValueError(f'Geometry key "{key}" not recognized.')

    return generate_dict, conductor_dict, id_column_dict

# Coils (idea cylinders)
def cylinder(r, h, xc=0, yc=0, zc=0, pitch=0., yaw=0., roll=0.,
             nt=100, nv=50, flip_angles=False):
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

def get_many_thick_cylinders(df):
    # lists to store all cylinder information (will become multi-dim np.arrays)
    xs = []
    ys = []
    zs = []
    cs = []
    # loop through all coils, by coil number
    for cn in df.Coil_Num:
        # get x,y,z for this cylinder
        # (both inner and outer walls)
        x_, y_, z_ = get_thick_cylinder_surface_xyz(df, cn)
        # pad x_, y_, z_ for transparency between coils
        x_ = np.insert(np.insert(x_, 0, x_[0], axis=0), -1, x_[-1], axis=0)
        y_ = np.insert(np.insert(y_, 0, y_[0], axis=0), -1, y_[-1], axis=0)
        z_ = np.insert(np.insert(z_, 0, z_[0], axis=0), -1, z_[-1], axis=0)
        # color array, with padded x,y,z flipped to zero (will set transparent)
        cs_ = np.ones_like(x_)
        cs_[0, :] = 0
        cs_[-1, :] = 0
        # add to lists
        xs.append(x_)
        ys.append(y_)
        zs.append(z_)
        cs.append(cs_)
    # create numpy arrays from gathered results
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    cs = np.concatenate(cs)

    return xs, ys, zs, cs

# bus bars
# straight sections (longitudinal and tangential)
def get_3d_straight(df, bar_num, nz=20, center=False):
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    euler2 = np.array([df_.Phi2, df_.theta2, df_.psi2])
    # extrinsic rotation
    rot = Rotation.from_euler("zyz", angles=euler2[::-1], degrees=True)
    I_flow = df_.I_flow
    if np.isclose(I_flow, 0.):
        x0 = df_.x0
        y0 = df_.y0
        z0 = df_.z0
    else:
        x0 = df_.x1
        y0 = df_.y1
        z0 = df_.z0
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    # length
    L = df_.length
    # start in starting frame
    #zs = np.arange(z0, z0+L+1e-2, 1e-2)
    xs_list = []
    ys_list = []
    zs_list = []
    if center:
        dx_list = [0.]
        dy_list = [0.]
    else:
        dx_list = [-W/2, -W/2, W/2, W/2]
        dy_list = [-T/2, T/2, T/2, -T/2]
    for dx, dy in zip(dx_list, dy_list):
        zs = np.linspace(0, L, nz)
        xs = dx*np.ones_like(zs)
        ys = dy*np.ones_like(zs)
        xs_list.append(xs)
        ys_list.append(ys)
        zs_list.append(zs)
    xs_array = np.concatenate(np.array(xs_list).T)
    ys_array = np.concatenate(np.array(ys_list).T)
    zs_array = np.concatenate(np.array(zs_list).T)
    pos = np.array([xs_array, ys_array, zs_array]).T
    #return pos.T
    rpos = rot.apply(pos)
    #return rpos.T
    rtpos = rpos + np.array([[x0, y0, z0]])[-1,:]
    return rtpos.T

def get_3d_straight_surface(df, bar_num, nphi=20):
    xs, ys, zs = get_3d_straight(df, bar_num, nphi)
    N = nphi
    # do appropriate reshaping
    x_tot = []
    y_tot = []
    z_tot = []
    c_tot = []
    for ind0, ind1, o in zip([0,1,2,3], [1,2,3,0], [1, -1, 1, -1]):
        x = xs.reshape((N, -1))[::o,[ind0,ind1]]
        y = ys.reshape((N, -1))[::o,[ind0,ind1]]
        z = zs.reshape((N, -1))[::o,[ind0,ind1]]
        c = np.ones_like(x)
        x_tot.append(x)
        y_tot.append(y)
        z_tot.append(z)
        c_tot.append(c)
    x_tot = np.concatenate(x_tot)
    y_tot = np.concatenate(y_tot)
    z_tot = np.concatenate(z_tot)
    c_tot = np.concatenate(c_tot)
    return x_tot, y_tot, z_tot, c_tot

def get_many_3d_straights(df):
    # lists to store all busbar information (will become multi-dim np.arrays)
    xs = []
    ys = []
    zs = []
    cs = []
    # loop through all arcs, by conductor number
    for cn in df['cond N']:
        # get x,y,z for this cylinder
        x_, y_, z_, cs_ = get_3d_straight_surface(df, cn)
        # pad x_, y_, z_ for transparency between bars
        x_ = np.insert(np.insert(x_, 0, x_[0], axis=0), -1, x_[-1], axis=0)
        y_ = np.insert(np.insert(y_, 0, y_[0], axis=0), -1, y_[-1], axis=0)
        z_ = np.insert(np.insert(z_, 0, z_[0], axis=0), -1, z_[-1], axis=0)
        cs_ = np.insert(np.insert(cs_, 0, cs_[0], axis=0), -1, cs_[-1], axis=0)
        # color array, with padded x,y,z flipped to zero (will set transparent)
        #cs_ = np.ones_like(x_)
        cs_[0, :] = 0
        cs_[-1, :] = 0
        # add to lists
        xs.append(x_)
        ys.append(y_)
        zs.append(z_)
        cs.append(cs_)
    # create numpy arrays from gathered results
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    cs = np.concatenate(cs)

    return xs, ys, zs, cs

# arcs
def get_3d_arc(df, bar_num, nphi=20):
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    euler2 = np.array([df_.Phi2, df_.theta2, df_.psi2])
    # extrinsic rotation
    rot = Rotation.from_euler("zyz", angles=euler2[::-1], degrees=True)
    x0 = df_.x0
    y0 = df_.y0
    z0 = df_.z0
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    # R, dphi
    if 'R_curve' in df.columns:
        R = df_.R_curve # different setup for transfer line arcs
    else:
        R = df_.R0 # does not work only for last 4 arcs (to transfer line)
    PHI = df_.dphi
    # start in starting frame
    phis = np.linspace(0, np.radians(PHI), nphi)
    xs_list = []
    ys_list = []
    zs_list = []
    for dx, dy in zip([-W/2, -W/2, W/2, W/2], [-T/2, T/2, T/2, -T/2]):
        ys = -(R+dy) * np.cos(phis) + R
        zs = (R+dy) * np.sin(phis)
        xs = np.zeros_like(phis) + dx
        xs_list.append(xs)
        ys_list.append(ys)
        zs_list.append(zs)
    xs_array = np.concatenate(np.array(xs_list).T)
    ys_array = np.concatenate(np.array(ys_list).T)
    zs_array = np.concatenate(np.array(zs_list).T)
    pos = np.array([xs_array, ys_array, zs_array]).T
    rpos = rot.apply(pos)
    rtpos = rpos + np.array([[x0, y0, z0]])[-1,:]
    return rtpos.T

def get_3d_arc_surface(df, bar_num, nphi=20):
    xs, ys, zs = get_3d_arc(df, bar_num, nphi)
    N = nphi
    # do appropriate reshaping
    x_tot = []
    y_tot = []
    z_tot = []
    c_tot = []
    for ind0, ind1, o in zip([0,1,2,3], [1,2,3,0], [1, -1, 1, -1]):
        x = xs.reshape((N, -1))[::o,[ind0,ind1]]
        y = ys.reshape((N, -1))[::o,[ind0,ind1]]
        z = zs.reshape((N, -1))[::o,[ind0,ind1]]
        c = np.ones_like(x)
        x_tot.append(x)
        y_tot.append(y)
        z_tot.append(z)
        c_tot.append(c)
    x_tot = np.concatenate(x_tot)
    y_tot = np.concatenate(y_tot)
    z_tot = np.concatenate(z_tot)
    c_tot = np.concatenate(c_tot)
    return x_tot, y_tot, z_tot, c_tot

def get_many_3d_arcs(df):
    # lists to store all busbar information (will become multi-dim np.arrays)
    xs = []
    ys = []
    zs = []
    cs = []
    # loop through all arcs, by conductor number
    for cn in df['cond N']:
        # get x,y,z for this cylinder
        x_, y_, z_, cs_ = get_3d_arc_surface(df, cn)
        # pad x_, y_, z_ for transparency between bars
        x_ = np.insert(np.insert(x_, 0, x_[0], axis=0), -1, x_[-1], axis=0)
        y_ = np.insert(np.insert(y_, 0, y_[0], axis=0), -1, y_[-1], axis=0)
        z_ = np.insert(np.insert(z_, 0, z_[0], axis=0), -1, z_[-1], axis=0)
        cs_ = np.insert(np.insert(cs_, 0, cs_[0], axis=0), -1, cs_[-1], axis=0)
        # color array, with padded x,y,z flipped to zero (will set transparent)
        # cs_ = np.ones_like(x_)
        cs_[0, :] = 0
        cs_[-1, :] = 0
        # add to lists
        xs.append(x_)
        ys.append(y_)
        zs.append(z_)
        cs.append(cs_)
    # create numpy arrays from gathered results
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    cs = np.concatenate(cs)

    return xs, ys, zs, cs

# Monte Carlo functions
# straight bars
def gen_straight_surface_points(df, bar_num, conductor_end='out',
                                surface_num=0, use_min_cross=True, N=1000):
    # conductor_end: 'full', 'out' or 'in'
    # surfaces when looking "head on" at the conductor end
    # surface_num: {0: 'end', 1: 'bottom', 2: 'left', 3: 'top', 4: 'right'}
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    euler2 = np.array([df_.Phi2, df_.theta2, df_.psi2])
    # extrinsic rotation
    rot = Rotation.from_euler("zyz", angles=euler2[::-1], degrees=True)
    I_flow = df_.I_flow
    if np.isclose(I_flow, 0.):
        x0 = df_.x0
        y0 = df_.y0
        z0 = df_.z0
    else:
        x0 = df_.x1
        y0 = df_.y1
        z0 = df_.z0
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    if use_min_cross:
        mc = np.min([W,T])
    else:
        mc = np.max([W,T])
    # length
    L = df_.length
    if conductor_end == 'full':
        mc = L
    if (conductor_end == 'in') or (conductor_end == 'full'):
        z_ = 0.
    else:
        z_ = L
    # start in conductor coordinates
    if surface_num == 0: # end
        xs = np.random.uniform(low=-W/2, high=W/2, size=N)
        ys = np.random.uniform(low=-T/2, high=T/2, size=N)
        zs = z_*np.ones_like(xs)
    elif (surface_num == 1) or (surface_num == 3): # bottom, top
        if surface_num == 1:
            y_ = -T/2
        else:
            y_ = T/2
        xs = np.random.uniform(low=-W/2, high=W/2, size=N)
        if (conductor_end == 'in') or (conductor_end == 'full'):
            zs = np.random.uniform(low=z_, high=mc, size=N)
        else:
            zs = np.random.uniform(low=z_-mc, high=z_, size=N)
        ys = y_*np.ones_like(xs)
    else: # left, right
        if surface_num == 2:
            x_ = -W/2
        else:
            x_ = W/2
        ys = np.random.uniform(low=-T/2, high=T/2, size=N)
        if (conductor_end == 'in') or (conductor_end == 'full'):
            zs = np.random.uniform(low=z_, high=mc, size=N)
        else:
            zs = np.random.uniform(low=z_-mc, high=z_, size=N)
        xs = x_*np.ones_like(ys)
    pos = np.array([xs, ys, zs]).T
    rpos = rot.apply(pos)
    rtpos = rpos + np.array([[x0, y0, z0]])[-1,:]
    return rtpos.T

def gen_straight_allsurface_points(df, bar_num, conductor_end='out',
                                   use_min_cross=True, N=1000, N_rel=False):
    # calculate surface area fraction in each face
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    if use_min_cross:
        mc = np.min([W,T])
    else:
        mc = np.max([W,T])
    L = df_.length
    if conductor_end == 'full':
        mc = L
    if N_rel:
        N = int(round(N*mc))
    areas = {0: W*T, 1: W*mc, 2: T*mc, 3: W*mc, 4: T*mc}
    tot_area = np.sum(list(areas.values()))
    sample_fracs = {k: v/tot_area for k, v in areas.items()}
    N_faces = {k:int(round(N*v)) for k, v in sample_fracs.items()}
    N_tot = np.sum(list(N_faces.values()))
    if N_tot != N:
        delta_N = N_tot - N
        N_faces[0] = N_faces[0] - delta_N
    # loop through faces, and save points
    xs = []
    ys = []
    zs = []
    for k, N_ in N_faces.items():
        _ = gen_straight_surface_points(df, bar_num,
                                        conductor_end=conductor_end,
                                        surface_num=k,
                                        use_min_cross=use_min_cross, N=N_)
        xs_, ys_, zs_ = _
        xs.append(xs_)
        ys.append(ys_)
        zs.append(zs_)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

# arcs
def gen_arc_surface_points(df, bar_num, conductor_end='out', surface_num=0,
                           use_min_cross=True, N=1000):
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    euler2 = np.array([df_.Phi2, df_.theta2, df_.psi2])
    # extrinsic rotation
    rot = Rotation.from_euler("zyz", angles=euler2[::-1], degrees=True)
    x0 = df_.x0
    y0 = df_.y0
    z0 = df_.z0
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    if use_min_cross:
        mc = np.min([W,T])
    else:
        mc = np.max([W,T])
    # R, dphi
    if 'R_curve' in df.columns:
        R = df_.R_curve # different setup for transfer line arcs
    else:
        R = df_.R0 # does not work only for last 4 arcs (to transfer line)
    PHI = np.radians(df_.dphi)
    dPHI = mc/R
    if conductor_end == 'full':
        dPHI = PHI
    #dPHI = min(PHI, np.radians(1))
    if (conductor_end == 'in') or (conductor_end == 'full'):
        phi_ = 0.
    else:
        phi_ = PHI
    # start in starting frame
    if surface_num == 0: # end
        dX = np.random.uniform(low=-W/2, high=W/2, size=N)
        dY = (np.random.random(N)*(2*R*T) + (R-T/2)**2)**(1/2) - R
        # reference:
        #dist = sqrt(rnd()*(R1^2-R2^2)+R2^2);
        xs = dX
        ys = -(R+dY) * np.cos(phi_) + R
        zs = (R+dY) * np.sin(phi_)
    elif (surface_num == 1) or (surface_num == 3): # bottom, top
        if surface_num == 1:
            dy_ = -T/2
        else:
            dy_ = T/2
        dX = np.random.uniform(low=-W/2, high=W/2, size=N)
        xs = dX
        if (conductor_end == 'in') or (conductor_end == 'full'):
            phis = np.random.uniform(low=phi_, high=dPHI, size=N)
        else:
            phis = np.random.uniform(low=phi_-dPHI, high=phi_, size=N)
        ys = -(R+dy_)*np.cos(phis) + R
        zs = (R+dy_)*np.sin(phis)
    else: # left, right
        if surface_num == 2:
            dx_ = -W/2
        else:
            dx_ = W/2
        dY = (np.random.random(N)*(2*R*T) + (R-T/2)**2)**(1/2) - R
        if (conductor_end == 'in') or (conductor_end == 'full'):
            phis = np.random.uniform(low=phi_, high=dPHI, size=N)
        else:
            phis = np.random.uniform(low=phi_-dPHI, high=phi_, size=N)
        ys = -(R+dY)*np.cos(phis) + R
        zs = (R+dY)*np.sin(phis)
        xs = dx_ * np.ones_like(ys)
    pos = np.array([xs, ys, zs]).T
    rpos = rot.apply(pos)
    rtpos = rpos + np.array([[x0, y0, z0]])[-1,:]
    return rtpos.T

def gen_arc_allsurface_points(df, bar_num, conductor_end='out',
                              use_min_cross=True, N=1000, N_rel=False):
    # calculate surface area fraction in each face
    df_ = df.query(f'`cond N` == {bar_num}').iloc[0]
    # W is x direction, T is y direction
    W = df_.W
    T = df_['T']
    if use_min_cross:
        mc = np.min([W,T])
    else:
        mc = np.max([W,T])
    # R, dphi
    if 'R_curve' in df.columns:
        R = df_.R_curve # different setup for transfer line arcs
    else:
        R = df_.R0 # does not work only for last 4 arcs (to transfer line)
    PHI = np.radians(df_.dphi)
    dPHI = mc/R
    if conductor_end == 'full':
        dPHI = PHI
    if N_rel:
        N = int(round(N * R*dPHI))
    areas = {0: W*T, 1: W*R*dPHI, 2: T*R*dPHI, 3: W*R*dPHI, 4: T*R*dPHI}
    tot_area = np.sum(list(areas.values()))
    sample_fracs = {k: v/tot_area for k, v in areas.items()}
    N_faces = {k:int(round(N*v)) for k, v in sample_fracs.items()}
    N_tot = np.sum(list(N_faces.values()))
    if N_tot != N:
        delta_N = N_tot - N
        N_faces[0] = N_faces[0] - delta_N
    # loop through faces, and save points
    xs = []
    ys = []
    zs = []
    for k, N_ in N_faces.items():
        _ = gen_arc_surface_points(df, bar_num, conductor_end=conductor_end,
                                   surface_num=k, use_min_cross=use_min_cross,
                                   N=N_)
        xs_, ys_, zs_ = _
        xs.append(xs_)
        ys.append(ys_)
        zs.append(zs_)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

# helical coils
def gen_helical_coil_surface_points(df, coil_num, conductor_end='out',
                                    surface_num=0, use_min_cross=True, N=1000):
    df_ = df.query(f'Coil_Num == {coil_num}').iloc[0]
    # get parameters from coil
    N_turns = int(df_.N_turns)
    N_layers = int(df_.N_layers)
    # calculate which layer we are in based on conductor_end
    if conductor_end=='out':
        layer = N_layers
    else:
        layer = 1
    #nphi = nphi_turn * N_turns + 1
    hel = df_.helicity * (-1)**(layer-1)
    zcen = df_.z
    L_proper = df_.L
    w = df_.w_sc # z
    h = df_.h_sc # radial
    phi0 = df_.phi0
    phi1 = df_.phi1
    # check how to wind
    if layer != N_layers:
        phi_start = phi0
        phi_end = phi0
    else:
        phi_start = phi0
        phi_end = phi1
    # switch phi0 or phi1 at -z if helicity = -1
    if hel < 0:
        hold = phi_start
        phi_start = phi_end
        phi_end = hold
    # how much to subtract in phis
    dphi = abs(phi_end-phi_start)
    # full number of turns
    full_dPHI = (N_turns*(2*np.pi)-dphi)
    # where are we referencing from? depends on helicity of the layer
    z_ref = zcen - hel * L_proper / 2
    # what is z of center at starting phi?
    if hel < 0:
        #z_start = z_ref - L_proper + (df_.t_gi + df_.t_ci + df_.w_cable/2)
        # not entirely certain why this is the prescription
        z_start = z_ref - L_proper + (df_.t_gi - df_.t_ci + df_.w_cable/2)
    else:
        z_start = z_ref + (df_.t_gi + df_.t_ci + df_.w_cable/2)
    # calculate actual phi range, depending on layer
    if use_min_cross:
        mc = np.min([w, h])
    else:
        mc = np.max([w, h])
    # R, dphi
    Rcen = df_.rho0_a + (layer-1)*(df_.h_cable + 2*df_.t_ci + df_.t_il)\
           + df_.h_sc/2
    #PHI = np.radians(df_.dphi)
    dPHI = mc/Rcen
    # check where 0 face is, and get phi range
    phi_range = np.array([phi_start, phi_start + hel*full_dPHI])
    if conductor_end=='in':
        phi_face0 = np.min(phi_range)
        phi_empty = phi_face0 + dPHI
    else:
        phi_face0 = np.max(phi_range)
        phi_empty = phi_face0 - dPHI
    # check which surface we are in and generate accordingly
    # start in starting frame
    if surface_num == 0: # end, phi fixed, randomize R and Z
        phis = phi_face0
        dZ = np.random.uniform(low=-w/2, high=w/2, size=N)
        dR = np.random.uniform(low=-h/2, high=h/2, size=N)

    elif (surface_num == 1) or (surface_num == 3): # bottom, top
        if surface_num == 1: # bottom, dZ const (-w/2)
            dZ = -w/2
        else: # top, dZ const (w/2)
            dZ = w/2
        dR = np.random.uniform(low=-h/2, high=h/2, size=N)
        phis = np.random.uniform(low=phi_face0, high=phi_empty, size=N)
    else: # left, right
        if surface_num == 2: # left, dR const (-h/2)
            dR = -h/2
        else: # top, dZ const (w/2)
            dR = h/2
        dZ = np.random.uniform(low=-w/2, high=w/2, size=N)
        phis = np.random.uniform(low=phi_face0, high=phi_empty, size=N)

    zs = (z_start+dZ) + np.abs(phis-phi_start)/(2*np.pi) * df_.pitch
    xs = df_.x + (Rcen+dR) * np.cos(phis)
    ys = df_.y + (Rcen+dR) * np.sin(phis)

    return xs, ys, zs

# same as before
def gen_helical_coil_allsurface_points(df, coil_num, conductor_end='out',
                                       use_min_cross=True, N=1000,
                                       N_rel=False):
    # calculate surface area fraction in each face
    df_ = df.query(f'Coil_Num == {coil_num}').iloc[0]
    N_turns = int(df_.N_turns)
    N_layers = int(df_.N_layers)
    # calculate which layer we are in based on conductor_end
    if conductor_end=='out':
        layer = N_layers
    else:
        layer = 1
    hel = df_.helicity * (-1)**(layer-1)
    # w is Z direction, h is R direction
    w = df_.w_sc
    h = df_.h_sc
    # how wide to make the sides?
    if use_min_cross:
        mc = np.min([w,h])
    else:
        mc = np.max([w,h])
    # R, dphi
    Rcen = df_.rho0_a + (layer-1)*(df_.h_cable + 2*df_.t_ci + df_.t_il)\
           + df_.h_sc/2
    R = Rcen
    #PHI = np.radians(df_.dphi)
    dPHI = mc/R
    # check if relative number given
    if N_rel:
        N = int(round(N * R*dPHI))
    # note using circular arc areas
    # may not be perfect, but pretty close
    areas = {0: w*h, 1: h*R*dPHI, 2: w*R*dPHI, 3: h*R*dPHI, 4: w*R*dPHI}
    tot_area = np.sum(list(areas.values()))
    sample_fracs = {k: v/tot_area for k, v in areas.items()}
    N_faces = {k:int(round(N*v)) for k, v in sample_fracs.items()}
    N_tot = np.sum(list(N_faces.values()))
    if N_tot != N:
        delta_N = N_tot - N
        N_faces[0] = N_faces[0] - delta_N
    # loop through faces, and save points
    xs = []
    ys = []
    zs = []
    for k, N_ in N_faces.items():
        _ = gen_helical_coil_surface_points(df, coil_num,
                                            conductor_end=conductor_end,
                                            surface_num=k,
                                            use_min_cross=use_min_cross,
                                            N=N_)
        xs_, ys_, zs_ = _
        xs.append(xs_)
        ys.append(ys_)
        zs.append(zs_)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

# single coil layer (for checking interlayer)
def gen_helical_coil_layer_surface_points(df, coil_num, conductor_end='out',
                                          layer=1, surface_num=0,
                                          use_min_cross=True, N=1000,
                                          interlayer_connect=True):
    df_ = df.query(f'Coil_Num == {coil_num}').iloc[0]
    # get parameters from coil
    N_turns = int(df_.N_turns)
    N_layers = int(df_.N_layers)
    hel = df_.helicity * (-1)**(layer-1)
    zcen = df_.z
    L_proper = df_.L
    w = df_.w_sc # z
    h = df_.h_sc # radial
    phi0 = df_.phi0
    phi1 = df_.phi1
    # check how to wind
    if layer != N_layers:
        phi_start = phi0
        phi_end = phi0
    else:
        phi_start = phi0
        phi_end = phi1
    # switch phi0 or phi1 at -z if helicity = -1
    if hel < 0:
        hold = phi_start
        phi_start = phi_end
        phi_end = hold
    # how much to subtract in phis
    dphi = abs(phi_end-phi_start)
    # full number of turns
    full_dPHI = (N_turns*(2*np.pi)-dphi)
    # where are we referencing from? depends on helicity of the layer
    z_ref = zcen - hel * L_proper / 2
    # what is z of center at starting phi?
    if hel < 0:
        #z_start = z_ref - L_proper + (df_.t_gi + df_.t_ci + df_.w_cable/2)
        # not entirely certain why this is the prescription
        z_start = z_ref - L_proper + (df_.t_gi - df_.t_ci + df_.w_cable/2)
    else:
        z_start = z_ref + (df_.t_gi + df_.t_ci + df_.w_cable/2)
    # calculate actual phi range, depending on layer
    if use_min_cross:
        mc = np.min([w, h])
    else:
        mc = np.max([w, h])
    # R, dphi
    Rcen = df_.rho0_a + (layer-1)*(df_.h_cable + 2*df_.t_ci + df_.t_il)\
           + df_.h_sc/2
    #PHI = np.radians(df_.dphi)
    dPHI = mc/Rcen
    # check where 0 face is, and get phi range
    phi_range = np.array([phi_start, phi_start + hel*full_dPHI])
    # check whether to shorten range for interlayer connect
    if layer != N_layers:
        if interlayer_connect:
            if hel < 0:
                # connect brick at phi_start
                phi_range[0] = phi_range[0] - np.radians(36.)
            else:
                # connect brick at phi_end
                phi_range[1] = phi_range[1] - np.radians(36.)
    if conductor_end=='in':
        phi_face0 = np.min(phi_range)
        phi_empty = phi_face0 + dPHI
    else:
        phi_face0 = np.max(phi_range)
        phi_empty = phi_face0 - dPHI
    # check which surface we are in and generate accordingly
    # start in starting frame
    if surface_num == 0: # end, phi fixed, randomize R and Z
        phis = phi_face0
        dZ = np.random.uniform(low=-w/2, high=w/2, size=N)
        dR = np.random.uniform(low=-h/2, high=h/2, size=N)

    elif (surface_num == 1) or (surface_num == 3): # bottom, top
        if surface_num == 1: # bottom, dZ const (-w/2)
            dZ = -w/2
        else: # top, dZ const (w/2)
            dZ = w/2
        dR = np.random.uniform(low=-h/2, high=h/2, size=N)
        phis = np.random.uniform(low=phi_face0, high=phi_empty, size=N)
    else: # left, right
        if surface_num == 2: # left, dR const (-h/2)
            dR = -h/2
        else: # top, dZ const (w/2)
            dR = h/2
        dZ = np.random.uniform(low=-w/2, high=w/2, size=N)
        phis = np.random.uniform(low=phi_face0, high=phi_empty, size=N)

    zs = (z_start+dZ) + np.abs(phis-phi_start)/(2*np.pi) * df_.pitch
    xs = df_.x + (Rcen+dR) * np.cos(phis)
    ys = df_.y + (Rcen+dR) * np.sin(phis)

    return xs, ys, zs

# same as before
def gen_helical_coil_layer_allsurface_points(df, coil_num, conductor_end='out',
                                             layer=1, interlayer_connect=True,
                                             use_min_cross=True, N=1000,
                                             N_rel=False):
    # calculate surface area fraction in each face
    df_ = df.query(f'Coil_Num == {coil_num}').iloc[0]
    N_turns = int(df_.N_turns)
    N_layers = int(df_.N_layers)
    hel = df_.helicity * (-1)**(layer-1)
    # w is Z direction, h is R direction
    w = df_.w_sc
    h = df_.h_sc
    # how wide to make the sides?
    if use_min_cross:
        mc = np.min([w,h])
    else:
        mc = np.max([w,h])
    # R, dphi
    Rcen = df_.rho0_a + (layer-1)*(df_.h_cable + 2*df_.t_ci + df_.t_il)\
           + df_.h_sc/2
    R = Rcen
    #PHI = np.radians(df_.dphi)
    dPHI = mc/R
    # check if relative number given
    if N_rel:
        N = int(round(N * R*dPHI))
    # note using circular arc areas
    # may not be perfect, but pretty close
    areas = {0: w*h, 1: h*R*dPHI, 2: w*R*dPHI, 3: h*R*dPHI, 4: w*R*dPHI}
    tot_area = np.sum(list(areas.values()))
    sample_fracs = {k: v/tot_area for k, v in areas.items()}
    N_faces = {k:int(round(N*v)) for k, v in sample_fracs.items()}
    N_tot = np.sum(list(N_faces.values()))
    if N_tot != N:
        delta_N = N_tot - N
        N_faces[0] = N_faces[0] - delta_N
    # loop through faces, and save points
    xs = []
    ys = []
    zs = []
    for k, N_ in N_faces.items():
        _ = gen_helical_coil_layer_surface_points(df, coil_num,
                                                  conductor_end=conductor_end,
                                                  layer=layer,
                                                  surface_num=k,
                                                  use_min_cross=use_min_cross,
                                                  N=N_,
                                                  interlayer_connect=interlayer_connect)
        xs_, ys_, zs_ = _
        xs.append(xs_)
        ys.append(ys_)
        zs.append(zs_)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
