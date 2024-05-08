import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def config_plots():
    #must run twice for some reason (glitch in Jupyter)
    for i in range(2):
        plt.rcParams['figure.figsize'] = [10, 8] # larger figures
        plt.rcParams['axes.grid'] = True         # turn grid lines on
        plt.rcParams['axes.axisbelow'] = True    # put grid below points
        plt.rcParams['grid.linestyle'] = '--'    # dashed grid
        plt.rcParams.update({'font.size': 12.0})   # increase plot font size

# create grid (cartesian)
def generate_cartesian_grid_df(grid_dict, dec_round=3):
    g = grid_dict
    edges = [np.round(np.linspace(g[f'{i}0'], g[f'{i}0']+(g[f'n{i}']-1)*g[f'd{i}'], g[f'n{i}']), decimals=dec_round)
             for i in ['X', 'Y', 'Z']]
    X, Y, Z = np.meshgrid(*edges, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    df = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})

    return df

# create grid (cylindrical)
def generate_cylindrical_grid_df(grid_dict, dec_round=3):
    # Note R may not be equally spaced. Allow passing in a list of R values
    # to R0. In this case nR and dR are meaningless.
    if type(grid_dict) is not list:
        grid_dict = [grid_dict]
    #g = grid_dict
    df = []
    for g in grid_dict:
        edges = [np.round(np.linspace(g[f'{i}0'], g[f'{i}0']+(g[f'n{i}']-1)*g[f'd{i}'], g[f'n{i}']), decimals=dec_round)
                 for i in ['Phi', 'Z']]
        if isinstance(g['R0'], Iterable):
            edges.insert(0, g['R0'])
        else:
            edges.insert(0, np.round(np.linspace(g['R0'], g['R0']+(g['nR']-1)*g['dR'], g['nR']), decimals=dec_round))
        R, Phi, Z = np.meshgrid(*edges, indexing='ij')
        R = R.flatten()
        Phi = Phi.flatten()
        Z = Z.flatten()
        # transform to X, Y, Z
        X = R * np.cos(Phi) + g['XOffset']
        Y = R * np.sin(Phi)
        if "HP_labels" in g.keys():
            label_map = {r: l for r,l in zip(g['R0'], g['HP_labels'])}
            # map probe name onto radius values
            hp_labs = np.vectorize(label_map.get)(R)
            df_ = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z, 'HP': hp_labs})
        else:
            df_ = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})
        df.append(df_)
    df = pd.concat(df)
    return df

# # create grid (2D, with cylindrical symmetry
# def generate_cyl2d_grid_df(grid_dict, dec_round=3):
#     g = grid_dict
#     edges = [np.round(np.linspace(g[f'{i}0'], g[f'{i}0']+(g[f'n{i}']-1)*g[f'd{i}'], g[f'n{i}']), decimals=dec_round)
#              for i in ['R', 'Z']]
#     R, Z = np.meshgrid(*edges, indexing='ij')
#     R = R.flatten()
#     Z = Z.flatten()
#     df = pd.DataFrame({'R':R, 'Z':Z})

#     return df

# add points for numerical Jacobian calculation
def add_points_for_J(df, dxyz=0.001):
    x0s, y0s, z0s = df[['X', 'Y', 'Z']].values.T
    xs = np.concatenate(np.array([x0s, x0s, x0s, x0s, x0s, x0s + dxyz, x0s - dxyz]).T)
    ys = np.concatenate(np.array([y0s, y0s, y0s, y0s + dxyz, y0s - dxyz, y0s, y0s]).T)
    zs = np.concatenate(np.array([z0s, z0s + dxyz, z0s - dxyz, z0s, z0s, z0s, z0s]).T)
    df = pd.DataFrame({'X': xs, 'Y': ys, 'Z': zs})
    return df

# calculate cable length of all coils in a solenoid
def calc_cable_lengths(geom_df, use_rho_0_a=True):
    N_cables = len(geom_df)
    L_cable_list = []
    L_cable_coils = 0.
    for i in np.arange(N_cables):
        geom_coil = geom_df.iloc[i]
        if use_rho_0_a:
            R0 = (geom_coil.rho0_a+geom_coil.rho1_a)/2.
        else:
            R0 = (geom_coil.Ri + geom_coil.h_cable/2.)
        N_L = int(geom_coil.N_layers)
        L_cable = 0.
        for l in range(N_L):
            R = R0 + l*geom_coil.h_cable
            L_cable += 2*np.pi*R*geom_coil.N_turns
        L_cable_list.append(L_cable)
        L_cable_coils += L_cable
    return L_cable_coils, L_cable_list
