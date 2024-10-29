import os
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from BFieldPINN.NN_callbacks import model_predict, model_predict_with_jacobian

### tensorflow GPU initialization
def init_GPU():
    print('Initializing GPU environment variables.')
    os.system('CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))')
    os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib')

def set_GPU(dev="1"):
    if type(dev) is not str:
        dev = str(int(dev))
    print(f'Setting GPU {dev}.')
    os.environ["CUDA_VISIBLE_DEVICES"] = dev

def get_GPU():
    out = subprocess.run('echo $CUDA_VISIBLE_DEVICES', shell=True, capture_output=True)
    dev = out.stdout.decode('utf-8').rstrip()
    return dev

### Prepare for refit
def make_LSQ_df_PINN_subtracted(df, fname=None, name=None):
    # evaluate new Bi values based on PINN prediction dBi_NN
    df_ = df.copy()
    for i in ['x', 'y', 'z', 'phi', 'r']:
        df_.loc[:, f'B{i}'] = df_.loc[:, f'B{i}'] - df_.loc[:, f'dB{i}_NN']
    # save only the relevant columns
    cols_to_save = ['X', 'Y', 'Z', 'Bx', 'By', 'Bz', 'R', 'Phi', 'Bphi', 'Br']
    # not in all dfs
    for k, loc in zip(['HP', 'fit_point'], [3, None]):
        if k in df.columns:
            if loc is None:
                cols_to_save.append(k)
            else:
                cols_to_save.insert(loc, k)
    df_ = df_[cols_to_save]
    if not fname is None:
        print(f'Saving {name}: {fname}')
        df_.to_pickle(fname)
    return df_

### EVALUATE PINN ON df
def evaluate_PINN_on_dfs(PINN_inst, df_dict, keys_to_eval, make_jac, N_chunk=100000):
    # any df that doesn't have an entry in keys_to_eval will be passed over
    # if df_test_jac, derivative results will be added to df_test
    # and exact derivatives
    for k in keys_to_eval:
        if k not in df_dict.keys():
            continue
        print(f'Evaluating PINN on {k}...')
        if (k == 'df_test') and (make_jac):
            func = model_predict_with_jacobian
        else:
            func = model_predict
        df_ = df_dict[k]
        pred, jac = func(PINN_inst, df_.X_norm.values, df_.Y_norm.values,
                         df_.Z_norm.values, N_i=N_chunk)
        # add to df
        for i, coord in enumerate(['x', 'y', 'z']):
            df_dict[k].loc[:, f'dB{coord}_NN'] = pred[:, i]
        # cylindrical
        df_.eval('dBphi_NN = -dBx_NN*sin(Phi)+dBy_NN*cos(Phi)', inplace=True)
        df_.eval('dBr_NN = dBx_NN*cos(Phi)+dBy_NN*sin(Phi)', inplace=True)
        # calculate div and curl if available -- should only happen once (df_test)
        # FIXME! (low priority) I can imagine someone wanting to evaluate div
        # and curl on a number of different dataframes.
        if not jac is None:
            div_test = calc_div(jac)
            curl_test = calc_curl(jac)
            curl_test_norm = np.linalg.norm(curl_test, axis=1)
        print(f'Done.')
    print('Evaluations done.')
    # combine jacobian calculations if needed
    if make_jac:
        print('Combining jacobian calculations (numerical and exact)')
        # calculate div and curl for df_test_jac
        df_jac, J = div_and_curl_calculations(df_dict['df_test_jac'],
                                              Bcols=['dBx_NN', 'dBy_NN', 'dBz_NN'],
                                              Gauss_to_T=False, m_to_mm=False)
        # add all derivatives to df_test
        # exact
        for col in ['divB', 'curlB', 'curlB_x', 'curlB_y', 'curlB_z']:
            df_dict['df_test'].loc[:, col] = df_jac.loc[:, col].values
        # numerical
        df_dict['df_test'].loc[:, 'divB_exact'] = div_test
        df_dict['df_test'].loc[:, 'curlB_exact'] = curl_test_norm
        df_dict['df_test'].loc[:, 'curlB_x_exact'] = curl_test[:, 0]
        df_dict['df_test'].loc[:, 'curlB_y_exact'] = curl_test[:, 1]
        df_dict['df_test'].loc[:, 'curlB_z_exact'] = curl_test[:, 2]
        # add df_jac to dict -- no, all info added to df_test
        #df_dict['df_jac'] = df_jac
        print('Done.')
    print()
    return df_dict

### PREP INPUTS FOR PINN ###
def prep_PINN_inputs(files_dict, NN_dict):
    ### DATA
    # load data
    df_dict, norm_dict = load_and_process_FMS_fit_data(files_dict['in']['meas'], files_dict['in']['test'], NN_dict)
    # split train / test
    df_meas = df_dict['df_meas']
    if 'fit_point' in df_dict['df_meas'].columns:
        df_meas_ = df_meas.query('fit_point').copy()
    else:
        df_meas_ = df_meas
    N_train = int(len(df_meas_) * NN_dict['perc_train'])
    df_meas_sample = df_meas_.sample(frac=1)
    df_train = df_meas_sample[:N_train]
    df_val = df_meas_sample[N_train:]
    # update df_dict
    df_dict['df_train'] = df_train
    df_dict['df_val'] = df_val
    ### PINN
    # data
    # get values ready for NN instantiation
    x_u = df_train.X_norm.values.reshape(len(df_train), 1)
    y_u = df_train.Y_norm.values.reshape(len(df_train), 1)
    z_u = df_train.Z_norm.values.reshape(len(df_train), 1)
    u_labels = df_train[['dBx', 'dBy', 'dBz']].values.reshape(len(df_train), 3)
    validation_labels = df_val[['dBx', 'dBy', 'dBz']].values.reshape(len(df_val), 3)
    validation_labels = tf.cast(validation_labels, dtype=tf.float32)
    validation_data = tf.concat([
        tf.cast(df_val.X_norm.values.reshape(len(df_val), 1), dtype=tf.float32),
        tf.cast(df_val.Y_norm.values.reshape(len(df_val), 1), dtype=tf.float32),
        tf.cast(df_val.Z_norm.values.reshape(len(df_val), 1), dtype=tf.float32)
    ], axis=1)
    # track data
    track_stride = NN_dict['track_stride']
    if NN_dict['track']:
        lines = []
        for q_str in NN_dict['track_queries']:
            lines.append(df_meas.query(q_str).copy())
        df_track = pd.concat(lines, ignore_index=True)
        X_ = df_track.X_norm.values.reshape(len(df_track), 1)
        Y_ = df_track.Y_norm.values.reshape(len(df_track), 1)
        Z_ = df_track.Z_norm.values.reshape(len(df_track), 1)
        if 'fit_point' in df_track.columns:
            F_ = df_track.fit_point.values.reshape(len(df_track), 1).astype(float)
        else:
            F_ = np.ones_like(X_)
        tracking_data = tf.concat([tf.cast(X_, dtype=tf.float32),
                                   tf.cast(Y_, dtype=tf.float32),
                                   tf.cast(Z_, dtype=tf.float32),
                                   tf.cast(F_, dtype=tf.float32),], axis=1)
    else:
        tracking_data = None
    # layers
    # depends on NN type
    layers_in = [3]
    for i in range(NN_dict['N_hidden']):
        layers_in.append(NN_dict['N_nodes'])
    # add correct number of final nodes, given NN type
    if NN_dict['NN_type'] == 'Scalar':
        layers_in.append(1)
    else:
        layers_in.append(3)
    # activation, lamdba, regularization
    activ = NN_dict['activ']
    snake_a = NN_dict['snake_a']
    lambda_ = NN_dict['lambda_']
    reg = NN_dict['reg']
    # number of collocation points
    N_f = NN_dict['N_f']
    # initializer
    if NN_dict['initializer_type'] == 'uniform':
        lim = NN_dict['initializer_lim']
        seed = NN_dict['initializer_seed']
        initializer_w = tf.keras.initializers.RandomUniform(minval=-lim, maxval=lim, seed=seed)
        initializer_b = tf.keras.initializers.RandomUniform(minval=-lim, maxval=lim, seed=seed)
        initializer = [initializer_w, initializer_b]
    else:
        # nothing else implemented
        initializer = None
    colloc_seed = NN_dict['colloc_seed']
    # add everything to a config_dict
    init_config = {}
    config_keys = [
        'norm_dict', 'x_u', 'y_u', 'z_u', 'validation_data', 'validation_labels',
        'u_labels', 'layers_in', 'activ', 'snake_a', 'lambda_', 'reg',
        'N_f', 'tracking_data', 'track_stride', 'initializer', 'colloc_seed',
    ]
    for k in config_keys:
        exec(f'init_config["{k}"] = {k}')
    # return initializer config and dfs
    return init_config, df_dict

### DATASETS ###
def load_and_process_FMS_fit_data(fname_meas, fname_test, NN_dict, remove_central_duplicates=False, remove_central_line=False):
    df_meas = pd.read_pickle(fname_meas)
    df_test = pd.read_pickle(fname_test)
    # remove central line / copies?
    if remove_central_line:
        q_remove = '(X == 0.0) & (Y == 0.0)'
        df_meas = df_meas.query(f'~({q_remove})').copy()
        df_test = df_test.query(f'~({q_remove})').copy()
    elif remove_central_duplicates:
        #q_remove = '(X == 0.0) & (Y == 0.0) & (Phi != 0)'
        q_remove = f'(X == 0.0) & (Y == 0.0) & ((Phi < {-1/8 * np.pi - 0.0001}) | (Phi > {-1/8 * np.pi + 0.0001}))'
        df_meas = df_meas.query(f'~({q_remove})').copy()
        df_test = df_test.query(f'~({q_remove})').copy()
    # sorting to match input and reset indexes
    df_meas.sort_values(by=['X', 'Y', 'Z'], inplace=True)
    df_meas.reset_index(drop=True, inplace=True)
    df_test.sort_values(by=['X', 'Y', 'Z'], inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    # convert from cylindrical to cartesian, calculate all deltas
    for df_ in [df_meas, df_test]:
        # field strength -- unused
        #df_.loc[:, 'B'] = (df_.loc[:, 'Bx']**2 + df_.loc[:, 'By']**2 + df_.loc[:, 'Bz']**2)**(1/2)
        #df_.loc[:, 'B_fit'] = (df_.loc[:, 'Bx_fit']**2 + df_.loc[:, 'By_fit']**2 + df_.loc[:, 'Bz_fit']**2)**(1/2)
        # calcualte deltas: data - fit, i.e. what we have left to model.
        # these are the values we will train the PINN on
        #for i in ['x', 'y', 'z', 'r', 'phi', '']:
        for i in ['x', 'y', 'z', 'r', 'phi']:
            df_.loc[:, f'dB{i}'] = df_.loc[:, f'B{i}'] - df_.loc[:, f'B{i}_fit']
    # add Jacobian, if it exists
    df_list = [df_meas, df_test]
    if NN_dict['make_jacobian_df']:
        df_test_jac = add_points_for_J(df_test, dxyz=NN_dict['jac_dxyz'])
        df_list.append(df_test_jac)
    else:
        df_test_jac = None
    # normalize inputs
    norm_dict = construct_normalizations_dict(df_meas, coords_to_normalize=['X', 'Y', 'Z'])
    for df_ in df_list:
        for coord in norm_dict.keys():
            df_.loc[:, f'{coord}_norm'] = world_to_NN(df_.loc[:, coord], coord, norm_dict)
    df_dict = {'df_meas': df_meas, 'df_test': df_test, 'df_test_jac': df_test_jac}
    return df_dict, norm_dict

def construct_normalizations_dict(df, coords_to_normalize=['X', 'Y', 'Z']):
    norm_dict = {i: {} for i in coords_to_normalize}
    for coord in coords_to_normalize:
        vals = df[coord]
        norm_dict[coord]['mean'] = (vals.max() + vals.min()) / 2.
        norm_dict[coord]['range'] = (vals.max() - vals.min())
    return norm_dict

def world_to_NN(world_values, coord, norm_dict):
    NN_values = (world_values - norm_dict[coord]['mean']) / (norm_dict[coord]['range'] / 2.)
    return NN_values

def NN_to_world(NN_values, coord, norm_dict):
    world_values = (NN_values * norm_dict[coord]['range'] / 2.) + norm_dict[coord]['mean']
    return world_values


### JACOBIAN (numerical)
# add points for numerical Jacobian calculation
def add_points_for_J(df, dxyz=0.001):
    x0s, y0s, z0s = df[['X', 'Y', 'Z']].values.T
    xs = np.concatenate(np.array([x0s, x0s, x0s, x0s, x0s, x0s + dxyz, x0s - dxyz]).T)
    ys = np.concatenate(np.array([y0s, y0s, y0s, y0s + dxyz, y0s - dxyz, y0s, y0s]).T)
    zs = np.concatenate(np.array([z0s, z0s + dxyz, z0s - dxyz, z0s, z0s, z0s, z0s]).T)
    df = pd.DataFrame({'X': xs, 'Y': ys, 'Z': zs})
    # evaluate cylindrical coords
    df.eval('R = sqrt(X**2+Y**2)', inplace=True)
    df.eval('Phi = arctan2(Y,X)', inplace=True)
    return df

# numerical derivatives
def calc_jacobian_numerical(df_, Bcols=['dBx', 'dBy', 'dBz']):
    '''
    df_ should be a pd.DataFrame containing repetitions of the following pattern order:
    0: nominal, 1: +Z, 2: -Z, 3: +Y, 4: -Y, 5: +X, 6: -X
    returns: J, where row indicates B component, column indicates numerical component
    '''
    cols = ['X', 'Y', 'Z'] + Bcols
    x, y, z, Bx, By, Bz = df_[cols].values.T
    J = np.zeros((len(x)//7, 3, 3))
    # dx
    J[:, 0, 0] = (Bx[5::7] - Bx[6::7]) / (x[5::7] - x[6::7]) # dBx
    J[:, 1, 0] = (By[5::7] - By[6::7]) / (x[5::7] - x[6::7]) # dBy
    J[:, 2, 0] = (Bz[5::7] - Bz[6::7]) / (x[5::7] - x[6::7]) # dBz
    # dy
    J[:, 0, 1] = (Bx[3::7] - Bx[4::7]) / (y[3::7] - y[4::7]) # dBx
    J[:, 1, 1] = (By[3::7] - By[4::7]) / (y[3::7] - y[4::7]) # dBy
    J[:, 2, 1] = (Bz[3::7] - Bz[4::7]) / (y[3::7] - y[4::7]) # dBz
    # dz
    J[:, 0, 2] = (Bx[1::7] - Bx[2::7]) / (z[1::7] - z[2::7]) # dBx
    J[:, 1, 2] = (By[1::7] - By[2::7]) / (z[1::7] - z[2::7]) # dBy
    J[:, 2, 2] = (Bz[1::7] - Bz[2::7]) / (z[1::7] - z[2::7]) # dBz

    # Gauss / m, when coming from NN -- be careful about X, Y, Z scaling
    return J

def scale_J(J, Gauss_to_T=False, m_to_mm=False):
    if Gauss_to_T:
        J = J*1e-4
    if m_to_mm:
        J = J*1e-3
    return J

def calc_div(J):
    #return np.sum(np.diag(J))
    return J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]

def calc_curl(J):
    curl_x = J[:, 2, 1] - J[:, 1, 2]
    curl_y = J[:, 0, 2] - J[:, 2, 0]
    curl_z = J[:, 1, 0] - J[:, 0, 1]
    return np.array([curl_x, curl_y, curl_z]).T

def div_and_curl_calculations(df, Bcols=['dBx', 'dBy', 'dBz'], Gauss_to_T=False, m_to_mm=False):
    J = calc_jacobian_numerical(df, Bcols)
    J = scale_J(J, Gauss_to_T, m_to_mm)
    div = calc_div(J)
    curl_vec = calc_curl(J)
    curl = np.linalg.norm(curl_vec, axis=1)
    # set in df
    df_nom = df.iloc[::7].copy()
    df_nom.reset_index(drop=True, inplace=True)
    df_nom.loc[:, 'divB'] = div
    df_nom.loc[:, 'curlB'] = curl
    df_nom.loc[:, 'curlB_x'] = curl_vec[:, 0]
    df_nom.loc[:, 'curlB_y'] = curl_vec[:, 1]
    df_nom.loc[:, 'curlB_z'] = curl_vec[:, 2]
    return df_nom, J
