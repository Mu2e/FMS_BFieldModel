import numpy as np
import pandas as pd

# numerical derivatives
def calc_jacobian_numerical(df_):
    '''
    df_ should be a pd.DataFrame containing repetitions of the following pattern order:
    0: nominal, 1: +Z, 2: -Z, 3: +Y, 4: -Y, 5: +X, 6: -X
    returns: J, where row indicates B component, column indicates numerical component
    '''
    x, y, z, Bx, By, Bz = df_[['X', 'Y', 'Z', 'Bx', 'By', 'Bz']].values.T
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

    # T / m, when coming directly from helicalc
    return J

def scale_J(J, T_to_Gauss=False, m_to_mm=False):
    if T_to_Gauss:
        J = J*1e4
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

def div_and_curl_calculations(df, T_to_Gauss=False, m_to_mm=False):
    J = calc_jacobian_numerical(df)
    J = scale_J(J, T_to_Gauss, m_to_mm)
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
