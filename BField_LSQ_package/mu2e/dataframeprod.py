#! /usr/bin/env python
"""Module for converting Mu2e data into DataFrames.

This module defines classes and functions that takes Mu2E-style input files, typically csv or ROOT
files, and generates :class:`pandas.DataFrame` for use in this Python Mu2E package.  This module
can also save the output DataFrames as compressed cPickle files.  The input and output data are
saved to a subdir of the user-defined `mu2e_ext_path`, and are not committed to github.

Example:
    Using the DataFrameMaker class::

        In[1]: from mu2e.dataframeprod import DataFrameMaker
        ...    from mu2e import mu2e_ext_path

        In[2]: print mu2e_ext_path
        '/User/local/local_data'

        In[3]: df = DataFrameMaker(
        ...        mu2e_ext_path + 'datafiles/FieldMapsGA02/Mu2e_DS_GA0',
        ...        use_pickle=True,
        ...        field_map_version='GA02'
        ...    ).data_frame

        In[4]: print df.head()
        ...  X       Y       Z        Bx        By        Bz
        0 -1200.0 -1200.0  3071.0  0.129280  0.132039  0.044327
        1 -1200.0 -1200.0  3096.0  0.132106  0.134879  0.041158
        2 -1200.0 -1200.0  3121.0  0.134885  0.137670  0.037726
        3 -1200.0 -1200.0  3146.0  0.137600  0.140397  0.034024
        4 -1200.0 -1200.0  3171.0  0.140235  0.143042  0.030045

        ...  R          Phi      Bphi        Br
        0 1697.056275 -2.356194 -0.001951 -0.184780
        1 1697.056275 -2.356194 -0.001960 -0.188787
        2 1697.056275 -2.356194 -0.001969 -0.192725
        3 1697.056275 -2.356194 -0.001977 -0.196573
        4 1697.056275 -2.356194 -0.001985 -0.200307


Todo:
    * Update the particle trapping function with something more flexible.


*2016 Brian Pollack, Northwestern University*

brianleepollack@gmail.com
"""


from __future__ import absolute_import
from __future__ import print_function
import re
import six.moves.cPickle as pkl
import numpy as np
import pandas as pd
# import mu2e.src.RowTransformations as rt
from tqdm import tqdm
from six.moves import range


class DataFrameMaker(object):
    """Convert a FieldMap csv into a pandas DataFrame.

    The DataFrameMaker acts as a wrapper for :func:`pandas.DataFrame.read_csv` when
    `use_pickle` is `False`. Due to multiple differing input data csv formats, the exact csv
    options are hardcoded, depending on the `field_map_version`.

    * It is assumed that the plaintext is formatted as a csv file, with comma or space delimiters.
    * The expected headers are: 'X Y Z Bx By Bz'
    * The DataFrameMaker converts these into `pandas` DFs, where each header is its own row, as
      expected.
    * The input must be in units of mm and T (certain GA maps are hard-coded to convert to mm).
    * Offsets in the X direction are applied if specified.
    * If the map only covers one region of Y, the map is doubled and reflected about Y, such that
      Y->-Y and By->-By.
    * The following columns are constructed and added to the DF by default: 'R Phi Br Bphi'

    The outputs should be saved as compressed pickle files, and should be loaded from those files
    for further use.  Each pickle contains a single DF.

    Args:
        file_name (str): File path and name for csv/txt/pickle file.  Do not include suffix.
        field_map_version (str): Specify field map type and version (Mau9, Mau10,
            GA01/2/3/4/5)
        header_names (:obj:`list` of :obj:`str`, optional): List of headers if default is not valid.
            Default is `['X', 'Y', 'Z', 'Bx', 'By', 'Bz']`.
        input_type(str, optional): Load data from different input types. Default is `csv`.

    Attributes:
        file_name (str): File path and name, no suffix.
        field_map_version (str, optional): Mau or GA simulation type. Default to 'Mau10'.
        data_frame (pandas.DataFrame): Output DF.



    """
    def __init__(self, file_name, field_map_version='Mau10', header_names=None, input_type='csv',
                 input_df=None):
        """The DataFrameMaker initialization process.

        """

        self.file_name = re.sub('\.\w*$', '', file_name)
        self.field_map_version = field_map_version
        if header_names is None:
            header_names = ['X', 'Y', 'Z', 'Bx', 'By', 'Bz']

        # Load from pickle (all are identical in format).  Otherwise, load from csv
        if input_type == 'pkl':
            try:
                self.data_frame = pd.read_pickle(self.file_name+'.p')
            except:
                self.data_frame = pd.read_pickle(self.file_name+'.pkl')
        elif input_type == 'df':
            self.data_frame = input_df

        elif 'Mau9' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=6)

        elif 'Mau10' in self.field_map_version and 'rand' in self.file_name:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True)

        elif 'Mau10' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=6)

        elif 'GA01' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.1', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA02' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.2', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA03' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.3', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA04' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'GA05' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=4, dtype=np.float64)

        elif 'Pure_Cyl' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Pure_Hel' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=1, dtype=np.float64)

        elif 'Only' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Ideal' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Glass_Helix_v4' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=4)

        elif 'Glass' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Mau11' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                skiprows=8)

        elif 'Mau12' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=4)

        elif 'Mau13' in self.field_map_version:
            try:
                self.data_frame = pd.read_csv(
                    self.file_name+'.table', header=None, names=header_names, delim_whitespace=True,
                    skiprows=4)
            except:
                self.data_frame = pd.read_csv(
                    self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                    skiprows=4)

        elif 'Cole' in self.field_map_version:
            self.data_frame = pd.read_csv(
                self.file_name+'.txt', header=None, names=header_names, delim_whitespace=True,
                skiprows=5)

        else:
            raise KeyError("'Mau' or 'GA' not found in field_map_version: "+self.field_map_version)

    def do_basic_modifications(self, offset=None, helix=False, pitch=None, reverse=False,
                               descale=False):
        """Perform the expected modifications to the input, needed for further analysis.

        Modify the field map to add more columns, offset the X axis so it is re-centered to 0,
        and reflect the map about the Y-axis, if applicable.  In general these modifactions should
        not be applied to already-pickled inputs, as they should have already been sufficiently
        modified.

        Note:
            * Default offset is 0.
            * The PS offset is +3904 (for Mau).
            * The DS offset is -3896 (for Mau).
            * GA field maps are converted from meters to millimeters.

        Args:
            offset (float, optional): If specified, apply this offset to x-axis. Value should be in
                millimeters.
            helix (boolean, optional): If True, calculate helical coordinates in addition to other
                modifications.
            pitch (float, optional): If helix is True, pitch must be specified in order to create
                helical coordinates. Expect units of mm.  7.53mm for DS8-9-10. 5.27 for DS1-7,11

        Returns:
            Nothing. Operations are performed in place, and modify the internal `data_frame`.

        """

        print('num of rows start', len(self.data_frame.index))
        print(self.data_frame.head())
        print(self.data_frame.tail())

        # Convert to mm for some hard-coded versions
        if (('GA' in self.field_map_version and '5' not in self.field_map_version) or
                ('rand' in self.file_name) or
                ('Pure' in self.field_map_version) or
                ('MIN' in self.file_name) or
                ('MAX' in self.file_name) or
                ('Only' in self.field_map_version) or
                ('Glass' in self.field_map_version and ('v3' not in self.field_map_version and
                                                        'v4' not in self.field_map_version and
                                                        'v5' not in self.field_map_version and
                                                        'v6' not in self.field_map_version)) or
                ('Mau11' in self.field_map_version and '5096' not in self.file_name) or
                ('Ideal' in self.field_map_version)):

            self.data_frame.eval('X = X*1000', inplace=True)
            self.data_frame.eval('Y = Y*1000', inplace=True)
            self.data_frame.eval('Z = Z*1000', inplace=True)

        if descale and not reverse:
            self.data_frame.eval('X = X/1000', inplace=True)
            self.data_frame.eval('Y = Y/1000', inplace=True)
            self.data_frame.eval('Z = Z/1000', inplace=True)
            self.data_frame.eval('Bx = Bx*10000', inplace=True)
            self.data_frame.eval('By = By*10000', inplace=True)
            self.data_frame.eval('Bz = Bz*10000', inplace=True)

        # Offset x-axis
        if offset:
            self.data_frame.eval('X = X-{0}'.format(offset), inplace=True)

        if not reverse:
            # Generate radial position column
            self.data_frame.eval('R = sqrt(X**2+Y**2)', inplace=True)
            
            # Generate negative Y-axis values for some hard-coded versions.
            if (any([vers in self.field_map_version for vers in ['Mau9', 'Mau10', 'GA01']]) and
               ('rand' not in self.file_name)):

                data_frame_lower = self.data_frame.query('Y >0').copy()
                data_frame_lower.eval('Y = Y*-1', inplace=True)
                data_frame_lower.eval('By = By*-1', inplace=True)
                self.data_frame = pd.concat([self.data_frame, data_frame_lower])

            self.data_frame.eval('Phi = arctan2(Y,X)', inplace=True)
                
            # If running on an idealized DSCylFMSAll grid, will have multiple copies of the R=0 point,
            # one for each phi step. However phi information gets lost when propagating through helicalc.
            # Need to add back phi information, before computing Bphi / Br
            phi_vals = np.unique(np.array(self.data_frame.Phi).round(decimals=6))
            if len(phi_vals) != 16:
                print(phi_vals)
                exit()
            z_steps_r0 = np.unique(self.data_frame[self.data_frame.R==0.]['Z'])
            updated = 0
            for z_step in z_steps_r0:
                # More than one row with R=0 for given Z
                if self.data_frame[(self.data_frame.R==0.) & (self.data_frame.Z==z_step)].shape[0] > 1:
                    # These are duplicates if Phi is ~same between rows (within machine precision)
                    if np.var(self.data_frame[(self.data_frame.R==0.) & (self.data_frame.Z==z_step)]['Phi']) < 1e-10:
                        # Check # duplicates actually equals nPhiSteps
                        if self.data_frame[(self.data_frame.R==0.) & (self.data_frame.Z==z_step)].shape[0] == len(phi_vals):
                            self.data_frame.loc[(self.data_frame.R==0.) & (self.data_frame.Z==z_step),'Phi'] = phi_vals
                            updated += 1
                        else:
                            print(f"Cannot fill Phi for R==0, Z=={z_step} -- {self.data_frame[(self.data_frame.R==0.) & (self.data_frame.Z==z_step)].shape[0]} lines but {len(phi_vals)} phi values")
            if updated != 0:
                print(f'Updated phi values at R=0 for {updated} z steps ({len(z_steps_r0)} z steps total)')

            self.data_frame.eval('Bphi = -Bx*sin(Phi)+By*cos(Phi)', inplace=True)
            self.data_frame.eval('Br = Bx*cos(Phi)+By*sin(Phi)', inplace=True)
            
        elif reverse and not descale:
            self.data_frame.eval('X = R*cos(Phi)', inplace=True)
            self.data_frame.eval('Y = R*sin(Phi)', inplace=True)

        elif reverse and descale:
            self.data_frame.eval('Z = Z/1000', inplace=True)
            self.data_frame.eval('R = R/1000', inplace=True)
            self.data_frame.eval('X = R*cos(Phi)', inplace=True)
            self.data_frame.eval('Y = R*sin(Phi)', inplace=True)
            self.data_frame.eval('Bz = Bz*10000', inplace=True)
            self.data_frame.eval('Br = Br*10000', inplace=True)
            self.data_frame.eval('Bphi = Bphi*10000', inplace=True)

        if helix:
            if not isinstance(pitch, float):
                raise TypeError("If `helix` is True, pitch must be a float")
            raise NotImplementedError('`helix` option disabled')

        # Clean up, sort, round off.
        self.data_frame.sort_values(['X', 'Y', 'Z'], inplace=True)
        self.data_frame.reset_index(inplace=True, drop=True)
        self.data_frame = self.data_frame.round(9)
        print('num of rows end', len(self.data_frame.index))

    def make_dump(self, suffix=''):
        """Create a pickle file containing the data_frame, in the same dir as the input.

        Args:
            suffix (str, optional): Attach a suffix to the file name, before the '.p' suffix.
        """
        pkl.dump(self.data_frame, open(self.file_name+suffix+'.p', "wb"), pkl.HIGHEST_PROTOCOL)

    def make_r(self, row):
        return np.sqrt(row['X']**2+row['Y']**2)

    def make_br(self, row):
        return np.sqrt(row['Bx']**2+row['By']**2)

    def make_theta(self, row):
        return np.arctan2(row['Y'], row['X'])

    def make_bottom_half(self, row):
        return (-row['Y'])


def g4root_to_df(input_name, make_pickle=False, do_basic_modifications=False,
                 trees=['vd', 'tvd', 'part'], cluster='', tree_prefix='readvd'):
    from root_pandas import read_root
    '''
    Quick converter for virtual detector ROOT files from the Mu2E Art Framework.

    Args:
         input_name (str): file path name without suffix.
         make_pickle (bool, optional): If `True`, no dfs are returned, and a pickle is created
             containing a tuple of the relevant dfs.
             *Note: Testing out HDF5 storage instead of pickle.*
         do_basic_modifications(bool, optional): If `True`, recenter x-axis, add column 'runevt' in
             order to identify individual events. Add total momentum for nttvd.
         trees(list, optional): List of up to three potential trees to reproduce. List can have
             'vd', 'tvd', and 'part' as its members.

    Return:
        A tuple of dataframes, or none.
    '''
    do_vd = do_tvd = do_part = False

    for tree in trees:
        if tree not in ['vd', 'tvd', 'part']:
            raise KeyError(str(tree)+' is not a valid tree')
        if tree == 'vd':
            do_vd = True
        elif tree == 'tvd':
            do_tvd = True
        elif tree == 'part':
            do_part = True

    input_root = input_name + '.root'

    df_ntpart = df_nttvd = df_ntvd = None

    if tree_prefix != '':
        tree_prefix = tree_prefix+'/'
    if do_part:
        df_ntpart = read_root(input_root, tree_prefix+'ntpart', ignore='*vd')
    if do_tvd:
        df_nttvd = read_root(input_root, tree_prefix+'nttvd')
    if do_vd:
        df_ntvd = read_root(input_root, tree_prefix+'ntvd')

    if do_basic_modifications:
        if do_part:
            df_ntpart.eval('x = x+3904', inplace=True)
            df_ntpart.eval('xstop = xstop+3904', inplace=True)
            df_ntpart.eval('parent_x = parent_x+3904', inplace=True)
        if do_tvd:
            df_nttvd.eval('x = x+3904', inplace=True)
            df_nttvd.eval('p = sqrt(px**2+py**2+pz**2)', inplace=True)
        if do_vd:
            df_ntvd.eval('x = x+3904', inplace=True)
            df_ntvd.eval('p = sqrt(px**2+py**2+pz**2)', inplace=True)

    if make_pickle:
        print('loading into hdf5')
        store = pd.HDFStore(input_name+'.h5')
        if do_part:
            store['df_ntpart'] = df_ntpart
        if do_tvd:
            store['df_nttvd'] = df_nttvd
        if do_vd:
            store['df_ntvd'] = df_ntvd
        store.close()
        print('file finished')
    else:
        return (df_ntpart, df_nttvd, df_ntvd)


def g4root_to_df_skim_and_combo(input_name, total_n):
    for i in tqdm(list(range(total_n))):
        df_ntpart, df_nttvd, df_ntvd = g4root_to_df(input_name+str(i), make_pickle=False,
                                                    do_basic_modifications=True, cluster=i)
        good_runevt = df_ntpart.query('pdg==11 and p>75').runevt.unique()
        df_ntpart = df_ntpart[df_ntpart.runevt.isin(good_runevt)]
        df_ntvd = df_ntvd[df_ntvd.runevt.isin(good_runevt)]
        df_nttvd = df_nttvd[df_nttvd.runevt.isin(good_runevt)]
        output_root = input_name+str(i)+'_skim.root'
        df_ntvd.to_root(output_root, tree_key='ntvd', mode='w')
        df_nttvd.to_root(output_root, tree_key='nttvd', mode='a')
        df_ntpart.to_root(output_root, tree_key='ntpart', mode='a')


if __name__ == "__main__":
    from mu2e import mu2e_ext_path
    '''
    # Regular grid
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_DSMap_V13',
        input_type='csv', field_map_version='Mau13')
    data_maker.do_basic_modifications(-3.896, descale=True)

    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/DSMap_V15_with_shielding',
        input_type='csv', field_map_version='Mau13')
    data_maker.do_basic_modifications(-3.896, descale=True)
    data_maker.make_dump('.Mu2E')

    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/DSMap_V15_with_shielding_no_endcap',
        input_type='csv', field_map_version='Mau13')
    data_maker.do_basic_modifications(-3.896, descale=True)
    '''
    # helicalc (PS+TS, DS helicalc coils, interlayer connects, and busbars)
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars.pkl',
        input_type='pkl', field_map_version='helicalc_coils')
    data_maker.do_basic_modifications(-3.904, descale=False)
    '''
    # Cartesian grid for validation
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_V13_DSCartVal_Helicalc_All_Coils_All_Busbars.pkl',
        input_type='pkl', field_map_version='helicalc_coils')
    data_maker.do_basic_modifications(-3.904, descale=False)
    '''
    data_maker.make_dump('.Mu2E')
    print(data_maker.data_frame.head())
    print(data_maker.data_frame.tail())
    '''
    # helicalc (PS+TS, DS helicalc coils)
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils.pkl',
        input_type='pkl', field_map_version='helicalc_coils')
    data_maker.do_basic_modifications(-3.904, descale=False)
    
    data_maker.make_dump('.Mu2E')
    print(data_maker.data_frame.head())
    print(data_maker.data_frame.tail())

    # helicalc (busbars)
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Busbars_Only.pkl',
        input_type='pkl', field_map_version='helicalc_coils')
    data_maker.do_basic_modifications(-3.904, descale=False)
    
    data_maker.make_dump('.Mu2E')
    print(data_maker.data_frame.head())
    print(data_maker.data_frame.tail())

    # helicalc (fine grid, PS+TS, DS helicalc coils, interlayer connects, and busbars)
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFine_Helicalc_All_Coils_All_Busbars.pkl',
        input_type='pkl', field_map_version='helicalc_coils')
    data_maker.do_basic_modifications(-3.904, descale=False)
    
    data_maker.make_dump('.Mu2E')
    print(data_maker.data_frame.head())
    print(data_maker.data_frame.tail())

    # DS only, ideal solenoid (SolCalc)
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_SolCalc_DS_Only.pkl',
        input_type='pkl', field_map_version='helicalc_coils')
    data_maker.do_basic_modifications(-3.904, descale=False)

    data_maker.make_dump('.Mu2E')
    print(data_maker.data_frame.head())
    print(data_maker.data_frame.tail())
    
    # DS+TS+PS, ideal solenoid (SolCalc)
    data_maker = DataFrameMaker(
        mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_SolCalc_All_Coils.pkl',
        input_type='pkl', field_map_version='helicalc_coils')
    data_maker.do_basic_modifications(-3.904, descale=False)

    data_maker.make_dump('.Mu2E')
    print(data_maker.data_frame.head())
    print(data_maker.data_frame.tail())

    '''
