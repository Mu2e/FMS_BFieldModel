import pandas as pd
import numpy as np
import six.moves.cPickle as pkl
from mu2e.dataframeprod import DataFrameMaker
from lmfit import Parameters, Model
from mu2e import mu2e_ext_path
from tqdm import tqdm

probes = ['BP1','BP2','BP3','BP4','BP5','SP1','SP2','SP3']

# Define a coordinate transformation -- X', Y', Z' in terms of Psi, Theta, Phi, X0, Y0, Z0, and X, Y, Z
def rot_z(angle):
    return np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
def rot_x(angle):
    return np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
def euler_transform(Psi,Theta,Phi,vec,inverse=False):
    if inverse:
        return np.linalg.multi_dot([rot_z(float(-Phi)),rot_x(float(-Theta)),rot_z(float(-Psi)),vec])
    else:
        return np.linalg.multi_dot([rot_z(float(Psi)),rot_x(float(Theta)),rot_z(float(Phi)),vec])

# Generic class for position uncertainties
# These are initialized with a nominal map (in 'raw helicalc output' format) and need an accessor
# method to get the helicalc output from running at the shifted points
class posunc:
    def __init__(self, nominal_map_file):
        self.nominal_map = pd.read_pickle(nominal_map_file)
        # Shift to DS coordinate frame and define Phi, without reordering points
        self.nominal_map.eval('X = X+3.904',         inplace=True)
        self.nominal_map.eval('Phi = arctan2(Y,X)',  inplace=True)
        # Update phi vals for R=0 (same procedure as dataframeprod)
        phi_vals = np.unique(np.array(self.nominal_map.Phi).round(decimals=9))
        if len(phi_vals) != 16:
            print(phi_vals)
            exit()
        z_steps_r0 = np.unique(self.nominal_map[(self.nominal_map.HP=='SP1')]['Z'])
        updated = 0
        for z_step in z_steps_r0:
            if self.nominal_map[(self.nominal_map.HP=='SP1') & (self.nominal_map.Z==z_step)].shape[0] > 1:
                if np.var(self.nominal_map[(self.nominal_map.HP=='SP1') & (self.nominal_map.Z==z_step)]['Phi']) < 1e-10:
                    if self.nominal_map[(self.nominal_map.HP=='SP1') & (self.nominal_map.Z==z_step)].shape[0] == len(phi_vals):
                        self.nominal_map.loc[(self.nominal_map.HP=='SP1') & (self.nominal_map.Z==z_step),'Phi'] = phi_vals
                        updated += 1
                    else:
                        print(f"Cannot fill Phi for R==0, Z=={z_step} -- {self.nominal_map[(self.nominal_map.HP=='SP1') & (self.nominal_map.Z==z_step)].shape[0]} lines but {len(phi_vals)} phi values")
        if updated != 0:
            print(f'Updated phi values at R=0 for {updated} z steps ({len(z_steps_r0)} z steps total)')

        self.nominal_map = self.nominal_map.round(9)
        # Dummy value here, overwrite
        self.shifted_field_file = ''

    # Merge nominal position coordinates with field values at shifted points -- output dataframe
    # will be passed to hallprober
    def get_shifted_field(self):
        df = pd.read_pickle(self.shifted_field_file)
        # We are going to use offset X from nominal here, rather than applying the offset in DataFrameMaker.
        for col in ['X','Y','Z']:
            df.loc[:,col] = self.nominal_map[col]
        # Convert with DataFrameMaker
        # Note field_map_version can be any value other than Mau9, Mau10, GA01 -- hardcoded processing
        data_maker = DataFrameMaker(self.shifted_field_file,field_map_version="Mau13",input_type='df',input_df=df)
        data_maker.do_basic_modifications()
        data_maker.make_dump('.Mu2E')
        return data_maker.data_frame
        
# Class for position uncertainties of propeller plane in global coordinate system due to uncertainties in
# positions of laser reflectors
class laserunc(posunc):
    def __init__(self,nominal_map_file=mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars.pkl'):
        super().__init__(nominal_map_file)
        self.param_names = ['dPsi_BP', 'dPsi_SP', 'dTheta_BP', 'dTheta_SP', 'dPhi', 'X0_BP', 'X0_SP', 'Y0_BP', 'Y0_SP', 'Z0']
        self.transformations = dict((k,[]) for k in self.param_names+['Z','Phi'])
        self.fit_status = {'Z' : [], 'Phi' : [], 'chisq' : [], 'redchi' : [], 'nfev' : [], 'success' : []}
        self.shifted_field_file = nominal_map_file.replace('.',f'_LaserUnc.')

        # Coordinates of reflectors in propeller plane (origin at center of large propeller,
        # x axis through long propeller blade, positive x along arm with HP at tip)
        # There are 4 reflectors per propeller
        df_dict = {'X'  : [0.724, -0.724, 0.0,    0.0,   0.067175,  0.067175, -0.067175, -0.067175],
                   'Y'  : [0.0,    0.0,   0.349, -0.349, 0.067175, -0.067175, -0.067175,  0.067175],
                   'Z'  : [0.0,    0.0,   0.0,    0.0,  -1.335,    -1.335,    -1.335,    -1.335   ],
                   'NR' : ['BPR1','BPR2','BPR3', 'BPR4','SPR1',    'SPR2',    'SPR3',    'SPR4']}
        self.df_prop = pd.DataFrame(data=df_dict)
        self.z_steps   = list(self.nominal_map[self.nominal_map['HP'].str.contains('BP')].Z.unique())
        self.phi_steps = list(self.nominal_map[self.nominal_map['HP'].str.contains('BP')].Phi.unique())

    # Output should look like X1,Y1,Z1,X2,Y2,Z2...XN,YN,ZN
    def get_transformed_field(self,reflectors,inverse=False):

        reflector_list = list(np.array(reflectors))
        
        def transformed_field(x, y, z, dPsi_BP, dPsi_SP, dTheta_BP, dTheta_SP, dPhi, X0_BP, X0_SP, Y0_BP, Y0_SP, Z0):
            rows = []
            for ip,point in enumerate(zip(x,y,z)):
                coord = np.array(point)
                if 'SP' in reflector_list[ip]:
                    coord_transform = euler_transform(dPsi_SP,dTheta_SP,dPhi,coord,inverse)
                    coord_transform += np.array([X0_SP,Y0_SP,Z0]).ravel()
                else:
                    coord_transform = euler_transform(dPsi_BP,dTheta_BP,dPhi,coord,inverse)
                    coord_transform += np.array([X0_BP,Y0_BP,Z0]).ravel()
                rows.append(coord_transform)
            result = np.stack(rows).ravel()
            return result
        return transformed_field

    # Generate reflector positions as a function of z and phi, shift according to laser position
    # uncertainties, and fit to extract origin and Euler angles of best description of propeller
    # plane. These parameters will later be used to transform the Hall probe field map. We use
    # independent rotations and translations for big and small propeller, except for Z (no compression
    # in rod) and Phi (no torsion in rod)
    def get_transformations(self):
        print('Running fits...')
        for z_step in tqdm(self.z_steps):
            for phi_step in self.phi_steps:
                # Get the reflector positions in global coordinates for this step
                df_nom = self.df_prop.copy()
                df_nom['Z'] = df_nom['Z']+z_step
                x = df_nom.X
                y = df_nom.Y
                df_nom['X'] = x*np.cos(phi_step)-y*np.sin(phi_step)
                df_nom['Y'] = y*np.cos(phi_step)+x*np.sin(phi_step)

                # Now apply randomly sampled X/Y/Z uncertainties to each reflector position, with
                # the correct magnitude for this position
                df_shift = df_nom.copy()
                shifts = np.random.randn(3,9)*1e-6
                xyscale = 15.+6.*(17.601-np.array(df_shift.Z))
                zscale = 10.
                df_shift.loc[:,'X'] += shifts[0]*xyscale
                df_shift.loc[:,'Y'] += shifts[1]*xyscale
                df_shift.loc[:,'Z'] += shifts[2]*zscale

                # Transform into the X1,Y1,Z1,...X9,Y9,Z9 format expected by the fit function
                data = np.array(df_shift[['X','Y','Z']]).ravel()
                # Fit
                field_transformer = self.get_transformed_field(df_nom.NR)
                mod = Model(field_transformer, independent_vars=['x','y','z'])
                params = Parameters()
                for param in self.param_names:
                    params.add(param,value=0,min=-0.1,max=0.1)
                result = mod.fit(data, x=df_nom.X, y=df_nom.Y, z=df_nom.Z, params=params)

                fitted_params = result.params

                self.transformations['Z'].append(z_step)
                self.transformations['Phi'].append(phi_step)
                for pname, par in fitted_params.items():
                    self.transformations[pname].append(par.value)

                self.fit_status['Z'].append(z_step)
                self.fit_status['Phi'].append(z_step)
                self.fit_status['chisq'].append(result.chisqr)
                self.fit_status['redchi'].append(result.redchi)
                self.fit_status['nfev'].append(result.nfev)
                self.fit_status['success'].append(result.success)

        self.transformations = pd.DataFrame(data=self.transformations)
        self.fit_status = pd.DataFrame(data=self.fit_status)

    # Get points in map corresponding to given Z, Phi step
    # Variable tolerance to pick up cases where Z has been shifted
    def get_slice(self,field_map,z_step,iphi,atol=1e-08):
        # Phi gives the angle of the positive x axis of the propeller plane w.r.t. the global x, e.g. the Phi position of BP1/3/5
        iphi_bp1 = iphi
        iphi_bp2 = iphi+8 if iphi < 8 else iphi-8
        # Going by the technical drawings, SP2 is at a pi/4 angle from BP5, e.g. 2 steps
        iphi_sp1 = iphi+2  if iphi < 14 else iphi-14
        iphi_sp2 = iphi+10 if iphi < 6  else iphi-6
        bp1 = np.logical_and.reduce([np.array(field_map['HP'].isin(['BP1','BP3','BP5'])), np.array(field_map['Phi'] == self.phi_steps[iphi_bp1]),
                                     np.isclose(np.array(field_map['Z']),z_step,atol=atol)])
        bp2 = np.logical_and.reduce([np.array(field_map['HP'].isin(['BP2','BP4'])),       np.array(field_map['Phi'] == self.phi_steps[iphi_bp2]),
                                     np.isclose(np.array(field_map['Z']),z_step,atol=atol)])
        sp1 = np.logical_and.reduce([np.array(field_map['HP'].isin(['SP1','SP2'])),       np.array(field_map['Phi'] == self.phi_steps[iphi_sp1]),
                                     np.isclose(np.array(field_map['Z']),z_step-1.335,atol=atol)])
        sp2 = np.logical_and.reduce([np.array(field_map['HP'].isin(['SP3'])),             np.array(field_map['Phi'] == self.phi_steps[iphi_sp2]),
                                     np.isclose(np.array(field_map['Z']),z_step-1.335,atol=atol)])
        return np.logical_or.reduce([bp1,bp2,sp1,sp2])
        
    # Once transformation per z, phi step has been defined, apply to nominal field map
    def generate_shifted_grid(self):
        shifted_map = self.nominal_map.copy()
        for z_step in tqdm(self.z_steps):
            # Hardcoding for 16 phi steps -- TODO make this more flexible?
            for iphi in range(16):
                filt = self.get_slice(shifted_map,z_step,iphi)
                chunk = shifted_map[filt]
                trans = pd.DataFrame(data=self.transformations)
                trans = self.transformations[np.array(self.transformations.Z==z_step) & np.array(self.transformations.Phi==self.phi_steps[iphi])]
                field_transformer = self.get_transformed_field(chunk.HP)
                xyz_trans = field_transformer(chunk.X, chunk.Y, chunk.Z, trans['dPsi_BP'], trans['dPsi_SP'], trans['dTheta_BP'], trans['dTheta_SP'], trans['dPhi'], trans['X0_BP'], trans['X0_SP'], trans['Y0_BP'], trans['Y0_SP'], trans['Z0'])
                shifted_map.loc[filt,'X'] = xyz_trans[0::3]
                shifted_map.loc[filt,'Y'] = xyz_trans[1::3]
                shifted_map.loc[filt,'Z'] = xyz_trans[2::3]
        shifted_map.eval('X = X-3.904',inplace=True)        
        shifted_map = shifted_map[['X','Y','Z','HP']]
        pkl.dump(shifted_map, open(f'coord_laserunc.p', "wb"), pkl.HIGHEST_PROTOCOL)

    # Overriding nominal method to also rotate magnetic field
    # The measured field in the rotated frame would be the true field (from helicalc) rotated with
    # the inverse of the propeller rotation
    def get_shifted_field(self):
        df = pd.read_pickle(self.shifted_field_file)
        # Applying nominal coordinates with offset, Phi
        for col in ['X','Y','Z','Phi']:
            df.loc[:,col] = self.nominal_map[col]
        for z_step in self.z_steps:
            for iphi in range(16):
                filt = self.get_slice(df,z_step,iphi,1e-02) # z step size is 5e-02
                chunk = df[filt]
                trans = self.transformations[np.array(self.transformations.Z==z_step) & np.array(self.transformations.Phi==self.phi_steps[iphi])]
                # We want to apply the inverse rotation to the magnetic field
                field_transformer = self.get_transformed_field(chunk.HP,True)
                Bxyz_trans = field_transformer(chunk.Bx, chunk.By, chunk.Bz, trans['dPsi_BP'], trans['dPsi_SP'], trans['dTheta_BP'], trans['dTheta_SP'], trans['dPhi'], 0.0, 0.0, 0.0, 0.0, 0.0)
                df.loc[filt,'Bx'] = Bxyz_trans[0::3]
                df.loc[filt,'By'] = Bxyz_trans[1::3]
                df.loc[filt,'Bz'] = Bxyz_trans[2::3]
        # Convert with DataFrameMaker
        # Note field_map_version can be any value other than Mau9, Mau10, GA01 -- hardcoded processing
        data_maker = DataFrameMaker(self.shifted_field_file,field_map_version="Mau13",input_type='df',input_df=df)
        data_maker.do_basic_modifications()
        data_maker.make_dump('.Mu2E')
        return data_maker.data_frame


'''
Position uncertainty due to misalignment of Hall probes on propeller.
Can arbitrarily define the X and Y shifts as the values when the probe is at Phi=0 --> shifts at other phi 
steps are rotations of the original shift 
For now, taking magnitude of uncertainty from FMS-DSFM-01-00 -- 0.02" or 500 um. This is hopefully conservative.
'''
class metunc(posunc):
    def __init__(self, nominal_map_file=mu2e_ext_path+'Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars.pkl', toy=''):
        super().__init__(nominal_map_file)
        self.shifted_field_file = nominal_map_file.replace('.',f'_MetUnc{toy}.')
        
    def generate_shifted_grids(self,ntoys):
        shift_dict = dict([(probe,{'dX' : [], 'dY' : [], 'dZ' : []}) for probe in probes])
        #np.random.seed(0) 
        for itoy in range(ntoys):
            shifted_map = self.nominal_map.copy()
            x_shift_prop = np.random.normal(0, 5e-4, len(probes))
            y_shift_prop = np.random.normal(0, 5e-4, len(probes))
            z_shift_prop = np.random.normal(0, 5e-4, len(probes))
            for ip, probe in enumerate(probes):
                shift_dict[probe]['dX'].append(x_shift_prop[ip])
                shift_dict[probe]['dY'].append(y_shift_prop[ip])
                shift_dict[probe]['dZ'].append(z_shift_prop[ip])
                Phi = shifted_map[shifted_map.HP == probe]['Phi']
                x_shift = x_shift_prop[ip] * np.cos(Phi) - y_shift_prop[ip] * np.sin(Phi)
                y_shift = y_shift_prop[ip] * np.cos(Phi) + x_shift_prop[ip] * np.sin(Phi)
                shifted_map.loc[shifted_map.HP == probe, 'X'] += x_shift
                shifted_map.loc[shifted_map.HP == probe, 'Y'] += y_shift
                shifted_map.loc[shifted_map.HP == probe, 'Z'] += z_shift_prop[ip]
            shifted_map.eval('X = X-3.904',inplace=True)        
            shifted_map = shifted_map[['X','Y','Z','HP']]
            pkl.dump(shifted_map, open(f'coord_metunc_{itoy}.p', "wb"), pkl.HIGHEST_PROTOCOL)
        pkl.dump(shift_dict, open('metunc_shifts.p', "wb"), pkl.HIGHEST_PROTOCOL)

        
'''
Magnetic field uncertainties can be applied by modifying the nominal field map on-the-fly
Since we don't need a two-step process, do this with a single function
'''
def apply_field_unc(syst_unc, nominal_map):

    '''
    Uncertainty in magnetic field strength from Hall probe calibration
    Apply randomly sampled SF to magnetic field at each measurement point
    Currently implemented as a relative 1E-4 uncertainty; can also 
    implement as an absolute uncertainty
    '''
    def calib_magnitude_unc(nominal_map):
        B_sf = np.random.normal(0, 1e-4, len(probes))
        B_sf = B_sf+1.
        print(B_sf)
        try:
            shift_dict = pkl.load(open('calibmagunc_shifts.pkl','rb'))
            for ip,probe in enumerate(probes):
                shift_dict[probe].append(B_sf[ip])
        except:
            shift_dict = dict([(probes[ip],[B_sf[ip]]) for ip in range(len(probes))])
        pkl.dump(shift_dict,open('calibmagunc_shifts.pkl','wb'),pkl.HIGHEST_PROTOCOL)
        for i, probe in enumerate(probes):
            nominal_map.loc[nominal_map.HP == probe, 'Br']   *= B_sf[i]
            nominal_map.loc[nominal_map.HP == probe, 'Bphi'] *= B_sf[i]
            nominal_map.loc[nominal_map.HP == probe, 'Bz']   *= B_sf[i]
        return nominal_map

    '''
    Uncertainty in magnetic field direction from Hall probe calibration
    Implement as an Euler rotation with a random value for each angle per probe
    For now, since the rotation is symmetric (i.e. the three Euler angles are drawn from
    the same normal distribution, no bias for one to be larger) we can arbitrarily apply this
    rotation in the lab frame, rather than in the Hall probe frame. If during the calibration we 
    discover that uncertainties are larger for rotation about a given axis, this will need to be
    adjusted.
    '''
    def calib_angle_unc(nominal_map):
        #np.random.seed(0)
        rot_probe = np.random.randn(9,3)*1e-4
        print(rot_probe)
        try:
            shift_dict = pkl.load(open('calibrotunc_shifts.pkl','rb'))
            for ip,probe in enumerate(probes):
                for ia,angle in enumerate(['psi','theta','phi']):
                    shift_dict[probe][angle].append(rot_probe[ip][ia])
        except:
            shift_dict = dict([(probe,dict([(angle,rot_probe[ip][ia]) for ia,angle in enumerate(['psi','theta','phi'])])) for ip,probe in enumerate(probes)])
        pkl.dump(shift_dict,open('calibrotunc_shifts.pkl','wb'),pkl.HIGHEST_PROTOCOL)
        for ip, probe in enumerate(probes):
            field_vals = nominal_map[nominal_map.HP == probe][['Br','Bphi','Bz']]
            field_vals = field_vals.apply(lambda x: euler_transform(rot_probe[ip][0],rot_probe[ip][1],rot_probe[ip][2],x), axis=1, result_type='broadcast')
            nominal_map.loc[nominal_map.HP == probe,'Br']   = field_vals.Br
            nominal_map.loc[nominal_map.HP == probe,'Bphi'] = field_vals.Bphi
            nominal_map.loc[nominal_map.HP == probe,'Bz']   = field_vals.Bz
        nominal_map.eval('Bx = Br*cos(Phi)-Bphi*sin(Phi)', inplace=True)
        nominal_map.eval('By = Bphi*cos(Phi)+Br*sin(Phi)', inplace=True)
        return nominal_map

    '''
    For now, assuming common temperature offset per Hall probe for all mapping measurements (most conservative)
    Thermistor uncertainty is +-0.2C
    Also assuming temperature is 22.2 C at center of tracker, with gradient of 0.5C/m in y and 0.03C/m in z
    '''
    def temp_unc(nominal_map):
        nominal_map.eval('Btesla=sqrt(Bx**2+By**2+Bz**2)/10000.', inplace=True)
        nominal_map.eval('T=22.2+0.03*(Z-9.05)+0.5*Y', inplace=True)
        nominal_map.eval('Bp=(Btesla-0.935)/0.225', inplace=True)
        nominal_map.eval('Tp=(T-24.75)/7.45', inplace=True)
        nominal_map.eval('dBdT=-(0.225/7.45)*(-7924.32-2246.80*Bp+4*127.18*Tp)/(641187.96-2246.80*Tp+4*2178.13*Bp)',inplace=True)
        # Temperature uncertainty of 0.2 C, fixed offset per HP (conservative)
        dT = np.random.normal(0, 0.2, len(probes))
        print(dT)
        try:
            shift_dict = pkl.load(open('tempunc_shifts.pkl','rb'))
            for ip,probe in enumerate(probes):
                shift_dict[probe].append(dT[ip])
        except:
            shift_dict = dict([(probes[ip],[dT[ip]]) for ip in range(len(probes))])
        pkl.dump(shift_dict,open('tempunc_shifts.pkl','wb'),pkl.HIGHEST_PROTOCOL)
        for i, probe in enumerate(probes):
            nominal_map.loc[nominal_map.HP == probe, 'dBdT'] *= dT[i] # dB/dT is now dB
        nominal_map.eval('SF=1.0+dBdT/Btesla',inplace=True)
        nominal_map.loc[:,'Br']   *= nominal_map.SF
        nominal_map.loc[:,'Bphi'] *= nominal_map.SF
        nominal_map.loc[:,'Bz']   *= nominal_map.SF
        nominal_map = nominal_map.drop(columns=['Btesla','T','Bp','Tp','dBdT','SF'])
        return nominal_map

    if syst_unc.startswith("CalibMagUnc"):
        return calib_magnitude_unc(nominal_map)
    elif syst_unc.startswith("CalibRotUnc"):
        return calib_angle_unc(nominal_map)
    elif syst_unc.startswith("TempUnc"):
        return temp_unc(nominal_map)
    else:
        print('Error -- magnetic field uncertainty must be one of the following: CalibMagUnc, CalibRotUnc, TempUnc)')
        exit()
    
#####################
# Running code here #
#####################

if __name__ == "__main__":

    '''
    myunc = laserunc("/home/sdittmer/data/Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars.pkl")
    myunc.get_transformations()
    with open("transformations.pkl","wb") as f:
        pkl.dump(myunc.transformations, f, pkl.HIGHEST_PROTOCOL)
    with open("fit_status.pkl","wb") as f:
        pkl.dump(myunc.fit_status, f, pkl.HIGHEST_PROTOCOL)
    '''
    #import matplotlib.pyplot as plt
    '''
    transformations = pkl.load(open('transformations.pkl',"rb"))
    fig1, axs1 = plt.subplots(1,2)
    axs1[0].hist(transformations['dPsi_BP'])
    axs1[1].hist(transformations['dPsi_SP'])
    plt.show()
    fig2, axs2 = plt.subplots(1,2)
    axs2[0].hist(transformations['dTheta_BP'])
    axs2[1].hist(transformations['dTheta_SP'])
    plt.show()
    fig3, axs3 = plt.subplots(1,1)
    axs3.hist(transformations['dPhi'])
    plt.show()
    fig4, axs4 = plt.subplots(1,2)
    axs4[0].hist(transformations['X0_BP'])
    axs4[1].hist(transformations['X0_SP'])
    plt.show()
    fig5, axs5 = plt.subplots(1,2)
    axs5[0].hist(transformations['Y0_BP'])
    axs5[1].hist(transformations['Y0_SP'])
    plt.show()
    fig6, axs6 = plt.subplots(1,1)
    axs6.hist(transformations['Z0'])
    plt.show()


    fit_status = pkl.load(open("fit_status.pkl","rb"))
    fig1, ax1 = plt.subplots(1,2)
    ax1[0].hist(fit_status['chisq'])
    ax1[1].hist(fit_status['redchi'])
    plt.show()
    fig2, ax2 = plt.subplots(1,2)
    ax2[0].scatter(fit_status['Z'],fit_status['chisq'])
    ax2[1].scatter(fit_status['Z'],fit_status['redchi'])
    plt.show()
    fig3, ax3 = plt.subplots(1,2)
    ax3[0].hist(fit_status['success'].astype(int))
    ax3[1].hist(fit_status['nfev'])
    plt.show()
    fig4, ax4 = plt.subplots(1,2)
    ax4[0].scatter(fit_status['Z'],fit_status['success'].astype(int))
    ax4[1].scatter(fit_status['Z'],fit_status['nfev'])
    plt.show()

    myunc.transformations = pkl.load(open('transformations.pkl',"rb"))
    #myunc.generate_shifted_grid()

    myunc.get_shifted_field()
    '''
    metuncert = metunc("/home/sdittmer/data/Bmaps/Mu2e_V13_DSCylFMSAll_Helicalc_All_Coils_All_Busbars.pkl")
    metuncert.generate_shifted_grids(10)
