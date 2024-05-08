# for reading in a OPERA conductor file and gettting brick node locations
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from helicalc import helicalc_dir

class Bricks(object):
    def __init__(self, filedir=helicalc_dir+'dev/params/OPERA/', filename='DS8_helix_V13.cond'):
        self.filedir = filedir
        self.filename = filename
        if filename[:2] == 'DS':
            self.solenoid = 'DS'
            if filename[2] == '_':
                self.geom_type = 'bus'
                self.solenoid_num = -1
            else:
                self.geom_type = 'helix'
                self.solenoid_num = int(filename[2:filename.index('_')])
        else:
            if 'PS' in filename:
                self.solenoid = 'PS_TS'
            else:
                self.solenoid = 'PS_TS_DS'
            self.geom_type = 'ideal'
            self.solenoid_num = -2
        with open(filedir+filename,'r') as f:
            self.filelines = [line[:-1] for line in f.readlines()]
        self.index_brick_starts = [i for i,l in enumerate(self.filelines) if l == 'DEFINE BR20']+[-1]
        self.bricks_lines = [self.filelines[self.index_brick_starts[i]:self.index_brick_starts[i+1]] for i in range(len(self.index_brick_starts)-1)]
        self.N_bricks = len(self.bricks_lines)
        self.pos0s = np.array([[[float(i) for i in self.bricks_lines[k][j].split(' ') if i != ''] for j in range(4, 24)] for k in range(len(self.bricks_lines))])
        self.xcen1s, self.ycen1s, self.zcen1s, self.phi1s, self.theta1s, self.psi1s = np.array([[float(i) for i in self.bricks_lines[k][1].split(' ') if i != ''] for k in range(len(self.bricks_lines))]).T
        self.xcen2s, self.ycen2s, self.zcen2s = np.array([[float(i) for i in self.bricks_lines[k][2].split(' ') if i != ''] for k in range(len(self.bricks_lines))]).T
        self.theta2s, self.phi2s, self.psi2s = np.array([[float(i) for i in self.bricks_lines[k][3].split(' ') if i != ''] for k in range(len(self.bricks_lines))]).T
        self.js, self.symmetrys, self.labels = np.array([[i for i in self.bricks_lines[k][24].split(' ') if i != ''] for k in range(len(self.bricks_lines))]).T
        self.js = self.js.astype(np.float)
        self.symmetrys = self.symmetrys.astype(np.int)
        self.irxys, self.iryzs, self.irzxs = np.array([[float(i) for i in self.bricks_lines[k][25].split(' ') if i != ''] for k in range(len(self.bricks_lines))]).T
        self.tolerances = np.array([[float(i) for i in self.bricks_lines[k][26].split(' ') if i != ''] for k in range(len(self.bricks_lines))]).flatten()
        self.df_brick_conds = pd.DataFrame({'xcen1':self.xcen1s, 'ycen1':self.ycen1s, 'zcen1':self.zcen1s, 'xcen2':self.xcen2s, 'ycen2':self.ycen2s, 'zcen2':self.zcen2s,
                                            'phi1':self.phi1s, 'theta1':self.theta1s, 'psi1':self.psi1s, 'phi2':self.phi2s, 'theta2':self.theta2s, 'psi2':self.psi2s,
                                            'j':self.js, 'symmetry':self.symmetrys, 'label':self.labels,
                                            'irxy':self.irxys, 'iryz':self.iryzs, 'irzx':self.irzxs, 'tolerance':self.tolerances})
        self.pos_bricks_global = []
        for i, zips in enumerate(zip(self.pos0s, np.radians(self.phi2s))):
            xyz, phi2 = zips
            # CHECK ANGLE SIGN
            # rot = Rotation.from_euler('Z',-phi2)
            rot = Rotation.from_euler('Z',phi2)
            xyz_rot = rot.apply(xyz)
            xyz_rot[:,0] += self.xcen1s[i] + self.xcen2s[i]
            xyz_rot[:,1] += self.ycen1s[i] + self.ycen2s[i]
            xyz_rot[:,2] += self.zcen1s[i] + self.zcen2s[i]
            self.pos_bricks_global.append(xyz_rot)
        self.pos_bricks_global = np.array(self.pos_bricks_global).reshape(-1,3)
        # self.pos_bricks_global[:,0] += self.xcen1s + self.xcen2s
        # self.pos_bricks_global[:,1] += self.ycen1s + self.ycen2s
        # self.pos_bricks_global[:,2] += self.zcen1s + self.zcen2s



