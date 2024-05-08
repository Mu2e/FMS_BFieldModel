# write_OPERA_solenoid_loop.py
# write single loop OPERA conductor file using solenoid geometry (SOLENOID)
# all length units are specified in meters

import numpy as np

# file
filename = 'SOLENOID_single_loop.cond'
label = 'SOLENOID_single_loop'

# get number of bricks/turn and opening angle of each brick
# N_bricks = 10
# dphi_deg = 360/N_bricks # deg
# dphi = dphi_deg*np.pi/180 # rad

# OPERA things
tolerance = 1e-5
symmetry = 1
irxy = 0
iryz = 0
irzx = 0
phi1 = 0
theta1 = 0
psi1 = 0
# Euler angles rotate axes to proper orientation
phi2   =  90  # deg
theta2 = -90  # deg
psi2   =  90  # deg
xcen1 = 0
ycen1 = 0
zcen1 = 0
xcen2 = 0
ycen2 = 0
zcen2 = 0

# Hand coded geometry (could add input file later)
# to match DS-8 radius
Ri = 1.050 # m
h_sc = 5.25e-3 # m  -- radial thickness
w_sc = 2.34e-3 # m  -- thickness in z
I = 6114 # Amps
j = I/(h_sc*w_sc) # Amps / m^2

h_cable = 20.1e-3 # m
t_gi = 0.5e-3 # m
t_ci = 0.233e-3 # m
rho0 = Ri + (h_cable - h_sc)/2. + t_gi + t_ci
rho1 = rho0 + h_sc

zeta0 = -w_sc/2 # center conductor in z
zeta1 = zeta0 + w_sc

x1, y1, x2, y2 = rho0, zeta0, rho0, zeta1
x3, y3, x4, y4 = rho1, zeta1, rho1, zeta0
cu1, cu2, cu3, cu4 = (0, 0, 0, 0)

# write to file
with open(filename, 'w') as f:
    # header
    f.write('CONDUCTOR\n')
    f.write('DEFINE GSOLENOID\n')
    f.write(f'{xcen1:9.5f} {ycen1:9.5f} {zcen1:9.5f} {phi1:4.1f} {theta1:4.1f} {psi1:4.1f}\n')
    f.write(f'{xcen2:9.5f} {ycen2:9.5f} {zcen2:9.5f}\n')
    f.write(f'{theta2:4.1f} {phi2:4.1f} {psi2:4.1f}\n')
    f.write(f'{x1:10.6f} {y1:10.6f} {x2:10.6f} {y2:10.6f}\n')
    f.write(f'{x3:10.6f} {y3:10.6f} {x4:10.6f} {y4:10.6f}\n')
    f.write(f'{cu1:3d} {cu2:3d} {cu3:3d}\n')
    f.write(f"{j:8.4e} {symmetry:3d} '{label}'\n")
    f.write(f'{irxy:2d} {iryz:2d} {irzx:2d}\n')
    f.write(f'{tolerance:8.4e}\n')
    f.write('QUIT')
