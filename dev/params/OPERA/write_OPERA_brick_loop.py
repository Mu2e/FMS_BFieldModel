# write_OPERA_brick_loop.py
# write single loop OPERA conductor file using 20 node bricks (BR20)
# all length units are specified in meters

import numpy as np

# file
filename = 'BR20_single_loop.cond'
label = 'BR2_single_loop'

# get number of bricks/turn and opening angle of each brick
N_bricks = 10
dphi_deg = 360/N_bricks # deg
dphi = dphi_deg*np.pi/180 # rad

# OPERA things
tolerance = 1e-5
symmetry = 1
irxy = 0
iryz = 0
irzx = 0
phi1 = 0
theta1 = 0
psi1 = 0
xcen1 = 0
ycen1 = 0
zcen1 = 0
xcen2 = 0
ycen2 = 0
zcen2 = 0
theta2 = 0
psi2 = 0

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

# brick nodes
xp = np.zeros(20)
yp = np.zeros(20)
zp = np.zeros(20)

# determine node coordinates
# FACE 1
xp[0] = rho0
yp[0] = 0
zp[0] = zeta0

xp[1] = xp[0]
yp[1] = 0
zp[1] = zeta1

xp[2] = rho1
yp[2] = 0
zp[2] = zp[1]

xp[3] = xp[2]
yp[3] = 0
zp[3] = zp[0]

# FACE 2
xp[4] = rho0*np.cos(dphi)
yp[4] = rho0*np.sin(dphi)
zp[4] = zeta0

xp[5] = xp[4]
yp[5] = yp[4]
zp[5] = zeta1

xp[6] = rho1*np.cos(dphi)
yp[6] = rho1*np.sin(dphi)
zp[6] = zp[5]

xp[7] = xp[6]
yp[7] = yp[6]
zp[7] = zp[4]

# MID-EDGE NODES
# face 1
# bw 0 and 1
xp[8] = xp[0]
yp[8] = yp[0]
zp[8] = 0.5*(zeta0+zeta1)

# bw 1 and 2
xp[9] = 0.5*(rho0+rho1)
yp[9] = yp[0]
zp[9] = zp[1]

# bw 2 and 3
xp[10] = xp[2]
yp[10] = yp[0]
zp[10] = zp[8]

# bw 3 and 0
xp[11] = xp[9]
yp[11] = yp[0]
zp[11] = zp[0]

# between faces
# bw 0 and 4
xp[12] = rho0*np.cos(dphi/2)
yp[12] = rho0*np.sin(dphi/2)
zp[12] = zeta0

# bw 1 and 5
xp[13] = rho0*np.cos(dphi/2)
yp[13] = rho0*np.sin(dphi/2)
zp[13] = zeta1

# bw 2 and 6
xp[14] = rho1*np.cos(dphi/2)
yp[14] = rho1*np.sin(dphi/2)
zp[14] = zeta1

# bw 3 and 7
xp[15] = rho1*np.cos(dphi/2)
yp[15] = rho1*np.sin(dphi/2)
zp[15] = zeta0

# FACE 2
# bw 4 and 5
xp[16] = rho0*np.cos(dphi)
yp[16] = rho0*np.sin(dphi)
zp[16] = 0.5*(zeta0+zeta1)

# bw 5 and 6
xp[17] = 0.5*(rho0+rho1)*np.cos(dphi)
yp[17] = 0.5*(rho0+rho1)*np.sin(dphi)
zp[17] = zeta1

# bw 6 and 7
xp[18] = rho1*np.cos(dphi)
yp[18] = rho1*np.sin(dphi)
zp[18] = 0.5*(zeta0+zeta1)

# bw 7 and 4
xp[19] = 0.5*(rho0+rho1)*np.cos(dphi)
yp[19] = 0.5*(rho0+rho1)*np.sin(dphi)
zp[19] = zeta0

# write to file
with open(filename, 'w') as f:
    # header
    f.write('CONDUCTOR\n')
    # loop through bricks
    for brick in range(N_bricks):
        phi2 = phi1 + brick * dphi_deg # loops bricks around
        f.write('DEFINE BR20\n')
        f.write(f'{xcen1:9.5f} {ycen1:9.5f} {zcen1:9.5f} {phi1:4.1f} {theta1:4.1f} {psi1:4.1f}\n')
        f.write(f'{xcen2:9.5f} {ycen2:9.5f} {zcen2:9.5f}\n')
        f.write(f'{theta2:4.1f} {phi2:4.1f} {psi2:4.1f}\n')
        # loop through nodes
        for x,y,z in zip(xp,yp,zp):
            f.write(f'{x:10.6f} {y:10.6f} {z:10.6f}\n')
        f.write(f"{j:8.4e} {symmetry:3d} '{label}'\n")
        f.write(f'{irxy:2d} {iryz:2d} {irzx:2d}\n')
        f.write(f'{tolerance:8.4e}\n')
    f.write('QUIT')
