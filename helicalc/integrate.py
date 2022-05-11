import numpy as np
import torch as tc

# main integrator
# 3D needed for helicalc
def trapz_3d(xs, ys, zs, integrand_xyz, int_func=tc.trapz):
    return int_func(int_func(int_func(integrand_xyz, axis=-1, x=zs), axis=-1, x=ys), axis=-1, x=xs)

# 2D needed for solcalc. Cylindrical integration
def trapz_2d(rs, zs, integrand_rz, int_func=tc.trapz):
    return int_func(int_func(integrand_rz, axis=-1, x=zs), axis=-1, x=rs)

# helpful functions
# maybe move into CoilIntegrator class? FIXME!
## HELIX
# wind in +phi
'''
def rx(rho, COSPHI, x):
    return x - rho*COSPHI
def ry(rho, SINPHI, hel, y):
    return y - hel*rho*SINPHI
def rz(zeta, phi, phi0, pitch_bar, L, t_gi, z_start, z):
    #return z - (zeta + (phi-phi0) * pitch_bar - L/2 + t_gi)
    return z - (z_start + zeta + (phi-phi0) * pitch_bar + t_gi)
'''
# wind in -phi
def rx(rho, COSPHI, x):
    return x - rho*COSPHI
def ry(rho, SINPHI, y):
    return y - rho*SINPHI
def rz(zeta, phi, phi0, pitch_bar, t_gi, z_start, z):
    #return z - (zeta + (phi-phi0) * pitch_bar - L/2 + t_gi)
    ##return z - (z_start + zeta + (phi-phi0) * pitch_bar + t_gi)
    # do we need "lib.abs"?
    return z - (z_start + zeta + abs(phi-phi0) * pitch_bar + t_gi)
# REF from plotting
# zs = z_start + np.abs(phis-phi0)/(2*np.pi) * df_.pitch

def helix_integrand_Bx(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    #return (rho * COSPHI * RZ - hel * pitch_bar * RY) / R2_32 # for +helicity
    ###return (hel * rho * COSPHI * RZ - pitch_bar * RY) / R2_32 # correct
    # test rho in h/2pi term
    return rho * (hel * COSPHI * RZ - pitch_bar * RY) / R2_32
def helix_integrand_By(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    #return (hel * rho * SINPHI * RZ + hel * pitch_bar * RX) / R2_32
    #return hel * (rho * SINPHI * RZ + pitch_bar * RX) / R2_32 # for +helicity
    # return (hel * rho * SINPHI * RZ + pitch_bar * RX) / R2_32 # correct
    # test rho in h/2pi term
    return rho * (hel * SINPHI * RZ + pitch_bar * RX) / R2_32

def helix_integrand_Bz(RX, RY, RZ, R2_32, rho, COSPHI, SINPHI, hel, pitch_bar, L):
    #return (-hel * rho * SINPHI * RY - rho * COSPHI * RX) / R2_32
    #return -rho * (hel * SINPHI * RY + COSPHI * RX) / R2_32 # for +helicity
    return - hel * rho * (SINPHI * RY + COSPHI * RX) / R2_32 # correct

# minimal terms
def rx_min(RHOCOSPHI, x):
    return x - RHOCOSPHI
def ry_min(RHOSINPHI, y):
    return y - RHOSINPHI
def rz_min(ZTERM, z):
    #return z - (zeta + (phi-phi0) * pitch_bar - L/2 + t_gi)
    return z - ZTERM

def helix_integrand_Bx_min(RX, RY, RZ, R2_32, HRHOCOSPHI, HRHOSINPHI, pitch_bar):
    return (HRHOCOSPHI * RZ - pitch_bar * RY) / R2_32
def helix_integrand_By_min(RX, RY, RZ, R2_32, HRHOCOSPHI, HRHOSINPHI, pitch_bar):
    return (HRHOSINPHI * RZ + pitch_bar * RX) / R2_32

def helix_integrand_Bz_min(RX, RY, RZ, R2_32, HRHOCOSPHI, HRHOSINPHI, pitch_bar):
    return - (HRHOSINPHI * RY + HRHOCOSPHI * RX) / R2_32

####

## CIRCULAR ARC BAR
# note in OPERA, orientation is with field in -x direction
# and arc phi0 is at origin, rather than the center of the arc at origin

def rx_arc(x0, xp):
    return x0 - xp
def ry_arc(y0, rho0, RHO, COSPHI):
    return y0 + RHO*COSPHI - rho0
def rz_arc(z0, RHO, SINPHI):
    return z0 - RHO*SINPHI

# kept order so it appears similar to the standard orientation
def arc_integrand_By(RHO, SINPHI, COSPHI, RX, RY, RZ, R2_32):
    return RHO*RX*COSPHI/R2_32
def arc_integrand_Bz(RHO, SINPHI, COSPHI, RX, RY, RZ, R2_32):
    return -RHO*RX*SINPHI/R2_32
def arc_integrand_Bx(RHO, SINPHI, COSPHI, RX, RY, RZ, R2_32):
    return RHO*(RZ*SINPHI - RY*COSPHI)/R2_32


####

## STRAIGHT BAR
def rx_str(x0, xp):
    return x0 - xp
def ry_str(y0, yp):
    return y0 - yp
def rz_str(z0, zp):
    return z0 - zp

def straight_integrand_Bx(RX, RY, R2_32):
    return -RY/R2_32
def straight_integrand_By(RX, RY, R2_32):
    return RX/R2_32
