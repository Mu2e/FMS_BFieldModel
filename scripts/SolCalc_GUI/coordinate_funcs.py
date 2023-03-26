# main functionality is going between z and s

# found via circle fit to TS1 coil centers
z0_TS2 = -2.9286

def s_PS(z):
    return z + (-5.58 - (z0_TS2))

def z_PS(s):
    return s - (-5.58 - (z0_TS2))
