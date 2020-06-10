############################################################################
# generate visual field coordinates of the rostral boundary of visual areas
###########################################################################

import numpy as np
from scipy.special import erfinv

##### Geometry of marmoset V1 and V2

# We need a function that takes a cortical distance x (in mm), and translates it to visual field eccentricity
# To derive this function, start with the linear cortical magnification function (CMF), take integral, and then calculate the inverse function

# According to Chaplin et al. (2012) J. Comp. Neurol., the CMF of marmoset V1 is
#     e^(-0.0065*ecc) * (5.52 / (ecc + 0.71))
# I approximated it by
#     1.55 * e^(0.7 - 0.57*log(ecc) - 0.098*log(ecc)^2)

def x2eccV1(x):
    return np.exp(2.19388 + 3.19438 * erfinv(-0.953486 + 0.0706123 * x))

# The CMF of marmoset V2 is from Rosa, Fritsches & Elston (1997), J. Comp. Neurol.
# x goes from 0.0 to 8mm, eccentricity goes from 0.8deg to 10deg
def x2eccV2_for_DM(x):
    return np.exp(1.7162 + 3.6761 * erfinv(-0.306111 + 0.0923636 * x))

# x goes from 0.0 to 10, eccentricity goes from 0.2deg to 8.5deg
def x2eccV2(x):
    return np.exp(1.7162 + 3.6761 * erfinv(-0.794 + 0.0923636 * x))


##### sample retintopy along visual area boundary

def ecc2coords(ecc, meridian, vf, tilt=0.0):
    """
    Given an eccentricity (in degree), return a visual field coordindate
    :meridian:      Vertical or horizontal meridian (VM or HM)
    :vf:            'upper' - upper quadrant; 'lower' - lower quadrant (only for VM)
    :tilt:          aplly a tilt (in degree) to the horizontal meridian
    """
    assert meridian in ['VM', 'HM']
    assert vf in ['lower', 'upper']

    deg = 1.0/360.0*2.0*np.pi
    
    if meridian == 'VM':
        if vf == 'upper':
            return [0.0, ecc]
        else:
            return [0.0, -1.0 * ecc]
    else:
        if vf == 'upper':
            return [ecc * np.cos( 1.0*tilt*deg), ecc * np.sin( 1.0*tilt*deg)]
        else:
            return [ecc * np.cos(-1.0*tilt*deg), ecc * np.sin(-1.0*tilt*deg)]

def generate_VM(boundary_len, div, branch, x2ecc):
    """
    The return array goes from ventral to dorsal. index 0 is the most ventral point

    :boundary_len: the length of the boundary in mm
    :div:          divide the booundary into div divisions
    :branch:       'v' - ventral; 'd' - dorsal; 'vd' - ventral and dorsal
    """
    assert branch in ['v', 'd', 'vd']

    if branch=='vd':
        l = boundary_len / 2.0
        d = int(div / 2)
    else:
        l = boundary_len
        d = div

    zz = []
    if branch in ['v', 'vd']:
        # the ventral branch, upper field representation
        for x in np.linspace(0.0, l, d):
            zz.append(ecc2coords(x2ecc(x), 'VM', 'upper'))
        zz.reverse()

    if branch in ['d', 'vd']:
        # the dorsal branch, lower field representation
        for x in np.linspace(0.0, l, d):
            zz.append(ecc2coords(x2ecc(x), 'VM', 'lower'))

    return np.array(zz)

def generate_HM(boundary_len, div, branch, x2ecc, tilt=10.0):
    """
    The return array goes from ventral to dorsal. index 0 is the most ventral point

    :boundary_len: the length of the boundary in mm
    :div:          divide the booundary into div divisions
    :branch:       'v' - ventral; 'd' - dorsal; 'vd' - ventral and dorsal
    :tilt:         Introduce a tilt to the horizontal meridian. in degree 
    """
    assert branch in ['v', 'd', 'vd']

    if branch=='vd':
        l = boundary_len / 2.0
        d = int(div / 2)
    else:
        l = boundary_len
        d = div

    zz = []
    if branch in ['v', 'vd']:
        # the ventral branch, upper field representation
        for x in np.linspace(0.0, l, d):
            zz.append(ecc2coords(x2ecc(x), 'HM', 'upper', tilt=tilt))
        zz.reverse()
    
    if branch in ['d', 'vd']:
        # the dorsal branch, lower field representation
        for x in np.linspace(0.0, l, d):
            zz.append(ecc2coords(x2ecc(x), 'HM', 'lower', tilt=tilt))

    return np.array(zz)

def generate_boundary_retinotopy(boundary_len, div, meridian, branch, x2ecc, filename="", tilt=10.0):
    assert meridian in ['VM', 'HM']

    if meridian == 'VM':
        res = generate_VM(boundary_len, div, branch, x2ecc)
    else:
        res = generate_HM(boundary_len, div, branch, x2ecc, tilt=tilt)

    if filename:
        np.savetxt(filename, res, fmt='%f', delimiter=",")
        
    return res

def generate_V1_boundary_retinotopy(boundary_len, div, branch='vd', filename=""):
    return generate_boundary_retinotopy(boundary_len, div, 'VM', branch, x2eccV1, filename=filename)

def generate_V2_boundary_retinotopy(boundary_len, div, branch='vd', filename="", tilt=10.0):
    return generate_boundary_retinotopy(boundary_len, div, 'HM', branch, x2eccV2, filename=filename, tilt=tilt)

def generate_V2_boundary_retinotopy_for_DM(boundary_len, div, filename=""):
    # this is the code I used for the original simulatio. It can be replaced with the more generalized version, but I am keeping it
    zz = []
    for x in np.linspace(0.0, boundary_len, div):
        # horizontal meridian
        coords = [x2eccV2_for_DM(x), 0.0]
        zz.append(coords)
        
    if filename:
        np.savetxt(filename, zz, fmt='%f', delimiter=",")

    return np.array(zz)

def test_V1():
    boundary_len = 19.0  # in milimeter
    map_h = 60           # this is on the boundary
    return generate_V1_boundary_retinotopy(boundary_len, map_h, branch='vd')

def test_V2():
    boundary_len = 11.0 * 2.0 # one branch is 11 mm
    map_h = 30
    return generate_V2_boundary_retinotopy(boundary_len, map_h, branch='vd', filename='testV2.csv', tilt=10.0)
