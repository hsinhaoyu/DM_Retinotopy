########################################
# generate landmarks on the visual field 
########################################

import numpy as np
from numpy import linalg as LA
import csv

### generate landmarks on the visual field
def spiral(c, r0, r1, n, shuffleP=True):
    """
        Make a spiral with n points.
        the distance from these points to (0, 0) ranges from r0 to r1
        the density of the dots falls off accoridng to r^c, where r is the distance to (0, 0)
        c should be a negative number
        returns a numpy array
    """
    def f(r):
        return np.sqrt(np.power(r, 1.0-c)/(1.0-c))

    n0 = np.power(r0 * r0 * (1.0-c), 1.0/(1.0-c))
    n1 = np.power(r1 * r1 * (1.0-c), 1.0/(1.0-c))

    lst = []
    for k in range(1, n+1):
        d = f( (n1-n0)/(n-1)*k + n0-(n1-n0)/(n-1) )
        theta = 3.1415926 * (3.0 - np.sqrt(5.0)) * (k-1.0)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        lst.append((x,y))
    res = np.asarray(lst, dtype=np.float64)
    if shuffleP:
        np.random.shuffle(res)
    return res

def generate_RVF_landmarks(c, r0, r1, n, noise=False):
    # generate landmarks on the RIGHT visual field
    # returns a numpy array

    # generate twice as many points because spiral() covers the entire visual field (left and right)
    pts = spiral(c, r0, r1, 2*n)

    # boolean array, find those that are in the right hemifield
    pts_idx = pts[:, 0] >= 0.0

    # make a mask array
    pts_idx = np.stack([pts_idx, pts_idx])
    pts_idx = np.transpose(pts_idx)

    # keep points whose x coordinate is larger or equal to 0
    zz = np.extract(pts_idx, pts)
    # but since the extracted is flattend, I have to reshape it
    zz = zz.reshape([int(zz.shape[0]/2), 2])

    if noise:
        zz = zz + np.random.normal(0.0, 0.1, zz.shape)

    return zz

def generate_vf_landmarks(c, r0, r1, n, vf='full', filename="", noise=False):
    assert vf in ['upper', 'lower', 'full']

    if vf in ['upper', 'lower']:
        NN = n * 2
    else:
        NN = n
    
    pts = generate_RVF_landmarks(c, r0, r1, NN, noise)

    # boolean array, find those that are in the right quadrant
    if vf=='upper':
        pts_idx = pts[:, 1] >= 0.0
    elif vf=='lower':
        pts_idx = pts[:, 1] <= 0.0
    else:
        # this should be all True
        pts_idx = pts[:, 0] >= 0.0
    
    # make a mask array
    pts_idx = np.stack([pts_idx, pts_idx])
    pts_idx = np.transpose(pts_idx)

    zz = np.extract(pts_idx, pts)
    # but since the extracted is flattend, I have to reshape it
    zz = zz.reshape([int(zz.shape[0]/2), 2])

    if filename:
        export_vf_landmarks(zz, filename=filename)

    return zz

def export_vf_landmarks(pts, filename='vf.csv'):
    with open(filename, mode='w') as vf_csv:
        writer = csv.writer(vf_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for pt in pts:
            writer.writerow(pt)
