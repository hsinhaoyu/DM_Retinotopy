# simulate a scenario where rostral to V2 is a long strip of area (like V3)

# this is a hack to import elastitopy without making it into a module
import sys
import os
sys.path.append(os.path.abspath('../elastictopy'))

from helpers import full_filename
from helpers import clean_outdir

from vf import generate_vf_landmarks
from vf import export_vf_landmarks
from retinotopy import generate_V2_boundary_retinotopy

from elastic_net import initialize_map
from elastic_net import optimize

##### general parameters
outdir          = "../results/simulateV3"
clean_dir       = True
save_interval   = 100
n_iterations    = 2000
random_seed     = 0
k0              = 30.0
kr              = 0.003
eta0            = 0.1
##### the visual field prototypes to for the elastioc net to cover
n_prototypes    = 400
magnification   = -2.0
ecc0            = 0.2
ecc1            = 12

##### Define The geometry of the cortical area to simulate
boundary_len    = 22.0   # in milimeter
map_h           = 40     
map_w           = 5      

##### model weights
b1 = 0.123                # smoothness
b2 = 0.123                # congruence

def main():
    if clean_dir:
        clean_outdir(outdir)
    
    x0 = generate_vf_landmarks(magnification, ecc0, ecc1, n_prototypes, vf='full',   filename=full_filename('vf', outdir))
    b0 = generate_V2_boundary_retinotopy(boundary_len, map_h,           branch='vd', filename=full_filename('boundary', outdir), tilt=10.0)

    y = initialize_map(map_h, map_w, x0, filename=full_filename('init', outdir), random_seed = random_seed)

    optimize(x0, b0, b1, b2, map_h, map_w, y,
             k0 = k0,
             kr = kr,
             eta0 = eta0, 
             outdir = outdir,
             save_interval = save_interval,
             n = n_iterations)

if __name__ == "__main__":
    main()
