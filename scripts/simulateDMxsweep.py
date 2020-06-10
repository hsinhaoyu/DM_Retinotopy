# simulate a scenario where rostral to the foveal representation of V2 is a DM-like area

import sys
from pathlib import Path
import numpy as np

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
outdir          = "../results/simulateDMxsweep"
clean_dir       = True
save_interval   = 100
n_iterations    = 1800
k0              = 30.0
kr              = 0.003
eta0            = 0.1
##### the visual field prototypes to for the elastioc net to cover
n_prototypes    = 400 
magnification   = -0.5
ecc0            = 0.05
ecc1            = 10

##### Define The geometry of the cortical area to simulate
boundary_len    = 7.0   # in milimeter
map_h           = 20    # this is on the boundary
map_w           = 10    

def main(rseed, outputdir):
    if clean_dir:
        clean_outdir(outputdir)
    
    x0 = generate_vf_landmarks(magnification, ecc0, ecc1, n_prototypes, vf='full',   filename=full_filename('vf', outputdir))
    b0 = generate_V2_boundary_retinotopy(boundary_len, map_h,           branch='vd', filename=full_filename('boundary', outputdir), tilt=10.0)

    y = initialize_map(map_h, map_w, x0, filename=full_filename('init', outdir), random_seed = rseed)

    # loop through b1 and b2
    b1s = [0.03*1.6**i for i in range(0, 8)]
    b2s = [0.03*1.6**i for i in range(0, 8)]
    print("b1:", b1s)
    print("b2:", b2s)
    for b1 in b1s:
        for b2 in b2s:
            print("b1 = %10.4f \t b2 = %10.4f"%(b1, b2))
            optimize(x0, b0, b1, b2, map_h, map_w, y,
                     k0 = k0,
                     kr = kr,
                     eta0 = eta0,
                     outdir = outputdir,
                     save_interval = 0,       # disable saving intermediate maps
                     n = n_iterations,
                     report_iterations = False,
                     log_iterations = False)

if __name__ == "__main__":
    if len(sys.argv)==2:
        rseed = int(sys.argv[1])
        path = (Path(outdir) / str(rseed).zfill(3)).resolve()
        path.mkdir(parents=False, exist_ok=True)
        main(rseed, str(path))
    else:
        rseed = 0
        main(rseed, outdir)
