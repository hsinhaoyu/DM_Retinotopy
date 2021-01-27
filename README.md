# Simulate the formation of retinotopic areas with elastic nets

### About
This code simulates the formation of retinotopic maps in the visual cortex, using a model that maps receptive fields of neurons in a visual area to the visual field. The model is constrained by the smoothness of the retinotopic map within the area, and by the congruence with a neighboring area.

This code was supported for a computational neurosciecne paper: Yu et al. (2020) A twisted visual field map in the primate dorsomedial cortex predicted by topographic continuity. _Science Advances_ vol. 6, no. 44, eaaz8673. [[link]](https://advances.sciencemag.org/content/6/44/eaaz8673).

### How to run
- Create a directory to save the results to under `results/`
- Evoke one of the scripts from the `scripts/` directory. For example `python simulateV2.py`

### Notes
- This code was tested with Python 3.7.7 and Tensorflow 2.0.0.
- The Tensorflow code was written using Tensorflow 1. To get it run with Tensorflow 2, I use the compatibility mode

### About initialization
- `elastic_net.optimize()` can only be run once. To repeat the same simulation with multiple random initialization, run the program multiple times with different random seeds, and save to different directories.
- Scripts such as `scripts/simulateV2sweep.py` run the simulation with different settings of the b1 (smoothness) and b2 (congruence) parameters. All these simulations will have the same initial map.
