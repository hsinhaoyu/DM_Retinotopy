# Simulate the formation of retinotopic areas with elastic nets

### About
This code simulates the formation of retinotopic maps in the visual cortex, using a model that maps receptive fields of neurons in a visual area to the visual field. The model is constrained by the smoothness of the retinotopic map within the area, and by the congruence with a neighboring area.

This code was written to support a paper under review.

### How to run
- Create a directory to save the results to under `results/`
- Evoke one of the scripts from the `scripts/` directory. For example `python simulateV2.py`

### Notes
- This code was tested with Python 3.7.7 and Tensorflow 2.0.0.
- The Tensorflow code was written using Tensorflow 1. To get it run with Tensorflow 2, I use the compatibility mode

### About initialization
- `elastic_net.optimize()` can only be run once. To repeat the same simulation with multiple random initialization, run the program multiple times with different random seeds, and save to different directories.
- Scripts such as `scripts/simulateV2_sweep.py` run the simulation with different settings of the b1 (smoothness) and b2 (congruence) parameters. All these simulations will have the same initial map.
