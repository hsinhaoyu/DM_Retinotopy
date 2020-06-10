# This code was written for Tensorflow 1. For Tensorflow 2, use the compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from pathlib import Path

from helpers import full_filename

# map_h is the side that has a boundary
# this geometry is defined by neighborhood()
# for example, (h, w) = (5, 3)
# index:
#    x 0 5 10
#    x 1 6 11
#    x 2 7 12
#    x 3 8 13
#    x 4 9 14
# x is boundary

def neighborhood(h, w):
    """
        Numbers {0...h*w-1} arranged in a h x w grid, produce a list of neighboring numbers.
        The ith element of the list is a list of numbers that neighbor i
    """
    
    # this line defines how the geometry is coded
    # for (h, w) = (4, 2)
    # 0 4
    # 1 5
    # 2 6
    # 3 7
    x = np.transpose(np.array(range(h*w)).reshape([w, h]))

    lst = []
    for i in range(w):
        for j in range(h):
            neighbors = []
            for ii in [-1, 0, 1]:
                for jj in [-1, 0, 1]:
                    if (i+ii>=0) and (i+ii<w) and (j+jj>=0) and (j+jj<h):
                        neighbors.append(x[j+jj,i+ii])
            lst.append(neighbors)
    return lst

def make_mask(h, w, neighbors):
    x = np.zeros([h*w, h*w], dtype=np.float64)
    for i in range(h*w):
        for j in neighbors[i]:
            x[i, j] = 1.0
    return x

def make_boundary_mask(n):
    """
        n: numer of elements on the boundary
        return a n*n array
    """
    x = np.zeros([n, n], dtype=np.float64)
    for i in range(n):
        for j in [-1, 0, 1]:
            if (i+j)>=0 and (i+j)<n:
                x[i, i+j] = 1.0
    return x

def initialize_map(h, w, x, filename="", random_seed=None):
    """
        Initialize cortical map, save to a file, and create a tensorflow variable "y"

        initialze a cortical map of the size map_w * map_h
        each point on the map is randomly assigned to prototypes on the visual field (x) with some added noise
    """

    if random_seed:
        np.random.seed(random_seed)
    
    map_size = h * w

    idx = np.random.randint(x.shape[0], size = map_size)
    y = x[idx, :] # randomly sample from x with replacement
    n = np.random.normal(0.0, 1.0, y.shape) # the standard deviation is set to 5.0 deg
    res = y + n

    if filename:
        np.savetxt(filename, res, fmt='%f', delimiter=",")

    y = tf.get_variable("y", dtype = tf.float64, initializer = res)
    
    return y

def main_cost(x, y, kappa):
    yx_diff = tf.expand_dims(y, 1) - tf.expand_dims(x, 0)
    yx_normsq = tf.einsum('ijk,ijk->ij', yx_diff, yx_diff)

    # note that all entries are between 0.0 and 1.0
    yx_gauss = tf.exp(-1.0 * yx_normsq / (2.0 * kappa * kappa))

    yx_gauss_sum = tf.reduce_sum(yx_gauss, axis=0)
    yx_gauss_sum = yx_gauss_sum

    # to protect the log from becoming -inf 
    yx_gauss_sum = tf.clip_by_value(yx_gauss_sum, clip_value_min=1.0e-306, clip_value_max=1.0e+306)

    yx_cost = -1.0 * kappa * tf.reduce_sum(tf.log(yx_gauss_sum))
    return yx_cost, yx_gauss_sum

def generate_elastic_net(vf_prototypes, boundary_retinotopy, y, map_h, map_w):
    # beta1: within-area smoothness
    # beta2: between-area congruence
    # kappa: annealing parameter
    beta1 = tf.placeholder(tf.float64, shape=(), name="b1")
    beta2 = tf.placeholder(tf.float64, shape=(), name="b2")
    kappa = tf.placeholder(tf.float64, shape=(), name="k")

    x = tf.constant(vf_prototypes, dtype=tf.float64, name='x')
    b = tf.constant(boundary_retinotopy, dtype=tf.float64, name='b')

    # this is the boundary of the cortical map
    yb = y[:map_h]

    yx_cost, yx_gauss = main_cost(x, y, kappa)
    
    ##### regularization term 1 - within-area smoothness
    # n is a list of map_h * map_w objects. The i-th item of n is a list containing the indices of nodes neighboring the i-th node
    n = neighborhood(map_h, map_w)
    mask = tf.constant(make_mask(map_h, map_w, n), dtype=tf.float64, name='mask')

    # pairwise distance: first use broadcast to calculate pairwise difference
    yy_diff = tf.expand_dims(y, 1) - tf.expand_dims(y, 0)
    yy_normsq = tf.einsum('ijk,ijk->ij', yy_diff, yy_diff)
    yy_normsq_masked = tf.multiply(mask, yy_normsq)

    reg1 = tf.reduce_sum(yy_normsq_masked)

    ##### regularization term 2 - between-area congruency
    # pairwise distance: first use broadcast to calculate pairwise difference
    yb_diff = tf.expand_dims(yb, 1) - tf.expand_dims(b, 0)
    yb_normsq = tf.einsum('ijk,ijk->ij', yb_diff, yb_diff)
    bmask = make_boundary_mask(map_h)
    yb_normsq_masked = tf.multiply(bmask, yb_normsq)

    reg2 = tf.reduce_sum(yb_normsq_masked)

    ##### total cost
    cost = yx_cost + beta1 * reg1 + beta2 * reg2

    total_cost = cost
    partial_costs = [yx_cost, reg1, reg2]
    tf_placeholders = [beta1, beta2, kappa]
    return total_cost, partial_costs, tf_placeholders, yx_gauss

def generate_optimizer(cost, eta0 = 0.05, m = 0.8, eta_decay=0.99):
    """
    cost - the cost function
    eta0 - initial learning rate
    m - momentum
    """
    
    global_step = tf.Variable(0, trainable=False)
    current_eta = tf.train.exponential_decay(eta0, global_step, 10, eta_decay, staircase=False)

    opt = tf.train.RMSPropOptimizer(current_eta, momentum = m).minimize(cost, global_step = global_step)
    return opt

def save_map(sess, y, b1, b2, i, outdir="", save_interval=0, quiet=False, suffix=''):
    if (outdir!="") and save_interval>0:
        if i%save_interval==0:
            filename = "y-{b1:4.3f}-{b2:4.3f}-{i}".format(b1=b1, b2=b2, i=str(i).zfill(5))
            zz = sess.run(y)
            
            if (not quiet):
                print("saving {filename}".format(filename=filename))
                
            np.savetxt(full_filename(filename, outdir, ext=".data", suffix=suffix),
                       np.array(zz), fmt = "%f", delimiter=",")

def report_and_log(partial_costs, tf_placeholders, i, k, b1, b2, sess, reportP=True, logP=True, logfile='log'):
    yx_cost = partial_costs[0]
    reg1    = partial_costs[1]
    reg2    = partial_costs[2]
    kappa   = tf_placeholders[2]

    [e0, e1, e2] = sess.run([yx_cost, reg1, reg2], {kappa: k})
    tt = e0 + b1*e1 + b2*e2
    if reportP:
        print("%d\t%5f\t%6.4f"%(i, k, tt)) # total cost
        print("\t\t%10.4f\t%10.4f\t%10.4f"%(e0, e1, e2))

    if logP and logfile:
        logstring = "%10.4f,%10.4f,%d,%10.4f,%10.4f,%10.4f,%10.4f,%10.4f\n"%(b1, b2, i, k, tt, e0, e1, e2)
        with open(logfile, 'a') as logfp:
            logfp.write(logstring)

def train(opt, total_cost, partial_costs, tf_placeholders, y, b1, b2, outdir="", save_interval=0, n=1000, k0=30.0, kr=0.005, quiet_save=False, report_iterations = True, log_iterations = True, logfile='log', suffix='', yx_gauss=None):
    """
    b1:              weights for the smoothness term
    b2:              weights for the congruency term
    n:               number of iterations
    k0:              initial annealing parameter
    kr:              reduction of the annealing parameter at each iteration
    outdir:          where to save the results.
    save_intervals:  save map every several iterations. if set to 0, saving is disabled
    """

    yx_cost = partial_costs[0]
    reg1    = partial_costs[1]
    reg2    = partial_costs[2]
    beta1   = tf_placeholders[0]
    beta2   = tf_placeholders[1]
    kappa   = tf_placeholders[2]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # save initial map
    #      always save the initial map (by setting save_interval to 1)
    save_map(sess, y, b1, b2, 0, outdir, save_interval=1, quiet=quiet_save, suffix=suffix)

    k = k0
    for i in range(1, n+1):
        # itertaively optimize the total cost function
        sess.run(opt, {kappa: k, beta1: b1, beta2: b2})

        report_and_log(partial_costs, tf_placeholders, i, k, b1, b2, sess,
                       reportP = report_iterations,
                       logP = log_iterations,
                       logfile=full_filename(logfile, outdir, suffix=suffix))

        if i>0:
            save_map(sess, y, b1, b2, i, outdir, save_interval, quiet=quiet_save, suffix=suffix)

        k = k - k*kr
    
    # save the final result
    #      set save_interval to 1, so that the final result is saved even if intermediate saving is disabled by save_interval=0
    save_map(sess, y, b1, b2, i, outdir, save_interval=1, quiet=quiet_save, suffix=suffix)
    
    report_and_log(partial_costs, tf_placeholders, i, k, b1, b2, sess,
                   reportP = False,
                   logP = True,
                   logfile = full_filename(logfile, outdir, suffix=suffix))

def optimize(x0, b0, b1, b2, map_h, map_w, y, outdir="", save_interval=0, n=1000, k0=30.0, kr=0.005, eta0 = 0.05, m = 0.8, quiet_save = False, report_iterations = True, log_iterations = True, logfile="log", suffix=""):
    """
    Easy interface to train()
    x0             - visual field prototypes
    b0             - boundary retinotopy
    b1             - model weights
    b2             - model weights
    map_h          - height of the cortical map
    map_w          - width of the cortical map
    y              - the cortical map to learn (tf variable)
    outdir:        directory to save the file to
    save_interval: save map for how many iterations (if 0, do not save intermediate maps)
    n:             how many iterations to update
    quiet_save:    do not print "saving to...."
    report_iterations: report every iteration?
    """
    
    (total_cost, partial_costs, tf_placeholders, yx_gauss) = generate_elastic_net(x0, b0, y, map_h, map_w)
    opt = generate_optimizer(total_cost, eta0 = eta0, m = m)
    
    train(opt,
          total_cost, partial_costs, tf_placeholders, y, b1, b2,
          outdir = outdir,
          save_interval = save_interval,
          n = n,
          k0 = k0,
          kr = kr,
          quiet_save = quiet_save,
          report_iterations = report_iterations,
          log_iterations = log_iterations,
          logfile = logfile,
          suffix = suffix,
          yx_gauss = yx_gauss
    )
