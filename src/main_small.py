
import numpy as np
import time, pickle, sys
import theano
import theano.tensor as T
import os

from util import load_data, plot_samples
from rbm import energy, rbm_vhv, sample_h_given_v_np, sample_v_given_h_np, sample_h_given_v_2wise_np, sample_rbm, sample_rbm_2wise, random_rbm, compute_dkl, compute_likelihood
from mpf import *


def test_mpf():
    nv, nh = 20, 16
    batch_size = 100
    n_epochs = 1
    n_train = 1000
    k = 2

    seed = np.random.seed(100)

    print 'generating data'
    samples, params = random_rbm(nv, nh, n_train, sample_every=100, burnin=1000, k=k)	
    X = samples

    seed = np.random.seed(200)

    theta_init = random_theta(nv, nh, k=1)
    param_init = split_theta(theta_init, nv, nh, k=1)

# compute dkl and likelihood before training
    L_rand = compute_likelihood(param_init, X) # infer k from length of param_init
    dkl_rand = compute_dkl(params, param_init)

    print 'init dkl', dkl_rand
    print 'init nll', -L_rand
    print ''

    ne = 1

    w, wh, bh, bv = param_init
    print "k2 bfgs ################################"
    train(nv, nh, batch_size, ne, X, 'bfgs', params, param_init, k=2)
    print "k1 bfgs ################################"
    train(nv, nh, batch_size, ne, X, 'bfgs', params, [w, bh, bv], k=1)
    print "k2 sgd ################################"
    train(nv, nh, batch_size, ne, X, 'sgd', params, param_init, lr=0.01, lrd=1, mom_i=0.5, mom_f =0.9, mom_switch=5, L1_reg=0.00, L2_reg=0.00, k=2)
    print "k1 sgd ################################"
    train(nv, nh, batch_size, ne, X, 'sgd', params, [w, bh, bv], lr=0.01, lrd=1, mom_i=0.5, mom_f =0.9, mom_switch=5, L1_reg=0.00, L2_reg=0.00, k=1)
    quit()
    train(nv, nh, batch_size, 10, X, 'cd', params, param_init, cdk=1)
    train(nv, nh, batch_size, 100, X, 'cd', params, param_init, cdk=1)
    train(nv, nh, batch_size, 10, X, 'cd', params, param_init, cdk=10)
    train(nv, nh, batch_size, 100, X, 'cd', params, param_init, cdk=10)
    train(nv, nh, batch_size, 5, X, 'bfgs', params, param_init)
    train(nv, nh, batch_size, 10, X, 'bfgs', params, param_init)
    train(nv, nh, batch_size, 20, X, 'bfgs', params, param_init)
    train(nv, nh, batch_size, 5, X, 'sof', params, param_init)
    train(nv, nh, batch_size, 10, X, 'sof', params, param_init)
    train(nv, nh, batch_size, 20, X, 'sof', params, param_init)



if __name__=='__main__':

	train_mnist()
	#test_mpf()
	

