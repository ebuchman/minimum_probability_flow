import numpy as np
import time, pickle, sys
import theano
import theano.tensor as T

from util import load_data, plot_samples
from rbm import energy, rbm_vhv, sample_rbm, random_rbm, compute_dkl, compute_likelihood
from mpf import *

def test_mpf():

    nv, nh = 10,5
    batch_size = 100
    n_epochs = 10
    n_train = 1000

    print 'generating data'
    samples, params = random_rbm(nv, nh, 1000, sample_every=100, burnin=1000)	
    X = samples

    theta_init = random_theta(nv, nh)
    param_init = split_theta(theta_init, nv, nh)

# compute dkl and likelihood before training
    L_rand = compute_likelihood(param_init, X)
    dkl_rand = compute_dkl(params, param_init)

    print 'init dkl', dkl_rand
    print 'init nll', -L_rand
    print ''

    train(nv, nh, batch_size, 5, X, 'bfgs', params, param_init)
    train(nv, nh, batch_size, 10, X, 'sgd', params, param_init, lr=0.01, lrd=1, mom_i=0.5, mom_f =0.9, mom_switch=5, L1_reg=0.00, L2_reg=0.00)
    quit()
    train(nv, nh, batch_size, 10, X, 'cd', params, param_init, k=1)
    train(nv, nh, batch_size, 100, X, 'cd', params, param_init, k=1)
    train(nv, nh, batch_size, 10, X, 'cd', params, param_init, k=10)
    train(nv, nh, batch_size, 100, X, 'cd', params, param_init, k=10)
    train(nv, nh, batch_size, 5, X, 'bfgs', params, param_init)
    train(nv, nh, batch_size, 10, X, 'bfgs', params, param_init)
    train(nv, nh, batch_size, 20, X, 'bfgs', params, param_init)
    train(nv, nh, batch_size, 5, X, 'sof', params, param_init)
    train(nv, nh, batch_size, 10, X, 'sof', params, param_init)
    train(nv, nh, batch_size, 20, X, 'sof', params, param_init)


def train_mnist():
	nv, nh = 28*28, 500
	batch_size = 100
	n_epochs = 10

	print 'generating data'
	f = open('/export/mlrg/ebuchman/datasets/mnistSMALL.pkl')
	d = pickle.load(f)
	f.close()
	tr, val, ts = d
	X = tr[0]

	theta_init = random_theta(nv, nh)
	param_init = split_theta(theta_init, nv, nh)

	param_init[0] = param_init[0].reshape(nv*nh)

	optimizer = 'bfgs'

	print 'initializing mpf'
	model = metaMPF(nv, nh, batch_size, n_epochs)
	print "training", optimizer
	start = time.time()
	model.fit(train_X = X, optimizer = optimizer, param_init = param_init)
	end = time.time()
	params_fit = split_theta(model.mpf.theta.get_value(), nv, nh)

	samples = sample_rbm(W, bh, bv, 20, sample_every=1000, burnin=1000)
	f = open('tosave.pkl')
	pickle.dump([samples, params_fit], f)
	f.close()



if __name__=='__main__':

	#	train_mnist()

	test_mpf()
	quit()

	train_ = 1

	data = load_data(ndata = 1000).astype(theano.config.floatX)

	if train_:
		W, bh, bv = train(tr_x=data, n_hidden= 200, batch_size=100, n_epochs = 5)

		s = time.time()
		f = open('results/%d.pkl'%s, 'w')
		pickle.dump(W, f)
		f.close()
	else:
		f = open('results/1395801107.pkl')
		W = pickle.load(f)
		f.close()

	samples = sample_rbm(W, bh, bv, 20, sample_every=1000, burnin=1000)

	plot_samples(samples)

