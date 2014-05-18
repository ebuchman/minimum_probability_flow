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

	seed = np.random.seed(100)

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
	train(nv, nh, batch_size, 2, X, 'sgd', params, param_init, lr=0.01, lrd=1, mom_i=0.5, mom_f =0.9, mom_switch=5, L1_reg=0.00, L2_reg=0.00)
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
	batch_size = 200
	n_epochs = 20
	learning_rate = 0.01
	decay = 0.99
	L2_reg = 0.0001
	momi, momf, momsw = 0.5, 0.9, 5

	print 'generating data'
	f = open('/export/mlrg/ebuchman/datasets/mnist.pkl')
	d = pickle.load(f)
	f.close()
	tr, val, ts = d
	X = tr[0]
	print X.shape

	theta_init = random_theta(nv, nh)
	param_init = split_theta(theta_init, nv, nh)

	param_init[0] = param_init[0].reshape(nv*nh)

	optimizer = 'sgd'

	print 'initializing mpf'
	model = metaMPF(nv, nh, batch_size, n_epochs, learning_rate=learning_rate, learning_rate_decay=decay, initial_momentum=momi, final_momentum=momf, momentum_switchover=momsw, L1_reg=0.0, L2_reg=L2_reg)
	print "training", optimizer
	start = time.time()
	model.fit(train_X = X, optimizer = optimizer, param_init = param_init)
	end = time.time()
	print 'training time:', end-start
	W, bh, bv = params_fit = split_theta(model.mpf.theta.get_value(), nv, nh)

	samples = sample_rbm(W, bh, bv, 20, sample_every=1000, burnin=1000)
	f = open('data/results/mnist_samples_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate), 'w')
	pickle.dump([samples, params_fit], f)
	f.close()



if __name__=='__main__':

	train_mnist()
	#test_mpf()
