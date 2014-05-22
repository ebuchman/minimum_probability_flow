import numpy as np
import time, pickle, sys
import theano
import theano.tensor as T
import os

from util import load_data, plot_samples
from rbm import energy, rbm_vhv, sample_rbm, sample_rbm_2wise, random_rbm, compute_dkl, compute_likelihood
from mpf import *

def test_mpf():
    nv, nh = 20, 16
    batch_size = 100
    n_epochs = 10
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


def train_mnist():
	nv, nh = 28*28, 500
	batch_size = 100
	n_epochs = 5
	learning_rate = 0.01
	decay = 0.99
	L2_reg = 0.001
	L1_reg = 0.0001
	momi, momf, momsw = 0.5, 0.9, 10
	k=2

	print 'generating data'
	f = open('/export/mlrg/ebuchman/datasets/mnist.pkl')
	d = pickle.load(f)
	f.close()
	tr, val, ts = d
	X = tr[0]
	print X.shape

	theta_init = random_theta(nv, nh, k=k)
	param_init = split_theta(theta_init, nv, nh, k=k)

	param_init[0] = param_init[0].reshape(nv*nh)

	optimizer = 'sgd'

	print 'initializing mpf'
	model = metaMPF(nv, nh, batch_size, n_epochs, learning_rate=learning_rate, learning_rate_decay=decay, initial_momentum=momi, final_momentum=momf, momentum_switchover=momsw, L1_reg=0.0, L2_reg=L2_reg, k=k)
	print "training", optimizer
	start = time.time()
	model.fit(train_X = X, optimizer = optimizer, param_init = param_init)
	end = time.time()
	print 'training time:', end-start

	params_fit = split_theta(model.mpf.theta.get_value(), nv, nh, k=k)

	if k == 1:
		w, bh, bv = params_fit
		samples = sample_rbm(w, bh, bv, 20, sample_every=1000, burnin=1000)
		save_name = 'mnist_samples_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate)
	elif k == 2:
		w, wh, bh, bv = params_fit
		samples = sample_rbm_2wise(w, wh, bh, bv, 20, sample_every=1000, burnin=1000)
		save_name = 'mnist_samples_2wise_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate)


	f = open(os.path.join('data', 'results', save_name), 'w')
	pickle.dump([samples, params_fit], f)
	f.close()

def sample_from_params():
	f = open('data/results/mnist_samples_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate))
	samples, params = pickle.load(f)
	f.close()
	
	samples = sample_rbm(params, 20, sample_every=1000, burnin=1000, k=1)




if __name__=='__main__':

	#train_mnist()
	test_mpf()
	


