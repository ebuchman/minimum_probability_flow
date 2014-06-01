import numpy as np
import time, pickle, sys
import theano
import theano.tensor as T
import os
import signal, sys
import random

from params import load_params, deep_samples, signal_handler
from util import load_data, plot_samples
from rbm import  sample_h_given_v_np, sample_h_given_v_2wise_np, random_rbm
from mpf import *
from cd_rbm import test_rbm


def _train(X, optimizer, param_init, nv, nh, batch_size, n_epochs, learning_rate, decay, momi, momf, momsw, L1_reg, L2_reg, k, sample_every=None):
	print 'initializing mpf'
	model = metaMPF(nv, nh, batch_size, n_epochs, learning_rate=learning_rate, learning_rate_decay=decay, initial_momentum=momi, final_momentum=momf, momentum_switchover=momsw, L1_reg=0.0, L2_reg=L2_reg, k=k)

	# register signal handler for ctrl_c
	f = lambda x, y : signal_handler(model, x, y) 
	signal.signal(signal.SIGINT, f)

	print "training", optimizer
	start = time.time()
	model.fit(train_X = X, optimizer = optimizer, param_init = param_init, sample_every=sample_every)
	end = time.time()
	print 'training time:', end-start
	return split_theta(model.mpf.theta.get_value(), nv, nh, k=k)

def train_mnist():
	nv, nhs = 28*28, [500] #, 200, 200, 200]
	batch_size = 20
	train_size = 10000
	n_epochs = 15
	learning_rate = 0.001
	decay = 1.
	L2_reg = 0.01
	L1_reg = 0.000
	momi, momf, momsw = 0.5, 0.9, 10
	k=1
	sample_every = None

	LOAD_PARAMS = False

	data_path = '/export/mlrg/ebuchman/datasets/mnistX_binary_lowres.pkl'
	data_path = '/export/mlrg/ebuchman/datasets/mnist_binary.pkl'
	data_path = '/mnt/data/datasets/mnist_binary.pkl'
	print 'opening data'
	f = open(data_path)
	d = pickle.load(f)
	f.close()
	if len(d) == 3:
		tr, val, ts = d
		X = tr[0]
	else:
		X = d

	print X.shape

	X0 = X
	optimizer = 'cd'

	n_layers = len(nhs)
	layer_params = []


	X = X[:100]
	#	X = X[random.sample(xrange(len(X0)), train_size)]


	w, bh, bv, t = test_rbm(X)

	print 'took', t
	f = open('data/results/mnist_cd_deeplearning_ref.pkl')
	pickle.dump([None, [w, bh, bv]], d)
	f.close()
	quit()



	# train each layer of DBN
	for layer in xrange(n_layers):
		
		if layer == 0:
			nv, nh = nv, nhs[0]
		else:
			nv, nh = nhs[layer-1], nhs[layer]	

		if LOAD_PARAMS: # this needs to be fixed...
			param_init = load_params(nv, nh, k, 10, learning_rate)
		else:
			theta_init = random_theta(nv, nh, k=k)
			param_init = split_theta(theta_init, nv, nh, k=k)
			param_init[0] = param_init[0].reshape(nv*nh)

		# fit rbm
		params_fit = _train(X, optimizer, param_init, nv, nh, batch_size, n_epochs, learning_rate, decay, momi, momf, momsw, L1_reg, L2_reg, k)
		layer_params.append(params_fit)
	#	sample_and_save(param_first_layer, nh, n_epochs, learning_rate, k, optimizer)

		# get data for next layer by propogating mnist up to current layer
		if k == 1:
			X = X0[random.sample(xrange(len(X0)), train_size)]
			for i in xrange(layer+1):
				W, bh, bv = layer_params[i]
				mean = False if i == layer else True
				X = sample_h_given_v_np(X, W, bh, nh, mean=mean)
		elif k == 2:
			X = X0[random.sample(xrange(len(X0)), train_size)]
			for i in xrange(layer+1):
				W, Wh, bh, bv = layer_params[i]
				mean = False if i == layer else True
				X = sample_h_given_v_2wise_np(X, W, Wh, bh, nh, mean = mean)

	#save_name = "mnist_


	#params_fit = split_theta(model.mpf.theta.get_value(), nv, nh, k=k)
	deep_samples(layer_params, nsamps=50, opt=optimizer)






if __name__=='__main__':

	train_mnist()
	


