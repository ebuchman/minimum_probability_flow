import numpy as np
import time, pickle, sys
import theano
import theano.tensor as T
import os
import signal, sys

from params import load_params, deep_samples
from view import save_view
from util import load_data, plot_samples
from rbm import energy, rbm_vhv, sample_h_given_v_np, sample_v_given_h_np, sample_h_given_v_2wise_np, sample_rbm, sample_rbm_2wise, random_rbm, compute_dkl, compute_likelihood
from mpf import *


def signal_handler(metaMPF, signal=None, frame=None):
	print 'shutting down'
	save_path = 'ctrl_c_params_%s_nh%d_ne%d'%(metaMPF.opt, metaMPF.nh, metaMPF.current_epoch)
	print save_path
	params = split_theta(metaMPF.mpf.theta.get_value(), metaMPF.nv, metaMPF.nh)
	save_params(save_path, params)
	sys.exit(0)


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
	nv, nh = 14*14, 200
	batch_size = 100
	n_epochs = 1
	learning_rate = 0.01
	decay = 0.99
	L2_reg = 0.001
	L1_reg = 0.001
	momi, momf, momsw = 0.5, 0.9, 10
	k=1
	sample_every = None

	LOAD_PARAMS = False

	print 'generating data'
	f = open('/export/mlrg/ebuchman/datasets/mnistX_binary_lowres.pkl')
	d = pickle.load(f)
	f.close()
	if len(d) == 3:
		tr, val, ts = d
		X = tr[0]
	else:
		X = d
	print X.shape

	if LOAD_PARAMS:
		param_init = load_params(nv, nh, k, 10, learning_rate)
	else:
		theta_init = random_theta(nv, nh, k=k)
		param_init = split_theta(theta_init, nv, nh, k=k)
		param_init[0] = param_init[0].reshape(nv*nh)

	optimizer = 'sof'

	param_first_layer = _train(X, optimizer, param_init, nv, nh, batch_size, n_epochs, learning_rate, decay, momi, momf, momsw, L1_reg, L2_reg, k)
	sample_and_save(param_first_layer, nh, n_epochs, learning_rate, k, optimizer)
#	w, bh, bv = param_first_layer
#	samples = sample_rbm(w, bh, bv, 20)
#	save_view(samples, nh, n_epochs, learning_rate)	
	quit()	



	if k == 1:
		W, bh, bv = param_first_layer
		X2 = sample_h_given_v_np(X, W, bh, nh)
	elif k == 2:
		W, Wh, bh, bv = param_first_layer
		X2 = sample_h_given_v_2wise_np(X, W, Wh, bh, nh)

	print X2.shape

	theta_init = random_theta(nh, nh, k=k)
	param_init = split_theta(theta_init, nh, nh, k=k)
	param_init[0] = param_init[0].reshape(nh*nh)

	param_second_layer = _train(X2, optimizer, param_init, nh, nh, batch_size, n_epochs, learning_rate, decay, momi, momf, momsw, L1_reg, L2_reg, k)


	if k == 1:
		W, bh, bv = param_second_layer
		X3 = sample_h_given_v_np(X2, W, bh, nh)
	elif k == 2:
		W, Wh, bh, bv = param_second_layer
		X3 = sample_h_given_v_2wise_np(X2, W, Wh, bh, nh)

	print X3.shape

	theta_init = random_theta(nh, nh, k=k)
	param_init = split_theta(theta_init, nh, nh, k=k)
	param_init[0] = param_init[0].reshape(nh*nh)

	param_third_layer = _train(X3, optimizer, param_init, nh, nh, batch_size, n_epochs, learning_rate, decay, momi, momf, momsw, L1_reg, L2_reg, k)


	#params_fit = split_theta(model.mpf.theta.get_value(), nv, nh, k=k)
	deep_samples([param_first_layer, param_second_layer, param_third_layer], 50)





if __name__=='__main__':

	train_mnist()
	#test_mpf()
	


