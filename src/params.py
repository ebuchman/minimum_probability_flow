import numpy as np
import time, pickle, sys, math
import theano
import theano.tensor as T
import os

from util import load_data, plot_samples
from rbm import energy, energy2wise, debug_energy2wise, sample_rbm, sample_rbm_2wise, sample_v_given_h_np, sample_h_given_v_np, rbm_vhv, sample_rbm, random_rbm, compute_dkl, compute_likelihood
from debug import print_debug

# save params at shutdown
def signal_handler(metaMPF, signal=None, frame=None):
	print 'shutting down'
	save_path = 'ctrl_c_params_%s_nh%d_ne%d'%(metaMPF.opt, metaMPF.nh, metaMPF.current_epoch)
	print save_path
	params = split_theta(metaMPF.mpf.theta.get_value(), metaMPF.nv, metaMPF.nh)
	save_params(save_path, params)
	sys.exit(0)

def load_params(nv, nh, k, n_epochs, learning_rate):
	if k == 1:
		path = 'mnist_samples_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate)
	elif k == 2:
		path = 'mnist_samples_2wise_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate)

	f = open(os.path.join('data', 'results', path))
	samples, params = pickle.load(f)
	f.close()

	params[0] = params[0].reshape(nv*nh)

	return params

def deep_samples(params, nsamps, opt, sample_every=1000, burnin=1000):
	layers = len(params)

	path = 'mnist_deep_samples_%s_'%opt


	w, bh, bv = params[-1]
	path += 'nh%d_'%len(bh)
	next_samples = sample_rbm(w, bh, bv, nsamps, sample_every=sample_every, burnin=burnin)
	next_samples = sample_h_given_v_np(next_samples, w, bh, len(bh))
	for i in xrange(layers-1):
		w, bh, bv = params[-2-i]
		path += 'nh%d_'%len(bh)
		mean = False if i == layers-2 else True	
		#next_samples = sample_v_given_h_np(next_samples, w, bv, len(bv), mean=mean)

	path += '.pkl'

	print path
	f = open(os.path.join('data', 'results', path), 'w')
	pickle.dump([next_samples, params], f)
	f.close()

def sample_Gamma(w, bh, bv, sample_every=1000, burnin=1000):
	pass


def sample_from_params():
	f = open('data/results/mnist_samples_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate))
	samples, params = pickle.load(f)
	f.close()
	
	samples = sample_rbm(params, 20, sample_every=1000, burnin=1000, k=1)

def random_theta(nv, nh, k=1):
	w_bound = 4*np.sqrt(6. / (nv + nh))
	w0 = np.asarray(np.random.uniform(size=(nv*nh), low=-w_bound, high=w_bound), dtype=theano.config.floatX)
	bh0 = np.asarray(np.zeros(nh), dtype=theano.config.floatX)
	bv0 = np.asarray(np.zeros(nv), dtype=theano.config.floatX)

	if k==2:
		assert(nh%2 == 0)
		wh = np.asarray(np.random.uniform(size=(nh/2), low=-0.01*w_bound, high=0.01*w_bound), dtype=theano.config.floatX)
		return np.asarray(np.concatenate((w0, wh, bh0, bv0)), dtype=theano.config.floatX)

	return np.asarray(np.concatenate((w0, bh0, bv0)), dtype=theano.config.floatX)

def split_theta(theta, nv, nh, k=1):
	params = []

	param_idx = 0
	w = theta[param_idx:param_idx+nv*nh].reshape(nv, nh)
	params.append(w)
	param_idx += nv*nh

	if k == 2:
		assert (nh % 2 == 0)
		wh = theta[param_idx:param_idx+nh/2]
		params.append(wh)
		param_idx += nh/2

	bh = theta[param_idx:param_idx+nh]
	params.append(bh)
	param_idx += nh
	
	bv = theta[param_idx:param_idx+nv]
	params.append(bv)
	param_idx += nv

#	assert(param_idx == theta_shape)
	return params

def save_params(save_path, params):
	f = open(os.path.join('data', 'results', save_path), 'w')
	pickle.dump(params, f)
	f.close()

def sample_and_save(params_fit, nh, n_epochs, learning_rate, k, opt, nsamps=20, sample_every=1000, burnin=1000):
	if k == 1:
		w, bh, bv = params_fit
		samples = sample_rbm(w, bh, bv, nsamps, sample_every=sample_every, burnin=burnin)
		save_name = 'mnist_samples_%s_nh%d_ne%d_lr%f.pkl'%(opt, nh, n_epochs, learning_rate)
	elif k == 2:
		w, wh, bh, bv = params_fit
		samples = sample_rbm_2wise(w, wh, bh, bv, nsamps, sample_every=sample_every, burnin=burnin)
		save_name = 'mnist_samples_2wise_%s_nh%d_ne%d_lr%f.pkl'%(opt, nh, n_epochs, learning_rate)

	print save_name
	f = open(os.path.join('data', 'results', save_name), 'w')
	pickle.dump([samples, params_fit], f)
	f.close()

def nCr(n, r):
	f = math.factorial
	return f(n) / f(r) / f(n-r)

