import numpy as np
import os, pickle
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as rng_mrg
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.linalg.kron import kron

def energy_np(X, W, bh, bv):
	wx_b = np.dot(X, W) + bh
	vbias_term = np.dot(X, bv)
	hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=-1)
	return -hidden_term - vbias_term

def energy(X, W, bh, bv):
	wx_b = T.dot(X, W) + bh
	vbias_term = T.dot(X, bv)
	hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=-1)
	return -hidden_term - vbias_term

def energy2wise(X, W, Wh, bh, bv, adder, nh):
	wx_b = T.dot(X, W) + bh
	e_wx_b = T.exp(wx_b)

	pairsum = T.dot(e_wx_b, adder.T)
	first = e_wx_b.T[T.arange(0, nh, 2)].T
	pairprod = pairsum*first - first**2

	hidden_term = T.sum(T.log(1 + pairsum + pairprod*Wh), axis=-1)

	vbias_term = T.dot(X, bv)
	return -hidden_term - vbias_term


theano_rng = rng_mrg.MRG_RandomStreams(seed=100)
#theano_rng = RandomStreams(100)


def rbm_vhv(v, W, bv, bh, nv, nh):
	prop_up = T.nnet.sigmoid(T.dot(v, W) + bh)
	h = theano_rng.binomial(n=1, p = prop_up, dtype=theano.config.floatX, size=(nh,), ndim=1)
	prop_down = T.nnet.sigmoid(T.dot(W, h) + bv)
	v = theano_rng.binomial(n=1, p = prop_down, dtype=theano.config.floatX, size=(nv,), ndim=1)

	return v, prop_down
	
def rbm_vhv_2wise(v, W, Wh, bv, bh, nv, nh, adder):
	phi = T.dot(v, W) + bh
	ephi = T.exp(phi)

	# wobble =  1 + exp(phi_2i) + exp(phi_{2i+1}) + exp(phi_2i + phi_{21+1} + Wh_i)
	# p(h_2i = 1 | v) = (exp(phi_2i) + exp(phi_2i + phi_{21+1} + Wh_i ) / wobble
	# p(h_{2i+1} = 1 | v) = (exp(phi_2i) + exp(phi_2i + phi_{2i+1} + Wh_i )) / wobble
	# the second term is the same in both - the pair term.  but it must be broadcasted (the kron!)
	# dotting by adder returns a vector of half the size of sums of pairs of elements

	pairsum = T.dot(ephi, adder.T)
	first = ephi.T[T.arange(0, nh, 2)].T
	pairprod = pairsum*first - first**2

	wobble = 1 + pairsum + pairprod*Wh
	
	pairterm = T.exp(pairsum + Wh)

	pairterm_broadcast = kron(pairterm.dimshuffle(0, 'x'), T.ones(2))
	wobble_broadcast = kron(wobble.dimshuffle(0, 'x'), T.ones(2))

	prop_up = T.exp(phi + pairterm_broadcast) / wobble_broadcast

	h = theano_rng.binomial(n=1, p = prop_up, dtype=theano.config.floatX, size=(nh,), ndim=1)
	prop_down = T.nnet.sigmoid(T.dot(W, h) + bv)
	v = theano_rng.binomial(n=1, p = prop_down, dtype=theano.config.floatX, size=(nv,), ndim=1)

	return v, prop_down

def sample_rbm(weights, bias_h, bias_v, n_samples, burnin=1000, sample_every=100, k=1):
	print "sampling from rbm"

	W = T.matrix('W')
	bv = T.vector('bv')
	bh = T.vector('bh')

	nv, nh = weights.shape

	init = theano_rng.binomial(p=0.5, size=(nv,), ndim=1, dtype=theano.config.floatX)

	[samples, prop_down], updates = theano.scan(rbm_vhv, outputs_info=[init, None],  non_sequences=[W, bv, bh, nv, nh], n_steps = burnin)
	init2 = samples[-1]	

	burn_in_f = theano.function([W, bv, bh], init2, on_unused_input='ignore', updates=updates)
	burnt_in = burn_in_f(weights, bias_v, bias_h)

	[samples2, prop_down2], updates2 = theano.scan(rbm_vhv, outputs_info=[init2, None ], non_sequences=[W, bv, bh, nv, nh], n_steps = n_samples*sample_every)

	final_samples = samples2[T.arange(0, n_samples*sample_every, sample_every)]	

	return_f = theano.function([W, bv, bh, init2], final_samples, on_unused_input='ignore', updates=updates2)
	return return_f(weights, bias_v, bias_h, burnt_in)


def sample_rbm_2wise(weights, weights_h, bias_h, bias_v, n_samples, burnin=1000, sample_every=100):
	print "sampling from rbm"

	W = T.matrix('W')
	Wh = T.vector('Wh')
	bv = T.vector('bv')
	bh = T.vector('bh')

	nv, nh = weights.shape

	adder = np.zeros((int(nh/2), nh), dtype=theano.config.floatX)
	for i in xrange(len(adder)):
		adder[i, 2*i] = 1
		adder[i, 2*i+1] = 1
	adder = theano.shared(adder)

	init = theano_rng.binomial(p=0.5, size=(nv,), ndim=1, dtype=theano.config.floatX)

	[samples, prop_down], updates = theano.scan(rbm_vhv_2wise, outputs_info=[init, None],  non_sequences=[W, Wh, bv, bh, nv, nh, adder], n_steps = burnin)
	init2 = samples[-1]	

	burn_in_f = theano.function([W, Wh, bv, bh], init2, on_unused_input='ignore', updates=updates)
	burnt_in = burn_in_f(weights, weights_h, bias_v, bias_h)

	[samples2, prop_down2], updates2 = theano.scan(rbm_vhv_2wise, outputs_info=[init2, None ], non_sequences=[W, Wh, bv, bh, nv, nh, adder], n_steps = n_samples*sample_every)

	final_samples = samples2[T.arange(0, n_samples*sample_every, sample_every)]	

	return_f = theano.function([W, Wh, bv, bh, init2], final_samples, on_unused_input='ignore', updates=updates2)
	return return_f(weights, weights_h, bias_v, bias_h, burnt_in)


def random_rbm(nv, nh, nsamples, sample_every=10, burnin=100):

	path = 'data/rbm_samples_%d_%d_%d_%d.pkl'%(nv, nsamples, sample_every, burnin)
	if os.path.exists(path):
		f = open(path)
		samples, params = pickle.load(f)
		f.close()
		return samples, params

	w_bound = np.sqrt(6. / ( nv + nh ))
	
	w0 = np.asarray(np.random.uniform(size=(nv, nh), low=-w_bound, high=w_bound), dtype=theano.config.floatX)
	bh0 = np.asarray(np.random.uniform(size=nh, low=-w_bound, high=w_bound), dtype=theano.config.floatX)
	bv0 = np.asarray(np.random.uniform(size=nv, low=-w_bound, high=w_bound), dtype=theano.config.floatX)


	samples = sample_rbm(w0, bh0, bv0, nsamples, sample_every=sample_every, burnin=burnin)

	params = [w0, bh0, bv0]

	f = open(path, 'w')
	pickle.dump([samples, params], f)
	f.close()

	return samples, params

def dec2bin(i, N):
	bini = bin(i)[2:]
	bini = '0'*(N - len(bini)) + bini
	
	return np.asarray(list(bini), dtype=int)


def generate_states(nv):

	states = np.zeros((2**nv, nv))

	row = np.zeros(nv)
	
	f = lambda s : dec2bin(s,nv)
	states = map(f, np.arange(nv))

	return np.asarray(states)

def compute_probs(params, states):

	W, bh, bv = params


	energies = energy_np(states, W, bh, bv)

	ps = np.exp(-energies)
	Z = ps.sum()

	return Z, ps/Z, energies


def compute_dkl(params, params_fit):

	assert(params[0].shape == params_fit[0].shape)
	
	nv, nh = params[0].shape
	
	# we are only interested in visible states
	states = generate_states(nv)

	Z1, P1, E1 = compute_probs(params, states)
	Z2, P2, E2 = compute_probs(params_fit, states)

	DKL = P2 * np.log(P2/P1)

	return DKL.sum()


def compute_likelihood(params, X):

	Z, P, E = compute_probs(params, X)

	L = - E.mean() - np.log(Z)
	
	L2 = np.log(P).mean()

#	print 'L1, L2'
#	print L
#	print L2
	#assert ( np.abs(L-L2) < 0.0001)

	return L






