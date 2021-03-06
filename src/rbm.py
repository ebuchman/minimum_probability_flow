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

def energy2wise_np(X, W, Wh, bh, bv):
	nh = len(bh)

	adder = np.zeros((nh/2, nh))
	for i in xrange(nh/2):
		adder[i][2*i] = 1
		adder[i][2*i+1] = 1

	wx_b = np.dot(X, W) + bh
	e_wx_b = np.exp(wx_b)

	pairsum = np.dot(e_wx_b, adder.T)
	first = e_wx_b.T[np.arange(0, nh, 2)].T
	pairprod = pairsum*first - first**2

	hidden_term = np.sum(np.log(1 + pairsum + pairprod*np.exp(Wh)), axis=-1)

	vbias_term = np.dot(X, bv)
	return -hidden_term - vbias_term

def energy(X, W, bh, bv):
	wx_b = T.dot(X, W) + bh
	vbias_term = T.dot(X, bv)
	hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=-1)
	return -hidden_term - vbias_term

def energy2wise(X, W, Wh, bh, bv, nh):
	wx_b = T.dot(X, W) + bh
	e_wx_b = T.exp(wx_b)

	adder = np.zeros((nh/2, nh), dtype=theano.config.floatX)
	for i in xrange(len(adder)):
		adder[i, 2*i] = 1
		adder[i, 2*i+1] = 1
	adder = theano.shared(adder)
	pairsum = T.dot(e_wx_b, adder.T)
	first = e_wx_b.T[T.arange(0, nh, 2)].T
	pairprod = pairsum*first - first**2

	hidden_term = T.sum(T.log(1 + pairsum + pairprod*T.exp(Wh)), axis=-1)

	vbias_term = T.dot(X, bv)
	return -hidden_term - vbias_term


def debug_energy2wise(X, W, Wh, bh, bv, nh):
	wx_b = T.dot(X, W) + bh
	e_wx_b = T.exp(wx_b)

	adder = np.zeros((nh/2, nh), dtype=theano.config.floatX)
	for i in xrange(len(adder)):
		adder[i, 2*i] = 1
		adder[i, 2*i+1] = 1
	adder = theano.shared(adder)
	pairsum = T.dot(e_wx_b, adder.T)
	first = e_wx_b.T[T.arange(0, nh, 2)].T
	pairprod = pairsum*first - first**2

	pairprodWh = pairprod*T.exp(Wh)

	logterm = T.log(1 + pairsum + pairprodWh)

	hidden_term = T.sum(logterm, axis=-1)

	vbias_term = T.dot(X, bv)

	energy =  -hidden_term - vbias_term

	return wx_b, e_wx_b, pairsum, first, pairprod, pairprodWh, logterm, hidden_term, vbias_term, energy


theano_rng = rng_mrg.MRG_RandomStreams(seed=100)
#theano_rng = RandomStreams(100)

def rbm_vhv_np(v, W, bh, bv):
    h = sample_h_given_v_np(v, W, bh, len(bh))
    v = sample_v_given_h_np(h, W, bv, len(bv))
    return v

def rbm_vhv_2wise_np(v, W, Wh, bh, bv):
    h = sample_h_given_v_2wise_np(v, W, Wh, bh, len(bh))
    v = sample_v_given_h_np(h, W, bv, len(bv))
    return v

def sample_v_given_h_np(h, W, bv, nv, mean=False):
    prop_down = 1./(1+np.exp(-(np.dot(h, W.T) + bv)))
    if mean:
        return prop_down
    else:
        return np.random.binomial(n=1, p = prop_down, size=(h.shape[0], nv))

def sample_h_given_v_np(v, W, bh, nh, mean=False):
    prop_up = 1./(1+np.exp(-(np.dot(v, W) + bh)))
    if mean:
        return prop_up
    else:
        return np.random.binomial(n=1, p = prop_up, size=(v.shape[0], nh))

def sample_h_given_v(v, W, bh, nh):
	prop_up = T.nnet.sigmoid(T.dot(v, W) + bh)
	h = theano_rng.binomial(n=1, p = prop_up, dtype=theano.config.floatX, size=(nh,), ndim=1)
	return h, prop_up

def sample_v_given_h(h, W, bv, nv):
	prop_down = T.nnet.sigmoid(T.dot(W, h) + bv)
	v = theano_rng.binomial(n=1, p = prop_down, dtype=theano.config.floatX, size=(nv,), ndim=1)
	return v, prop_down

def rbm_vhv(v, W, bv, bh, nv, nh):
	h, prop_up = sample_h_given_v(v, W, bh, nh)
	v, prop_down = sample_v_given_h(h, W, bv, nv)
	return v, prop_down

def rbm_hvh(h, W, bv, bh, nv, nh):
	v, prop_down = sample_v_given_h(h, W, bv, nv)
	h, prop_up = sample_h_given_v(v, W, bh, nh)
	return h, prop_up

def sample_h_given_v_2wise_np(v, W, Wh, bh, nh):
    phi = np.dot(v, W) + bh
    ephi = np.exp(phi)

    adder = np.zeros((nh/2, nh))
    for i in xrange(len(adder)):
        adder[i, 2*i] = 1
        adder[i, 2*i+1] = 1

    pairsum = np.dot(ephi, adder.T)
    first = ephi.T[np.arange(0, nh, 2)].T
    pairprod = pairsum*first - first**2
    pairterm = pairprod*np.exp(Wh)

    wobble = 1 + pairsum + pairterm

    pairterm_broadcast = np.kron(pairterm, np.ones(2))
    wobble_broadcast = np.kron(wobble, np.ones(2))

    prop_up = (ephi + pairterm_broadcast) / wobble_broadcast

    h = np.random.binomial(n=1, p = prop_up, size=(v.shape[0], nh))

    return h

#this only works if v is one dimensional cuz of dimshuffle...
def sample_h_given_v_2wise(v, W, Wh, bh, nh):
	phi = T.dot(v, W) + bh
	ephi = T.exp(phi)

	adder = np.zeros((nh/2, nh), dtype=theano.config.floatX)
	for i in xrange(len(adder)):
		adder[i, 2*i] = 1
		adder[i, 2*i+1] = 1
	adder = theano.shared(adder)
	# wobble =  1 + exp(phi_2i) + exp(phi_{2i+1}) + exp(phi_2i + phi_{21+1} + Wh_i)
	# p(h_2i = 1 | v) = (exp(phi_2i) + exp(phi_2i + phi_{21+1} + Wh_i ) / wobble
	# p(h_{2i+1} = 1 | v) = (exp(phi_2i) + exp(phi_2i + phi_{2i+1} + Wh_i )) / wobble
	# the second term is the same in both - the pair term.  but it must be broadcasted (the kron!)
	# dotting by adder returns a vector of half the size of sums of pairs of elements

	pairsum = T.dot(ephi, adder.T)
	first = ephi.T[T.arange(0, nh, 2)].T
	pairprod = pairsum*first - first**2
	pairterm = pairprod*T.exp(Wh)

	wobble = 1 + pairsum + pairterm

	pairterm_broadcast = kron(pairterm.dimshuffle(0, 'x'), T.ones(2))
	wobble_broadcast = kron(wobble.dimshuffle(0, 'x'), T.ones(2))

	prop_up = (ephi + pairterm_broadcast) / wobble_broadcast

	h = theano_rng.binomial(n=1, p = prop_up, dtype=theano.config.floatX, size=(nh,), ndim=1)

	return h
	
def rbm_vhv_2wise(v, W, Wh, bv, bh, nv, nh):
	h = sample_h_given_v_2wise(v, W, Wh, bh, nh)
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

	init = theano_rng.binomial(p=0.5, size=(nv,), ndim=1, dtype=theano.config.floatX)

	[samples, prop_down], updates = theano.scan(rbm_vhv_2wise, outputs_info=[init, None],  non_sequences=[W, Wh, bv, bh, nv, nh], n_steps = burnin)
	init2 = samples[-1]	

	burn_in_f = theano.function([W, Wh, bv, bh], init2, on_unused_input='ignore', updates=updates)
	burnt_in = burn_in_f(weights, weights_h, bias_v, bias_h)

	[samples2, prop_down2], updates2 = theano.scan(rbm_vhv_2wise, outputs_info=[init2, None ], non_sequences=[W, Wh, bv, bh, nv, nh], n_steps = n_samples*sample_every)

	final_samples = samples2[T.arange(0, n_samples*sample_every, sample_every)]	

	return_f = theano.function([W, Wh, bv, bh, init2], final_samples, on_unused_input='ignore', updates=updates2)
	return return_f(weights, weights_h, bias_v, bias_h, burnt_in)


def random_rbm(nv, nh, nsamples, sample_every=10, burnin=100, k=1):

	path = 'data/rbm_samples_nv%d_nsamp%d_sampevery%d_burn%d_k%d.pkl'%(nv, nsamples, sample_every, burnin, k)
	if os.path.exists(path):
		f = open(path)
		samples, params = pickle.load(f)
		f.close()
		return samples, params

	w_bound = np.sqrt(6. / ( nv + nh ))
	
	w0 = np.asarray(np.random.uniform(size=(nv, nh), low=-w_bound, high=w_bound), dtype=theano.config.floatX)
	bh0 = np.asarray(np.random.uniform(size=nh, low=-w_bound, high=w_bound), dtype=theano.config.floatX)
	bv0 = np.asarray(np.random.uniform(size=nv, low=-w_bound, high=w_bound), dtype=theano.config.floatX)

	if k == 1:
		samples = sample_rbm(w0, bh0, bv0, nsamples, sample_every=sample_every, burnin=burnin)
		params = [w0, bh0, bv0]
	elif k == 2:
		assert(nh % 2 == 0)
		wh = np.asarray(np.random.uniform(size=(nh/2), low=-w_bound, high=w_bound), dtype=theano.config.floatX)
		samples = sample_rbm_2wise(w0, wh, bh0, bv0, nsamples, sample_every=sample_every, burnin=burnin)
		params = [w0, wh, bh0, bv0]
	

	f = open(path, 'w')
	pickle.dump([samples, params], f)
	f.close()

	return samples, params

def dec2bin(i, N):
	bini = bin(i)[2:]
	bini = '0'*(N - len(bini)) + bini
	
	return np.asarray(list(bini), dtype=int)

def bin2dec(state):
	state = list(state[::-1])
	r = 0
	while len(state):
		r *= 2
		r += state.pop()
	return r


def generate_states(nv):
	f = lambda s : dec2bin(s,nv)
	states = map(f, np.arange(2**nv))

	return np.asarray(states)

def compute_probs(params, states):

	if len(params) == 3:
		W, bh, bv = params
		energies = energy_np(states, W, bh, bv)
	elif len(params) == 4:
		W, Wh, bh, bv = params
		energies = energy2wise_np(states, W, Wh, bh, bv)

	ps = np.exp(-energies)
	Z = ps.sum()

	return Z, ps/Z, energies

def compute_probs_mrf(params, states):
	W, bv = params
	energies = energy_mrf(params, states)
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

def sample_mrf(params, n_samples=10, sample_every=10, n_flips=1):
	W, bv = params
	nv = len(bv)

	start = np.random.binomial(1, 0.5, nv)
	samples = [start]	
	rejections = 0
	for i in xrange(n_samples):
		for j in xrange(sample_every):
			to_flip = np.random.randint(nv, size=n_flips)
			flipped = start.copy()

			prior_p = np.exp(-energy_mrf(params, flipped)) # p before flip
			flipped[to_flip] = (flipped[to_flip] + 1)%2 # flip bit
			post_p = np.exp(-energy_mrf(params, flipped)) # p after flip
			
			if post_p > prior_p:
				start = flipped
			elif np.random.rand() < post_p/prior_p:
				start = flipped
			else:
				rejections += 1
				start = start

		samples.append(start)

	return samples, rejections

def energy_mrf(params, states):
	W, bv = params
	return - np.sum(np.dot(states, W)*states, -1) - np.dot(states, bv)

if __name__ == '__main__':
	nv = 8

	W = np.random.randn(nv, nv)*np.tri(nv, k=-1).T
	bv = np.random.randn(nv)
	params = [W, bv]
	
	samples, rej  =	sample_mrf(params, n_samples=100000, sample_every=1, n_flips=3)
	print rej

	states = generate_states(nv)
	Z, p, E = compute_probs_mrf(params, states)

	samples_int = map(bin2dec, samples)
	counts, xs = np.histogram(samples_int, bins=2**nv)
	sampled_p = counts / float(counts.sum())

	print len(sampled_p[sampled_p <= 0.00001])

	print np.sum(sampled_p * np.log(sampled_p / p))
	print np.sum((sampled_p - p)**2)
