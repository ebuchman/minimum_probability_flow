import numpy as np
import time, pickle, sys
import theano
import theano.tensor as T

from util import load_data, plot_samples
from rbm import energy, rbm_vhv, sample_rbm, random_rbm, compute_dkl, compute_likelihood

def random_theta(nv, nh):
	w_bound = np.sqrt(6. / (nv + nh))
	w0 = np.asarray(np.random.uniform(size=(nv*nh), low=-w_bound, high=w_bound), dtype=theano.config.floatX)
	bh0 = np.asarray(np.zeros(nv), dtype=theano.config.floatX)
	bv0 = np.asarray(np.zeros(nh), dtype=theano.config.floatX)

	return np.asarray(np.concatenate((w0, bh0, bv0)), dtype=theano.config.floatX)

def split_theta(theta, nv, nh):

	param_idx = 0
	w = theta[param_idx:param_idx+nv*nh].reshape(nv, nh)
	param_idx += nv*nh

	bh = theta[param_idx:param_idx+nh]
	param_idx += nh
	
	bv = theta[param_idx:param_idx+nv]
	param_idx += nv

#	assert(param_idx == theta_shape)

	return [w, bh, bv]
	


class MPF(object):
	def __init__(self, n_visible, n_hidden):
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		D, DH = self.n_visible, self.n_hidden

		theta_shape = D*DH + D + DH

		w_bound = np.sqrt(6. / ( D + DH))
		self.theta = theano.shared(value=np.zeros(theta_shape, dtype=theano.config.floatX))
		self.theta.name = 'theta'

		param_idx = 0

		self.W = self.theta[param_idx : param_idx + D*DH].reshape((D, DH), 2)
		self.W.name = 'W'
		param_idx += D*DH
		
		self.bh = self.theta[param_idx : param_idx + DH]
		self.bh.name = 'bh'
		param_idx += DH

		self.bv = self.theta[param_idx : param_idx + D]
		self.bv.name='bv'
		param_idx += D

	def rbm_K_dK(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden
		W, bh, bv = self.W, self.bh, self.bv

		Y = X.reshape((effective_batch_size, 1, D), 3) * T.ones((1, D, 1)) #tile out data vectors (repeat each one D times)
		Y = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 

		Z = T.exp(0.5*energy(X, W, bh, bv))
		Z.name = 'Z'
		Y = T.exp(-0.5*energy(Y, W, bh, bv)).reshape((effective_batch_size, D), 2)
		Y.name = 'Y'

		K = T.sum(Z*T.sum(Y, axis=-1))  / effective_batch_size #+ 0.0001*T.sum(W**2)
		K.name = 'K'

		G = T.grad(K, self.theta)
		G.name = 'G'

		return K, G


class metaMPF():
	def __init__(self, nv, nh, batch_size, n_epochs):
		self.nv = nv
		self.nh = nh
		self.batch_size = batch_size
		self.n_epochs = n_epochs

		self.ready()

	def ready(self):
		self.X = T.matrix('X')
		self.mpf = MPF(n_visible = self.nv, n_hidden = self.nh)
	
	def shared_dataset(self, data_X, borrow=True):
		return theano.shared(np.asarray(data_X, dtype=theano.config.floatX))

	def fit(self, train_X, optimizer, param_init = None):
		n_train, n_vis = train_X.shape
		batch_size = self.batch_size

		assert(n_vis == self.nv)
	
		train_X = self.shared_dataset(train_X)
		n_batches = np.ceil(n_train / float(batch_size)).astype('int')

		index, n_ex = T.iscalars('batch_index', 'n_ex')

		batch_start = index*batch_size
		batch_stop = T.minimum(n_ex, (index + 1)*batch_size)
		effective_batch_size = batch_stop - batch_start

		
		K, dK = self.mpf.rbm_K_dK(self.X, effective_batch_size)

		print "compiling theano functions"
		get_batch_size = theano.function([index, n_ex], effective_batch_size, allow_input_downcast=True, name='get_batch_size')
		batch_cost = theano.function([index, n_ex], [K, dK], givens={self.X: train_X[batch_start:batch_stop, :]}, allow_input_downcast=True, name='batch_cost')

		print "actually training ..."

		if optimizer == 'sgd':
			pass

		elif optimizer == 'cg' or optimizer == 'bfgs':

			def train_fn(theta_value):

				self.mpf.theta.set_value(np.asarray(theta_value, dtype=theano.config.floatX), borrow=True)
				#theta.set_value(theta_value, borrow=True)
				train_losses_grads = [batch_cost(i, n_train) for i in xrange(n_batches)]

				train_losses = [i[0] for i in train_losses_grads]
				train_grads = [i[1] for i in train_losses_grads]

				train_batch_sizes = [get_batch_size(i, n_train) for i in xrange(n_batches)]

				return np.average(train_losses, weights=train_batch_sizes), np.average(train_grads, weights=train_batch_sizes, axis=0)

			###############
			# TRAIN MODEL #
			###############
			if param_init == None:
				theta.set_value(random_theta(D, DH))
			else:
				w0, bh0, bv0 = param_init
				self.mpf.theta.set_value(np.asarray(np.concatenate((w0, bh0, bv0)), dtype=theano.config.floatX))

			from scipy.optimize import minimize
			if optimizer == 'cg':
				pass
			elif optimizer == 'bfgs':
				print 'using bfgs'
				#theta_opt, f_theta_opt, info = fmin_l_bfgs_b(train_fn, self.mpf.theta.get_value(), iprint=1, maxfun=self.n_epochs)
				start = time.time()
				result_obj = minimize(train_fn, self.mpf.theta.get_value(), jac=True, method='BFGS', options={'maxiter':11})
				end = time.time()
				theta_opt = result_obj.x

		elif optimizer == 'sof':

			def train_fn(theta_value, i):
				self.mpf.theta.set_value(np.asarray(theta_value, dtype=theano.config.floatX), borrow=True)

				train_losses, train_grads = batch_cost(i, n_train)
				
				return train_losses, train_grads

			###############
			# TRAIN MODEL #
			###############
			if param_init == None:
				theta.set_value(random_theta(D, DH))
			else:
				w0, bh0, bv0 = param_init
				self.mpf.theta.set_value(np.asarray(np.concatenate((w0, bh0, bv0)), dtype=theano.config.floatX))


			print 'using sof'
			sys.path.append('/export/mlrg/ebuchman/Programming/Sum-of-Functions-Optimizer')
			from sfo import SFO
			optimizer = SFO(train_fn, self.mpf.theta.get_value(), np.arange(n_batches))
			start = time.time()
			theta_opt = optimizer.optimize(num_passes = self.n_epochs)
			end = time.time()

		
		self.mpf.theta.set_value(theta_opt.astype(theano.config.floatX), borrow=True)
		return end-start

def cd_rbm(nv, nh, batch_size, n_epochs, X, params, k):
	from cd_rbm import test_rbm
	if k == 0:
		print 'k must be nonzero'
		quit()

	start = time.time()
	W, bh, bv, t= test_rbm(X, learning_rate=0.1, training_epochs=n_epochs, batch_size=batch_size,
             n_chains=1, n_samples=10, output_folder='rbm_plots',
             n_hidden=5, nvisible=10, k=k)
	end = time.time()
	params_fit = [W, bh, bv]
	dkl = compute_dkl(params, params_fit)
	#L_true = compute_likelihood(params, X)
	#L_fit = compute_likelihood(params_fit, X)

	#print 'fit dkl: ', dkl
	#print 'true nll', -L_true
	#print 'fit nll', -L_fit

	print 'saving to results_cd.dat!'
	f = open('results.dat', 'a')
	f.write('CD-%d & %.4f & %.4f & %d \\\\\n'%(k, dkl, t, n_epochs))
	f.close()



def train(nv, nh, batch_size, n_epochs, X, optimizer, params, param_init, k=0):
	if 'cd' in optimizer:
		cd_rbm(nv, nh, batch_size, n_epochs, X, params, k)
	else:
		param_init[0] = param_init[0].reshape(nv*nh)

		print 'initializing mpf'
		model = metaMPF(nv, nh, batch_size, n_epochs)
		print "training", optimizer
		start = time.time()
		t = model.fit(train_X = X, optimizer = optimizer, param_init = param_init)
		end = time.time()
		params_fit = split_theta(model.mpf.theta.get_value(), nv, nh)

		dkl = compute_dkl(params, params_fit)

		print 'saving to results.dat!'
		f = open('results.dat', 'a')
		f.write('MPF(%s) & %.4f & %.4f & %d \\\\\n'%(optimizer, dkl, t, n_epochs))
		f.close()


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

