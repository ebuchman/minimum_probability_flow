import numpy as np
import time, pickle, sys, math
import theano
import theano.tensor as T
import os

from util import load_data, plot_samples
from rbm import energy, energy2wise, debug_energy2wise, sample_rbm, sample_rbm_2wise, rbm_vhv, sample_rbm, random_rbm, compute_dkl, compute_likelihood
from debug import print_debug

DEBUG = 0

def random_theta(nv, nh, k=1):
	w_bound = np.sqrt(6. / (nv + nh))
	w0 = np.asarray(np.random.uniform(size=(nv*nh), low=-w_bound, high=w_bound), dtype=theano.config.floatX)
	bh0 = np.asarray(np.zeros(nv), dtype=theano.config.floatX)
	bv0 = np.asarray(np.zeros(nh), dtype=theano.config.floatX)

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

def sample_and_save(params_fit, nh, n_epochs, learning_rate, k, nsamps=20, sample_every=1000, burnin=1000):
	if k == 1:
		w, bh, bv = params_fit
		samples = sample_rbm(w, bh, bv, nsamps, sample_every=sample_every, burnin=burnin)
		save_name = 'mnist_samples_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate)
	elif k == 2:
		w, wh, bh, bv = params_fit
		samples = sample_rbm_2wise(w, wh, bh, bv, nsamps, sample_every=sample_every, burnin=burnin)
		save_name = 'mnist_samples_2wise_nh%d_ne%d_lr%2f.pkl'%(nh, n_epochs, learning_rate)

	print save_name
	f = open(os.path.join('data', 'results', save_name), 'w')
	pickle.dump([samples, params_fit], f)
	f.close()

def nCr(n, r):
	f = math.factorial
	return f(n) / f(r) / f(n-r)

class MPF(object):
	def __init__(self, n_visible, n_hidden, k=1):
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.k = k

		D, DH = self.n_visible, self.n_hidden

		theta_shape = D*DH + D + DH

		# accomodate for weights between groups of k hiddens
		if k == 2:
			assert (DH % 2 == 0)
			wh_size = (DH/2)
			theta_shape += wh_size

		self.theta = theano.shared(value=np.zeros(theta_shape, dtype=theano.config.floatX))
		self.theta.name = 'theta'

		self.theta_update = theano.shared(value=np.zeros(theta_shape, dtype=theano.config.floatX))

		param_idx = 0

		self.W = self.theta[param_idx : param_idx + D*DH].reshape((D, DH), 2)
		self.W.name = 'W'
		param_idx += D*DH

		if k == 2:
			self.Wh = self.theta[param_idx : param_idx + wh_size]
			self.Wh.name = 'Wh'
			param_idx += wh_size

		self.bh = self.theta[param_idx : param_idx + DH]
		self.bh.name = 'bh'
		param_idx += DH

		self.bv = self.theta[param_idx : param_idx + D]
		self.bv.name='bv'
		param_idx += D

		self.L1 = abs(self.W.sum())
		self.L2 = (self.W ** 2).sum()
		if k == 2:
			self.L1 += abs(self.Wh.sum())
			self.L2 += (self.Wh ** 2).sum()

	def rbm_K(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden
		W, bh, bv = self.W, self.bh, self.bv

		Y = X.reshape((effective_batch_size, 1, D), 3) * T.ones((1, D, 1)) #tile out data vectors (repeat each one D times)
		Y = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 

		Z = T.exp(0.5*(energy(X, W, bh, bv).dimshuffle(0, 'x') - energy(Y, W, bh, bv)))
		K = T.sum(Z) / effective_batch_size
		K.name = 'K'
		'''
		Z = T.exp(0.5*energy(X, W, bh, bv))
		Z.name = 'Z'
		Y = T.exp(-0.5*energy(Y, W, bh, bv)).reshape((effective_batch_size, D), 2)
		Y.name = 'Y'

		K = T.sum(Z*T.sum(Y, axis=-1))  / effective_batch_size #+ 0.0001*T.sum(W**2)
		K.name = 'K'
		'''
		return K

	def rbm_K_2wise(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden

		adder = np.zeros((DH/2, DH), dtype=theano.config.floatX)
		for i in xrange(len(adder)):
			adder[i, 2*i] = 1
			adder[i, 2*i+1] = 1
		adder = theano.shared(adder)

		W, Wh, bh, bv = self.W, self.Wh, self.bh, self.bv

		Y = X.reshape((effective_batch_size, 1, D), 3) * T.ones((1, D, 1)) #tile out data vectors (repeat each one D times)
		Y = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 

		Z = T.exp(0.5*(energy2wise(X, W, Wh, bh, bv, adder, DH).dimshuffle(0, 'x') - energy2wise(Y, W, Wh, bh, bv, adder, DH)))
		K = T.sum(Z) / effective_batch_size
		K.name = 'K'
		return K

	def debug_rbm_K_2wise(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden

		adder = np.zeros((DH/2, DH), dtype=theano.config.floatX)
		for i in xrange(len(adder)):
			adder[i, 2*i] = 1
			adder[i, 2*i+1] = 1
		adder = theano.shared(adder)

		W, Wh, bh, bv = self.W, self.Wh, self.bh, self.bv

		Y = X.reshape((effective_batch_size, 1, D), 3) * T.ones((1, D, 1)) #tile out data vectors (repeat each one D times)
		Y1 = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 

		wx_b1, e_wx_b1, pairsum1, first1, pairprod1, pairprodWh1, logterm1, hidden_term1, vbias_term1, energy1 = debug_energy2wise(X, W, Wh, bh, bv, adder, DH)
		wx_b2, e_wx_b2, pairsum2, first2, pairprod2, pairprodWh2, logterm2, hidden_term2, vbias_term2, energy2 = debug_energy2wise(Y1, W, Wh, bh, bv, adder, DH)

		Z = T.exp(0.5*(energy1.dimshuffle(0, 'x') - energy2))
		K = T.sum(Z) / effective_batch_size
		K.name = 'K'
		return Y, Y1, \
			wx_b1, e_wx_b1, pairsum1, first1, pairprod1, pairprodWh1, logterm1, hidden_term1, vbias_term1, energy1,\
			wx_b2, e_wx_b2, pairsum2, first2, pairprod2, pairprodWh2, logterm2, hidden_term2, vbias_term2, energy2, \
			Z, K
			
class metaMPF():
    def __init__(self, nv, nh, batch_size, n_epochs, learning_rate=0.01, learning_rate_decay=1, initial_momentum=0.5, final_momentum=0.9, momentum_switchover=5, L1_reg=0.00, L2_reg=0.00, k=1):
        self.nv = nv
        self.nh = nh
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.momentum_switchover = momentum_switchover
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.k = k

        self.ready()

    def ready(self):
        self.X = T.matrix('X')
        self.mpf = MPF(n_visible = self.nv, n_hidden = self.nh, k = self.k)

    def shared_dataset(self, data_X, borrow=True):
        return theano.shared(np.asarray(data_X, dtype=theano.config.floatX))

    def fit(self, train_X, optimizer, param_init = None, sample_every=None):
		n_train, n_vis = train_X.shape
		batch_size = self.batch_size

		if sample_every == None:
			sample_every = self.n_epochs
		#theano.config.profile = True
		#theano.config.exception_verbosity='high'

		assert(n_vis == self.nv)

		train_X = self.shared_dataset(train_X)
		n_batches = np.ceil(n_train / float(batch_size)).astype('int')

		index, n_ex = T.iscalars('batch_index', 'n_ex')

		batch_start = index*batch_size
		batch_stop = T.minimum(n_ex, (index + 1)*batch_size)
		effective_batch_size = batch_stop - batch_start

		lr = T.scalar('lr', dtype=theano.config.floatX)
		mom = T.scalar('mom', dtype=theano.config.floatX)

		if self.k == 1:
			K = self.mpf.rbm_K(self.X, effective_batch_size)
		elif self.k == 2:
			if DEBUG:
				return_values = self.mpf.debug_rbm_K_2wise(self.X, effective_batch_size)	
				K = return_values[-1]
			else:
				K = self.mpf.rbm_K_2wise(self.X, effective_batch_size)
		else:
			raise('NotImplemented')
		cost = K + self.L1_reg * self.mpf.L1 + self.L2_reg * self.mpf.L2

		grads = T.grad(cost, self.mpf.theta)
		grads.name = 'G'


		print "compiling theano functions"
		get_batch_size = theano.function([index, n_ex], effective_batch_size, name='get_batch_size')
		batch_cost = theano.function([index, n_ex], [cost, grads], givens={self.X: train_X[batch_start:batch_stop, :]}, name='batch_cost')

		if param_init == None:
			self.mpf.theta.set_value(random_theta(D, DH, k=self.k))
		else:
			self.mpf.theta.set_value(np.asarray(np.concatenate(param_init), dtype=theano.config.floatX))

		print "actually training ..."

		if optimizer == 'sgd':
			updates = []
			theta = self.mpf.theta
			theta_update = self.mpf.theta_update

			upd = mom * theta_update - lr * grads
			updates.append((theta_update, upd))
			updates.append((theta, theta + upd))

			if DEBUG:
				return_values = list(return_values)
				return_values.append(cost)
				return_values.append(grads)
				train_model = theano.function(inputs=[index, n_ex, lr, mom], outputs=return_values, updates=updates, givens={self.X: train_X[batch_start:batch_stop]})
			else:
				train_model = theano.function(inputs=[index, n_ex, lr, mom], outputs=cost, updates=updates, givens={self.X: train_X[batch_start:batch_stop]})

			epoch = 0
			start = time.time()
			learning_rate_init = self.learning_rate
			while epoch < self.n_epochs:
				print 'epoch:', epoch
				epoch += 1
				effective_mom = self.final_momentum if epoch > self.momentum_switchover else self.initial_momentum

				avg_epoch_cost = 0
				last_debug = None
				for minibatch_idx in xrange(n_batches):
					avg_cost = train_model(minibatch_idx, n_train, self.learning_rate, effective_mom)
					if DEBUG:
						return_values, avg_cost, gradients = avg_cost[:-2], avg_cost[-2], avg_cost[-1]
						print_debug(return_values, last_debug)
						last_debug = return_values
					avg_epoch_cost += avg_cost
					#print '\t', minibatch_idx, avg_cost
				print '\t avg epoch cost:', avg_epoch_cost/n_batches
				self.learning_rate *= self.learning_rate_decay

				theta_fit = split_theta(self.mpf.theta.get_value(), self.mpf.n_visible, self.mpf.n_hidden, k=self.mpf.k)
				if (epoch % sample_every == 0):
					sample_and_save(theta_fit, self.mpf.n_hidden, epoch, learning_rate_init, self.mpf.k)

			theta_opt = self.mpf.theta.get_value()
			end = time.time()

		elif optimizer == 'cg' or optimizer == 'bfgs':

			def train_fn(theta_value):

				self.mpf.theta.set_value(np.asarray(theta_value, dtype=theano.config.floatX), borrow=True)
				train_losses_grads = [batch_cost(i, n_train) for i in xrange(n_batches)]

				train_losses = [i[0] for i in train_losses_grads]
				train_grads = [i[1] for i in train_losses_grads]

				train_batch_sizes = [get_batch_size(i, n_train) for i in xrange(n_batches)]

				return np.average(train_losses, weights=train_batch_sizes), np.average(train_grads, weights=train_batch_sizes, axis=0)

			###############
			# TRAIN MODEL #
			###############

			from scipy.optimize import minimize
			if optimizer == 'cg':
				pass
			elif optimizer == 'bfgs':
				print 'using bfgs'
				#theta_opt, f_theta_opt, info = fmin_l_bfgs_b(train_fn, self.mpf.theta.get_value(), iprint=1, maxfun=self.n_epochs)
				start = time.time()
				disp = True
				result_obj = minimize(train_fn, self.mpf.theta.get_value(), jac=True, method='BFGS', options={'maxiter':self.n_epochs, 'disp':disp})
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


def train(nv, nh, batch_size, n_epochs, X, optimizer, params, param_init, cdk=0, lr=0.01, lrd=1, mom_i=0.5, mom_f =0.9, mom_switch=5, L1_reg=0.00, L2_reg=0.00, k=1):
    if 'cd' in optimizer:
        cd_rbm(nv, nh, batch_size, n_epochs, X, params, cdk)
    else:
        param_init[0] = param_init[0].reshape(nv*nh)

        print 'initializing mpf'
        model = metaMPF(nv, nh, batch_size, n_epochs, learning_rate=lr, learning_rate_decay=lrd, initial_momentum=mom_i, final_momentum=mom_f, momentum_switchover=mom_switch, L1_reg=L1_reg, L2_reg=L2_reg, k=k)
        print "training", optimizer
        start = time.time()
        t = model.fit(train_X = X, optimizer = optimizer, param_init = param_init)
        end = time.time()
        params_fit = split_theta(model.mpf.theta.get_value(), nv, nh, k=k)

        dkl = compute_dkl(params, params_fit)
        print 'dkl:', dkl
        print 'saving to results.dat!'
        f = open('results.dat', 'a')
        f.write('MPF(%s) & %.4f & %.4f & %d \\\\\n'%(optimizer, dkl, t, n_epochs))
        f.close()


