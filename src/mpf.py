import numpy as np
import time, pickle, sys, math
import theano
import theano.tensor as T
import os

from util import load_data, plot_samples
from rbm import energy, energy2wise, debug_energy2wise, sample_rbm, sample_rbm_2wise, rbm_vhv, sample_rbm, random_rbm, compute_dkl, compute_likelihood
from debug import print_debug
from params import random_theta, split_theta, sample_and_save, save_params

DEBUG = 0

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

	def connect_off_states(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden
		W, bh, bv = self.W, self.bh, self.bv

		onehots = T.eye(D)
		blanks = T.zeros(D)
	
		eX = energy(X, W, bh, bv)	
		eO = energy(onehots, W, bh, bv)
		eB = energy(blanks, W, bh, bv)

		onehot_dif = eX.dimshuffle(0, 'x') - eO.dimshuffle('x', 0)

		Z_onehot = T.exp(0.5*(onehot_dif))
		Z_onehot = T.sum(Z_onehot) / effective_batch_size

	#	return Z_onehot 

		Z_blank = T.exp(0.5*(eX - eB))
		Z_blank = T.sum(Z_blank) / effective_batch_size

		return Z_blank

		#return Z_onehot + Z_blank

	# single epoch full mnist downsamp batch_size 100 : 16 seconds
	def rbm_K(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden
		W, bh, bv = self.W, self.bh, self.bv

		#one bit flipped connected states
		Y = X.reshape((effective_batch_size, 1, D), 3) * T.ones((1, D, 1)) #tile out data vectors (repeat each one D times)
		Y1 = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 

		# minimal activation connected states
		onehots = T.eye(D)
		blanks = T.zeros(D)
	
		eX = energy(X, W, bh, bv)	
		eY = energy(Y1, W, bh, bv)
		eO = energy(onehots, W, bh, bv)
		eB = energy(blanks, W, bh, bv)
		
		edif = eX.dimshuffle(0, 'x') - eY #- eB #- eO.dimshuffle('x', 0) - eB

		Z = T.exp(0.5*edif)
		K = T.sum(Z) / effective_batch_size
		K.name = 'K'

		K = T.cast(K, 'float32')
		return K 

	def K_connected_states(self, x, W, bh, bv, D, DH):
		Y = x.reshape((1, D), 2) * T.ones((D, 1))
		Y = (Y + T.eye(D))%2

		Z = T.exp(0.5*(energy(x, W, bh, bv) - energy(Y, W, bh, bv)))
		Z = T.sum(Z)
		G = theano.grad(Z, self.theta)
		return Z, G

	# single epoch full mnist downsamp batchsize 100 : 58 seconds 
	def rbm_K2G(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden
		W, bh, bv = self.W, self.bh, self.bv

		# scan over data vectors.  for each, compute contribution to K from connected (non-data) states
		[K, G], upds = theano.scan(self.K_connected_states, non_sequences=[W, bh, bv, D, DH], sequences=X)
		K = T.sum(K) / effective_batch_size
		G = T.sum(G, axis=0) / effective_batch_size
		K.name = 'K'
		G.name = 'G'
		K = T.cast(K, 'float32')
		G = T.cast(G, 'float32')
		return K, G

	def rbm_K_2wise(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden

		W, Wh, bh, bv = self.W, self.Wh, self.bh, self.bv

		Y = X.reshape((effective_batch_size, 1, D), 3) * T.ones((1, D, 1)) #tile out data vectors (repeat each one D times)
		Y = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 

		Z = T.exp(0.5*(energy2wise(X, W, Wh, bh, bv, DH).dimshuffle(0, 'x') - energy2wise(Y, W, Wh, bh, bv, DH)))
		K = T.sum(Z) / effective_batch_size
		K.name = 'K'
		return K

	def debug_rbm_K_2wise(self, X, effective_batch_size):
		D, DH = self.n_visible, self.n_hidden

		W, Wh, bh, bv = self.W, self.Wh, self.bh, self.bv

		Y = X.reshape((effective_batch_size, 1, D), 3) * T.ones((1, D, 1)) #tile out data vectors (repeat each one D times)
		Y1 = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 

		wx_b1, e_wx_b1, pairsum1, first1, pairprod1, pairprodWh1, logterm1, hidden_term1, vbias_term1, energy1 = debug_energy2wise(X, W, Wh, bh, bv, DH)
		wx_b2, e_wx_b2, pairsum2, first2, pairprod2, pairprodWh2, logterm2, hidden_term2, vbias_term2, energy2 = debug_energy2wise(Y1, W, Wh, bh, bv, DH)

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
		self.current_epoch = 0

		self.ready()

    def ready(self):
        self.X = T.matrix('X')
        self.mpf = MPF(n_visible = self.nv, n_hidden = self.nh, k = self.k)

    def shared_dataset(self, data_X, borrow=True):
        return theano.shared(np.asarray(data_X, dtype=theano.config.floatX))

    def fit(self, train_X, optimizer, param_init = None, sample_every=None):
		self.opt = optimizer
		n_train, n_vis = train_X.shape
		batch_size = self.batch_size

		if sample_every == None:
			sample_every = 10000000

		#theano.config.profile = True
		#theano.config.exception_verbosity='high'

		assert(n_vis == self.nv)

		train_X = self.shared_dataset(train_X)
		n_batches = np.ceil(n_train / float(batch_size)).astype('int')

		# theano variables for managing data (index minibatches, n examples in batch)
		index, n_ex = T.iscalars('batch_index', 'n_ex')
		batch_start = index*batch_size
		batch_stop = T.minimum(n_ex, (index + 1)*batch_size)
		effective_batch_size = batch_stop - batch_start

		# theano variables for learning
		lr = T.scalar('lr', dtype=theano.config.floatX)
		mom = T.scalar('mom', dtype=theano.config.floatX)

		if self.k == 1:
			# this one is for scaning over a batch and getting connectivity for each example
			# return grads too because T.grads through scan is awful
			# takes ~3x longer, but can experiment connectivity
			#K, grads = self.mpf.rbm_K2G(self.X, effective_batch_size)

			# this tiles out the minibatch matrix into a 3D tensor to compute connectivity
			#K, offs, y, y1, z= self.mpf.rbm_K(self.X, effective_batch_size)
			K = self.mpf.rbm_K(self.X, effective_batch_size)

		elif self.k == 2:
			if DEBUG:
				return_values = self.mpf.debug_rbm_K_2wise(self.X, effective_batch_size)	
				K = return_values[-1]
			else:
				K = self.mpf.rbm_K_2wise(self.X, effective_batch_size)
		else:
			raise('NotImplemented')

		reg = self.L1_reg * self.mpf.L1 + self.L2_reg * self.mpf.L2
		reg_grad = T.grad(reg, self.mpf.theta)

		# if not scan (tile out matrix into tensor)
		cost = K + reg
		grads = T.grad(cost, self.mpf.theta)

		# otherwise
		#grads = grads + reg_grad

		if param_init == None:
			self.mpf.theta.set_value(random_theta(D, DH, k=self.k))
		else:
			self.mpf.theta.set_value(np.asarray(np.concatenate(param_init), dtype=theano.config.floatX))

		if optimizer == 'sgd':
			updates = []
			theta = self.mpf.theta
			theta_update = self.mpf.theta_update

			upd = mom * theta_update - lr * grads
			updates.append((theta_update, upd))
			updates.append((theta, theta + upd))

			print 'compiling theano function'
			if DEBUG:
				return_values = list(return_values)
				return_values.append(cost)
				return_values.append(grads)
				train_model = theano.function(inputs=[index, n_ex, lr, mom], outputs=return_values, updates=updates, givens={self.X: train_X[batch_start:batch_stop]})
			else:
				train_model = theano.function(inputs=[index, n_ex, lr, mom], outputs=cost, updates=updates, givens={self.X: train_X[batch_start:batch_stop]})

			self.current_epoch = 0
			start = time.time()
			learning_rate_init = self.learning_rate
			while self.current_epoch < self.n_epochs:
				print 'epoch:', self.current_epoch
				self.current_epoch += 1
				effective_mom = self.final_momentum if self.current_epoch > self.momentum_switchover else self.initial_momentum

				avg_epoch_cost = 0
				last_debug = None
				for minibatch_idx in xrange(n_batches):
					avg_cost = train_model(minibatch_idx, n_train, self.learning_rate, effective_mom)
					#print '\t\t', np.isnan(gr).sum(), np.isnan(yy).sum(), np.isnan(yy1).sum(), np.isnan(zz).sum()
					if DEBUG:
						return_values, avg_cost, gradients = avg_cost[:-2], avg_cost[-2], avg_cost[-1]
						print_debug(return_values, last_debug)
						last_debug = return_values
					avg_epoch_cost += avg_cost
					#print '\t', minibatch_idx, avg_cost
				print '\t avg epoch cost:', avg_epoch_cost/n_batches
				self.learning_rate *= self.learning_rate_decay

				theta_fit = split_theta(self.mpf.theta.get_value(), self.mpf.n_visible, self.mpf.n_hidden, k=self.mpf.k)
				if (self.current_epoch % sample_every == 0):
					sample_and_save(theta_fit, self.mpf.n_hidden, self.current_epoch, learning_rate_init, self.mpf.k, self.opt)

			theta_opt = self.mpf.theta.get_value()
			end = time.time()

		elif optimizer == 'cg' or optimizer == 'bfgs':
			print "compiling theano functions"
			get_batch_size = theano.function([index, n_ex], effective_batch_size, name='get_batch_size')
			batch_cost_grads = theano.function([index, n_ex], [cost, grads], givens={self.X: train_X[batch_start:batch_stop, :]}, name='batch_cost')
			batch_cost = theano.function([index, n_ex], cost, givens={self.X: train_X[batch_start:batch_stop, :]}, name='batch_cost')
			batch_grads = theano.function([index, n_ex], grads, givens={self.X: train_X[batch_start:batch_stop, :]}, name='batch_cost')


			def train_fn_cost_grads(theta_value):
				print 'nbatches', n_batches

				self.mpf.theta.set_value(np.asarray(theta_value, dtype=theano.config.floatX), borrow=True)
				train_losses_grads = [batch_cost_gradst(i, n_train) for i in xrange(n_batches)]

				train_losses = [i[0] for i in train_losses_grads]
				train_grads = [i[1] for i in train_losses_grads]

				train_batch_sizes = [get_batch_size(i, n_train) for i in xrange(n_batches)]

				print len(train_losses), len(train_grads)
				print train_losses[0].shape, train_grads[0].shape
				returns = np.average(train_losses, weights=train_batch_sizes), np.average(train_grads, weights=train_batch_sizes, axis=0)
				return returns


			def train_fn_cost(theta_value):
				print 'nbatches', n_batches

				self.mpf.theta.set_value(np.asarray(theta_value, dtype=theano.config.floatX), borrow=True)
				train_costs = [batch_cost(i, n_train) for i in xrange(n_batches)]
				train_batch_sizes = [get_batch_size(i, n_train) for i in xrange(n_batches)]

				return np.average(train_costs, weights=train_batch_sizes)

			def train_fn_grads(theta_value):
				print 'nbatches', n_batches

				self.mpf.theta.set_value(np.asarray(theta_value, dtype=theano.config.floatX), borrow=True)
				train_grads = [batch_grads(i, n_train) for i in xrange(n_batches)]
				train_batch_sizes = [get_batch_size(i, n_train) for i in xrange(n_batches)]

				return np.average(train_grads, weights=train_batch_sizes, axis=0)


			###############
			# TRAIN MODEL #
			###############
			def my_callback():
				print 'wtf'

			from scipy.optimize import minimize
			from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
			if optimizer == 'cg':
				pass
			elif optimizer == 'bfgs':
				print 'using bfgs'
				#theta_opt, f_theta_opt, info = fmin_l_bfgs_b(train_fn, self.mpf.theta.get_value(), iprint=1, maxfun=self.n_epochs)
				start = time.time()
				disp = True
				print 'ready to minimize'
				#result_obj = minimize(train_fn, self.mpf.theta.get_value(), jac=True, method='BFGS', options={'maxiter':self.n_epochs, 'disp':disp}, callback=my_callback())
				#theta_opt = fmin_bfgs(f=train_fn_cost, x0=self.mpf.theta.get_value(), fprime=train_fn_grads, disp=1, maxiter=self.n_epochs)
				theta_opt, fff, ddd = fmin_l_bfgs_b(func=train_fn_cost, x0=self.mpf.theta.get_value(), fprime=train_fn_grads, disp=1, maxiter=self.n_epochs)
				print 'done minimize ya right'
				end = time.time()

		elif optimizer == 'sof':
			print "compiling theano functions"
			batch_cost_grads = theano.function([index, n_ex], [cost, grads], givens={self.X: train_X[batch_start:batch_stop, :]}, name='batch_cost')
			batch_cost = theano.function([index, n_ex], cost, givens={self.X: train_X[batch_start:batch_stop, :]}, name='batch_cost')
			batch_grads = theano.function([index, n_ex], grads, givens={self.X: train_X[batch_start:batch_stop, :]}, name='batch_cost')


			def train_fn(theta_value, i):
				self.mpf.theta.set_value(np.asarray(theta_value, dtype=theano.config.floatX), borrow=True)

				train_losses, train_grads = batch_cost_grads(i, n_train)
				
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
			print 'n batches', n_batches
			print 'n epochs' , self.n_epochs
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


