minimum_probability_flow
========================

Minimum Probability Flow (http://arxiv.org/abs/0906.4779) implemented in python, using theano.

MPF is an unsupervised learning algorithm for parameter estimation in probabilistic graphical models from the exponential family. It is an elegant solution to the problem posed by contrastive divergence.

This code uses MPF to train restricted boltzmann machines (RBM) and partially restricted boltzmann machines (with connectivity between hidden units).

Optimization options:
- SGD       (minibatch stochastic gradient descent with momentum)
- BFGS 		(scipy.optimize)
- SOF       (https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer must be installed).

