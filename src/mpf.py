import numpy as np
import pickle
import theano
import theano.tensor as T

print "loading data"
f = open('/Users/BatBuddha/Programming/Datasets/mnist/mnistSMALL_binary.pkl')
tr, val, tst = pickle.load(f)
f.close()

tr_x, tr_y = tr

N, D = tr_x.shape
DH = 100
print N, D, DH

print "compiling theano"

W = theano.shared(np.random.rand(D, DH)/100)

def energy(X, W=W):
    E = T.log(1 + T.exp( T.dot(X, W) ))
    return E.sum(axis = -1)


X = T.matrix('X')
eps = T.scalar('eps')

Y = X.reshape((N, D, 1), 3) * T.ones((1, 1, D)) #tile out data vectors (repeat each one D times)
Y = (Y + T.eye(D).reshape((1, D, D), 3))%2 # flip each bit once 
# (N, D, D) - first D is num dimensions, second D is num states that this state is connected too (in this case, its D since we are only connected to states one bit flip away)
# we should swap these dimensions so we multiply the vectors by the weight matrix
Y = Y.dimshuffle(0, 2, 1)

Z = T.exp(0.5*energy(X))
Y = T.exp(-0.5*energy(Y))

K = T.sum(Z*Y.sum(axis=1))

G = T.grad(K, W)

upd = -eps*G
updates = [(W, W+upd)]


f = theano.function([X, eps], [K], updates=updates)

#xx = np.random.binomial(1, 0.5, (N, D))

print "training ... "
epsilon = 0.1
for i in xrange(10):
    k = f(tr_x, epsilon)
    
    print k
    epsilon *= 0.999




