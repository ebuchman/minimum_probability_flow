
import numpy as np
import theano
import time
import pickle
#import matplotlib.pyplot as plt
import socket

def load_data(ndata=1000):
	print "loading data"
	if 'MacBook' in socket.gethostname():
		f = open('/Users/BatBuddha/Programming/Datasets/mnist/mnistSMALL_binary.pkl')
	else:
		f = open('/export/mlrg/ebuchman/datasets/mnist_binary.pkl')

	d = pickle.load(f)
	f.close()

	if len(d) == 3:
		tr, val, tst = d
		tr_x, tr_y = tr
	else:
		tr_x = d 

	tr_x = tr_x.astype(theano.config.floatX)

	return tr_x[:ndata]

def plot_samples(samples):
	N_plots = len(samples)
	w_plots = np.ceil(N_plots**0.5)

	ww_image = len(samples[0])
	w_image = np.int(ww_image**0.5)

	plt.figure()
	for i in xrange(N_plots):
		plt.subplot(w_plots, w_plots, i+1)
		plt.imshow(samples[i].reshape(w_image, w_image))

	plt.show()

