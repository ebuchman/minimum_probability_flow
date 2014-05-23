import matplotlib.pyplot as plt
import numpy as np
import pickle
import rbm
import os
from mpf import sample_and_save

def load_samples_and_params():
    for j in xrange(1, 10):
        print j
        nh = 500
        k = 2
        if k == 1:
            path = 'mnist_samples_nh%d_ne%d_lr0.001000.pkl'%(nh, j*10)
        elif k == 2:
            path = 'mnist_samples_2wise_nh%d_ne%d_lr0.001000.pkl'%(nh, j*10)
        #path = 'mnist_samples_nh100_ne%d_lr0.001000.pkl'%(j*10)
        f = open(os.path.join('data', 'results', path))

        d = pickle.load(f)
        f.close()

        samples, params = d

        if len(params) == 3:
            w, bh, bv = params
        elif len(params) == 4:
            w, wh, bh, bv = params

        n = int(w.shape[0]**0.5)

        plt.figure()
        for i, s in enumerate(samples):
            plt.subplot(5,5,i+1)
            plt.imshow(s.reshape(n, n), cmap='gray')

        plt.savefig(os.path.join('data', 'plots', 'plots_nh%d_k%d_epoch%d'%(nh, k, j)))

        plt.figure()
        for i, s in enumerate(w.T):
            if i == 10*10:
                break
            plt.subplot(10,10,i+1)
            plt.imshow(s.reshape(n,n), cmap='gray')
        plt.savefig(os.path.join('data', 'plots', 'params_nh%d_k%d_epoch%d'%(nh, k, j)))

def load_params_generate_samples():
    for j in xrange(1, 10):
        print j
        nh = 500
        k = 2
        if k == 1:
            path = 'mnist_samples_nh%d_ne%d_lr0.001000.pkl'%(nh, j*10)
        elif k == 2:
            path = 'mnist_samples_2wise_nh%d_ne%d_lr0.001000.pkl'%(nh, j*10)
        #path = 'mnist_samples_nh100_ne%d_lr0.001000.pkl'%(j*10)
        f = open(os.path.join('data', 'results', path))

        d = pickle.load(f)
        f.close()

        samples, params = d

        if len(params) == 3:
            w, bh, bv = params
        elif len(params) == 4:
            w, wh, bh, bv = params


        n = int(w.shape[0]**0.5)

    
        sample_and_save(params, nh, j*10, 0.001000, k, nsamps=20, sample_every=1000, burnin=1000)


if __name__ == '__main__':
    load_params_generate_samples()
    #load_samples_and_params()
