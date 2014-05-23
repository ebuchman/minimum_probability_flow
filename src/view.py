import matplotlib.pyplot as plt
import numpy as np
import pickle
import rbm
import os
from mpf import sample_and_save
from rbm import rbm_vhv_np, rbm_vhv_2wise_np

def view_simple():
    
    path = os.path.join('data', 'results', 'mnist_deep_samples_nh500_nh500_.pkl')
    f = open(path)
    d = pickle.load(f)
    f.close()
    samples, params = d

    w, bh, bv = params[0]

    n = int(w.shape[0]**0.5)

    plt.figure()
    for i, s in enumerate(samples):
        plt.subplot(8,8,i+1)
        plt.imshow(s.reshape(n, n), cmap='gray')
    plt.show()

def view_one_step():
    path = os.path.join('data', 'results', 'mnist_deep_samples_nh500_nh500_.pkl')
    path = os.path.join('data', 'results', 'mnist_samples_2wise_nh500_ne50_lr0.001000.pkl')

    f = open(path)
    d = pickle.load(f)
    f.close()
    samples, params = d

    w, wh, bh, bv = params

    datapath = 'mnistX_binary_lowres.pkl'
    f = open(datapath, 'r')
    X  = pickle.load(f)
    f.close()

    X = X[:100]

    #V = rbm_vhv_np(X, w, bh, bv)
    V = rbm_vhv_2wise_np(X, w, wh, bh, bv)

    fig = plt.figure()
    plt.title('real')
    for i in xrange(0, 50):
        plt.subplot(10,10,2*i+1)
        plt.imshow(X[i].reshape(14, 14))
        plt.subplot(10,10,2*i+2)
        plt.imshow(V[i].reshape(14, 14))

    V = rbm_vhv_np(X, w, bh, bv)

    fig = plt.figure()
    plt.title('almost')
    for i in xrange(0, 50):
        plt.subplot(10,10,2*i+1)
        plt.imshow(X[i].reshape(14, 14))
        plt.subplot(10,10,2*i+2)
        plt.imshow(V[i].reshape(14, 14))


    nv, nh = w.shape
    wbound = np.sqrt(6. / (nv + nh))
    w = np.random.uniform(size=w.shape, low=-wbound, high=wbound)
    wh = np.random.uniform(size=nh/2, low=-wbound, high=wbound)
    bh = np.asarray(np.zeros(nh))
    bv = np.asarray(np.zeros(nv))

    #V = rbm_vhv_np(X, w, bh, bv)
    V = rbm_vhv_2wise_np(X, w, wh, bh, bv)

    fig = plt.figure()
    plt.title('random')
    for i in xrange(0, 50):
        plt.subplot(10,10,2*i+1)
        plt.imshow(X[i].reshape(14, 14))
        plt.subplot(10,10,2*i+2)
        plt.imshow(V[i].reshape(14, 14))

    plt.show()


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
    view_simple()
    #view_one_step()
    #load_params_generate_samples()
    #load_samples_and_params()
