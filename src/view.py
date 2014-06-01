import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import rbm
import os
from mpf import sample_and_save
from rbm import rbm_vhv_np, rbm_vhv_2wise_np, sample_h_given_v_np, sample_v_given_h_np, energy_np

def save_view(samples, nh, nepochs, lr):
	nsamps, nv = samples.shape
	w_img = int(nv**0.5)
	w_plts = int(nsamps**0.5+1)

	plt.figure()
	for i, s in enumerate(samples):
		plt.subplot(w_plts, w_plts,i+1)
		plt.imshow(s.reshape(w_img, w_img), cmap='gray')
	plt.savefig(os.path.join('data', 'plots', 'samples_mnist_nh%d_ne%s_lr%f'%(nh, nepochs, lr)))

def view_params(save_path, N=10, save=False):
    path = os.path.join('data', 'results', save_path)
    f = open(path)
    d = pickle.load(f)
    f.close()

    if len(d) == 2:
        samples, params = d
    else:
        params = d 

    if 'deep' in save_path and not 'cd' in save_path:
        w, bh, bv = params[0]
    else:
        w, bh, bv = params

    n = int(w.shape[0]**0.5)

    plt.figure()
    for i in xrange(N*N):
        plt.subplot(N, N,i+1)
        plt.imshow(w.T[i].reshape(n, n), cmap='gray')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    if save:
        plt.savefig('param_plots_'+save_path[:-4])
    else:
        plt.show()

def view_simple(save_path):
    
    path = os.path.join('data', 'results', save_path)
    f = open(path)
    d = pickle.load(f)
    f.close()
    samples, params = d

    if 'deep' in save_path:
        w, bh, bv = params[0]
    else:
        w, bh, bv = params

    n = int(w.shape[0]**0.5)

    plt.figure()
    for i, s in enumerate(samples):
        plt.subplot(8,8,i+1)
        plt.imshow(s.reshape(n, n), cmap='gray')
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.show()


def samples(save_path, save=False):
    path = os.path.join('data', 'results', save_path)

    print 'opening samples, params'
    f = open(path)
    d = pickle.load(f)
    f.close()
    if len(d) == 2:
        samples, params = d
    else:
        params = d

    datapath = 'mnist_binary.pkl'
    print 'opening', datapath
    f = open(os.path.join('/Users', 'BatBuddha', 'Programming', 'Datasets', 'mnist', datapath))
    X = pickle.load(f)
    f.close()

    tr, val, tst = X
    trX = tr[0]
    tstX = tst[0]

    sample_every = 5
    n_samplers = 20
    n_samples = 10
    nlayers= len(params)


    if 'deep' in save_path and not 'cd' in save_path:
        w, bh, bv = params[0]
    else:
        w, bh, bv = params
    nv, nh = w.shape

    rand_idx = np.random.randint(len(tstX) - n_samplers)
    X = tstX[rand_idx:rand_idx+n_samplers]

    X = np.random.binomial(1, 0.5, (n_samplers, 28*28))

    samples = [X]
    for j in xrange(n_samples):        
        print j
        for i in xrange(sample_every):
            H = sample_h_given_v_np(X, w, bh, len(bh))
            X = sample_v_given_h_np(H, w, bv, len(bv))
        samples.append(X)
        
    w_img = int(w.shape[0]**0.5)

    fig = plt.figure()
    plt.title('real')
    print len(samples)
    print len(samples[0])
    for i in xrange(0, n_samples):
        for j in xrange(n_samplers):
            print i, j
            plt.subplot(n_samples,n_samplers, i*n_samplers + j+1)
            plt.imshow(samples[i][j].reshape(w_img, w_img), cmap='gray')
            plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
            plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.tight_layout()
    if save:
        plt.savefig('samples_plots_'+save_path[:-4])
    else:
        plt.show()    
    
    compare_energies(samples, [w, bh, bv], trX, tstX)

def compare_energies(samples, params, trX, tstX):
    n_chains = len(samples[0])
    w, bh, bv = params

    randoms = np.random.binomial(1, 0.5, (100, 28*28))
    blank = np.zeros(28*28)
    onehot = np.eye(28*28)

    rand_energy = energy_np(randoms, w, bh, bv).mean()
    tr_energy = energy_np(trX, w, bh, bv).mean()
    tst_energy = energy_np(tstX, w, bh, bv).mean()
    blnk_energy = energy_np(blank, w, bh, bv)
    onehot_energy = energy_np(onehot, w, bh, bv).mean()
    

    print 'average train energy:', tr_energy
    print 'average test energy:', tst_energy
    for i in xrange(1, len(samples)):
        sampd_energy = energy_np(samples[i], w, bh, bv).mean()
        print 'average sampled energy %d:'%i, sampd_energy
    print 'average random energy:', rand_energy
    print 'blank energy:', blnk_energy
    print ' average one hot energy:', onehot_energy


    





def deep_samples2(save_path):
    path = os.path.join('data', 'results', save_path)

    print 'opening samples, params'
    f = open(path)
    d = pickle.load(f)
    f.close()
    samples, params = d

    sample_every = 1000
    burnin = 1000
    n_samples = 99
    nlayers= len(params)


    w, bh, bv = params[-1]
    nv, nh = w.shape

    H = np.random.binomial(1, 0.5, (n_samples, nh))
    print 'burnin'
    for i in xrange(burnin):
        v = sample_v_given_h_np(H, w, bv, len(bv), mean=True)
        H = sample_h_given_v_np(v, w, bh, len(bh))

    X = H
    # prop down    
    for l, prm in enumerate(params[::-1]):
        w, bh, bv = prm
        print w.shape, bh.shape, bv.shape
        mean = False if l==nlayers-1 else True
        X = sample_v_given_h_np(X, w, bv, len(bv))
        
    print X.shape
    w_img = int(w.shape[0]**0.5)
    fig = plt.figure()
    plt.title('real')
    for i in xrange(0, n_samples):
        plt.subplot(10,10,i+1)
        plt.imshow(X[i].reshape(w_img, w_img))
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.show()    

def deep_samples(save_path):
    path = os.path.join('data', 'results', save_path)

    print 'opening samples, params'
    f = open(path)
    d = pickle.load(f)
    f.close()
    samples, params = d

    sample_every = 1000
    burnin = 1000
    n_samples = 50
    nlayers= len(params)


    w, bh, bv = params[-1]
    nv, nh = w.shape

    H = np.random.binomial(1, 0.5, nh)
    print 'burnin'
    for i in xrange(burnin):
        v = sample_v_given_h_np(H, w, bv, len(bv), mean=True)
        H = sample_h_given_v_np(v, w, bh, len(bh))

    samples = []
    for i in xrange(n_samples):
        print 'sample', i
        for j in xrange(sample_every):
            v = sample_v_given_h_np(H, w, bv, len(bv), mean=True)
            H = sample_h_given_v_np(v, w, bh, len(bh))

        X = H
        # prop down    
        for l, prm in enumerate(params[::-1]):
            w, bh, bv = prm
            print w.shape, bh.shape, bv.shape
            mean = False if l==nlayers-1 else True
            X = sample_v_given_h_np(X, w, bv, len(bv))
        samples.append(X)
        

    w_img = int(nv**0.5)
    fig = plt.figure()
    plt.title('real')
    for i in xrange(0, n_samples):
        plt.subplot(10,10,2*i+1)
        plt.imshow(samples[i].reshape(w_img, w_img))
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    




def view_one_step(save_path):
    path = os.path.join('data', 'results', save_path)

    print 'opening samples, params'
    f = open(path)
    d = pickle.load(f)
    f.close()
    samples, params = d



    #datapath = 'mnistX_binary_lowres.pkl'
    datapath = 'mnist_binary.pkl'
    print 'opening', datapath
    f = open(os.path.join('/Users', 'BatBuddha', 'Programming', 'Datasets', 'mnist', datapath))
    X = pickle.load(f)
    f.close()

    if len(X) == 3:
        X = X[0][0]

    X = X[:100]
    nsteps = 100

    V = X
    for step in xrange(nsteps):

        if 'deep' in save_path:
            # prop up
            print 'prop up'
            nlayers = len(params)
            for i, prm in enumerate(params):
                w, bh, bv = prm
                print w.shape, bh.shape, bv.shape
                mean = False if i==nlayers-1 else True
                V = sample_h_given_v_np(V, w, bh, len(bh), mean=mean)
            # prop down
            print 'prop down'
            for i, prm in enumerate(params[::-1]):
                w, bh, bv = prm
                print w.shape, bh.shape, bv.shape
                mean = False if i==nlayers-1 else True
                V = sample_v_given_h_np(V, w, bv, len(bv))

        else:
            if len(params) == 3:
                w, bh, bv = params
                V = rbm_vhv_np(X, w, bh, bv)
            elif len(params) == 4:
                w, wh, bh, bv = params
                V = rbm_vhv_2wise_np(X, w, wh, bh, bv)

    w_img = int(w.shape[0]**0.5)
    fig = plt.figure()
    plt.title('real')
    for i in xrange(0, 50):
        plt.subplot(10,10,2*i+1)
        plt.imshow(X[i].reshape(w_img, w_img))
        plt.subplot(10,10,2*i+2)
        plt.imshow(V[i].reshape(w_img, w_img))
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')


    V = X
    for step in xrange(nsteps):
        V = rbm_vhv_np(V, w, bh, bv)

    fig = plt.figure()
    plt.title('almost')
    for i in xrange(0, 50):
        plt.subplot(10,10,2*i+1)
        plt.imshow(X[i].reshape(w_img, w_img))
        plt.subplot(10,10,2*i+2)
        plt.imshow(V[i].reshape(w_img, w_img))
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
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
    save_path = 'mnist_samples_nh200_ne1_lr0.010000.pkl'
    save_path = 'mnist_deep_samples_nh200_nh200_nh200_.pkl'
    save_path = 'mnist_samples_sof_nh200_ne1_lr0.010000.pkl'
    save_path = 'mnist_deep_samples_sgd_nh200_nh200_nh200_nh200_.pkl'
    save_path = 'mnist_deep_samples_sgd_nh500_.pkl'
    #save_path = 'mnist_cd_deeplearning_orig.pkl'
    #save_path = 'ctrl_c_params_sof_nh500_ne0'

    #view_params(save_path, 14)#, save=True)
    #view_simple(save_path)
    #view_one_step(save_path)
    #deep_samples2(save_path)
    samples(save_path)#, save=True)
    #load_params_generate_samples()





    #load_samples_and_params()
