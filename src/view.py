import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import rbm
import os
from mpf import sample_and_save
from rbm import rbm_vhv_np, rbm_vhv_2wise_np, sample_h_given_v_np, sample_v_given_h_np, energy_np, sample_mrf


def view_params(save_path, N=10, save=False):
    params = get_params(save_path)
    w = params[0]
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


def view_params2(save_path, N=10, save=False):
    params = get_params(save_path)

    if 'deep' in save_path and not 'cd' in save_path:
        w, bh, bv = params[0]
    else:
        w, bh, bv = params

    n = int(w.shape[0]**0.5)


    w = 1./(1 + np.exp(-(w + bv.reshape(bv.shape[0], 1))))

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



def get_params(path):
    print 'opening samples, params', path
    f = open(path)
    d = pickle.load(f)
    f.close()
    if len(d) == 2:
        samples, params = d
    else:
        params = d
    return params

def get_data(datapath = 'mnist_binary.pkl'):
    print 'opening', datapath
    f = open(os.path.join('/Users', 'BatBuddha', 'Programming', 'Datasets', 'mnist', datapath))
    X = pickle.load(f)
    f.close()

    if len(X) == 3:
        tr, val, tst = X
        trX = tr[0]
        tstX = tst[0]
    else:
        trX = X
    return trX

def _sample_rbm(samples, n_samples, n_samplers, sample_every, params, tX):
    w, bh, bv = params
    X = tX
    for j in xrange(n_samples):        
        print j
        for i in xrange(sample_every):
            H = sample_h_given_v_np(X, w, bh, len(bh))
            X = sample_v_given_h_np(H, w, bv, len(bv))
        samples.append(X)

def _sample_mrf(samples, n_samples, n_samplers, sample_every, params, tX):
    rand_idx = np.random.randint(len(tX) - n_samplers)

    for i in xrange(n_samplers):
        #X = np.random.binomial(1, 0.5, nv)
        X = tX[rand_idx+i]
        print i
        s, r = sample_mrf(params, n_samples=n_samples, sample_every=1000, start=X)
        samples.append(s)

def _sample_srbm(samples, n_samples, n_samplers, sample_every, params, tX):
    w, wv, bh, bv = params
    rand_idx = np.random.randint(len(tX) - n_samplers)

    for k in xrange(n_samplers):
        X = tX[rand_idx+k]
        these_samps = [X]
        for j in xrange(n_samples):        
            print k, j
            for i in xrange(sample_every):
                H = sample_h_given_v_np(X.reshape(1, X.shape[0]), w, bh, len(bh))
                V = sample_v_given_h_np(H, w, bv, len(bv))
                _bv = bv + np.dot(w, H.T).ravel()
                V = V.reshape(V.shape[1])
                X, r = sample_mrf([wv, _bv], n_samples=1, sample_every=1000, start = V)
                X = np.asarray(X[-1]) # X was [start, sample]
            these_samps.append(X)
        samples.append(these_samps)

def samples_mrf(save_path, save=False):
    params = get_params(save_path)

    sample_every = 5
    n_samplers = 20
    n_samples = 10

    w, bv = params
    nv = len(bv)
   
    if nv == 28*28:
        datapath = 'mnist_binary.pkl'
    elif nv == 14*14:
        datapath = 'mnistX_binary_lowres.pkl'

    X = get_data(datapath)

    samples = []
    _sample_mrf(samples, n_samples, n_samplers, sample_every, params, X)
        
    w_img = int(w.shape[0]**0.5)


    fig = plt.figure()
    plt.title('real')
    for i in xrange(0, n_samples):
        for j in xrange(n_samplers):
            print i, j
            plt.subplot(n_samples,n_samplers, i*n_samplers + j+1)
            plt.imshow(np.asarray(samples[j][i]).reshape(w_img, w_img), cmap='gray')
            plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
            plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.tight_layout()
    if save:
        plt.savefig('samples_plots_'+save_path[:-4])
    else:
        plt.show()    
    
    compare_energies(samples, [w, bh, bv], trX, tstX)


def samples_srbm(save_path, save=False):

    params = get_params(save_path)
    nv = params[0].shape[0]
    if nv == 28*28:
        datapath = 'mnist_binary.pkl'
    elif nv == 14*14:
        datapath = 'mnistX_binary_lowres.pkl'
    X = get_data(datapath)

    sample_every = 1
    n_samplers = 20
    n_samples = 10
    nlayers= len(params)

    if 'deep' in save_path and not 'cd' in save_path:
        params = params[0]

    rand_idx = np.random.randint(len(X) - n_samplers)
    #X = X[rand_idx:rand_idx+n_samplers]

    samples = []
    _sample_srbm(samples, n_samples, n_samplers, sample_every, params, X)
        
    w_img = int(X[0].shape[0]**0.5)

    fig = plt.figure()
    plt.title('real')
    print len(samples)
    print len(samples[0])
    for j in xrange(n_samplers):
        for i in xrange(0, n_samples):
            print i, j
            plt.subplot(n_samples,n_samplers, j*n_samples + i+1)
            plt.imshow(samples[j][i].reshape(w_img, w_img), cmap='gray')
            plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
            plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    plt.tight_layout()
    if save:
        plt.savefig('samples_plots_'+save_path[:-4])
    else:
        plt.show()    
    
    compare_energies(samples, [w, bh, bv], trX, tstX)



def samples_rbm(save_path, save=False):

    params = get_params(save_path)
    X = get_data()

    sample_every = 100
    n_samplers = 20
    n_samples = 10
    nlayers= len(params)

    if 'deep' in save_path and not 'cd' in save_path:
        params = params[0]

    rand_idx = np.random.randint(len(X) - n_samplers)
    X = X[rand_idx:rand_idx+n_samplers]

    nv = len(X[0])

    #X = np.random.binomial(1, 0.5, (n_samplers, nv))

    samples = [X]
    _sample_rbm(samples, n_samples, n_samplers, sample_every, params, X)
        
    w_img = int(X[0].shape[0]**0.5)

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
    
    compare_energies(samples, params, X, X)

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
    



if __name__ == '__main__':
    save_path = 'mnist_samples_nh200_ne1_lr0.010000.pkl'
    save_path = 'mnist_deep_samples_nh200_nh200_nh200_.pkl'
    save_path = 'mnist_samples_sof_nh200_ne1_lr0.010000.pkl'
    save_path = 'mnist_deep_samples_sgd_nh200_nh200_nh200_nh200_.pkl'
    save_path = 'mnist_deep_samples_sgd_nh500_.pkl'
    #save_path = 'mnist_params_mrf_sgd_nh200_.pkl'
    save_path = 'mnist_params_srbm_sgd_nh500_.pkl'
    save_path = 'mnist_params_srbm_sof_nh500_.pkl'
    #save_path = 'mnist_cd_deeplearning_orig.pkl'
    #save_path = 'ctrl_c_params_sof_nh500_ne0'
    #save_path = 'mnist_params_srbm_sgd_nh500_bsize20_ntrain10000_lr0.0010_dec0.9900_l20.00100_l10.00000_mom0.50_0.90_10_ne5_.pklne10_.pklne15_.pklne20_.pklne25_.pklne30_.pkl'
    #save_path = 'mnist_params_srbm_sgd_nh500_bsize20_ntrain10000_lr0.0010_dec0.9900_l20.00100_l10.00000_mom0.50_0.90_10_ne5_.pklne10_.pklne15_.pklne20_.pklne25_.pklne30_.pklne35_.pklne40_.pklne45_.pklne50_.pkl'
   # save_path = 'mnist_params_srbm_sgd_nh500_bsize20_ntrain10000_lr0.0010_dec0.9900_l20.01000_l10.00100_mom0.50_0.90_10_ne50_.pkl'

    #save_path = 'mnist_params_srbm_sgd_nh200_bsize20_ntrain10000_lr0.0010_dec0.9900_l20.00100_l10.00000_mom0.50_0.90_10_ne50_.pkl'
    #save_path = 'mnist_params_mrfsrbm_sof_nh500_bsize100_ntrain10000_lr0.0010_dec0.9900_l20.00100_l10.00000_mom0.50_0.90_10_ne1_.pkl'
    #save_path = 'mnist_params_mrfsrbm_sof_nh500_bsize1000_ntrain10000_lr0.0010_dec0.9900_l20.00100_l10.00000_mom0.50_0.90_10_ne5_.pkl'

    save_path = 'mnist_params_mrf_sgd_nh500_bsize20_ntrain10000_lr0.0010_dec0.9900_l20.00100_l10.00000_mom0.50_0.90_10_ne20_.pkl'
    save_path = 'mnist_params_rbm_sgd_nh500_bsize20_ntrain10000_lr0.0010_dec0.9900_l20.00100_l10.00000_mom0.50_0.90_10_ne20_.pkl'
    save_path = 'mnist_params_rbm_sgd_nh500_bsize100_ntrain50000_lr0.0010_dec0.9900_l20.00000_l10.00000_mom0.50_0.90_10_ne30_.pkl'
    save_path = os.path.join('data', 'results', save_path)


    #view_params(save_path, 14)#, save=True)
    samples_rbm(save_path)#, save=True)
    #samples_srbm(save_path)





    #load_samples_and_params()
