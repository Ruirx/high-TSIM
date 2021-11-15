#!/usr/bin/env python
# coding: utf-8

# In[32]:


import tensorly as tl
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import copy
from tensorly import decomposition
import warnings
warnings.filterwarnings('ignore')


# # Basic function for TSIM

# In[33]:


def gen_samp(reg_X, coe_B, samplesize):
    inner_product = np.zeros(samplesize)
    for i in range(samplesize):
        inner_product[i] = tl.tenalg.inner(reg_X[i], coe_B)
        
    return inner_product

def sample_vectorization(tensor, mod, samplesize):
    for i in range(samplesize):
        if i == 0:
            x_mat = np.mat(tensors_to_vec(tensor[0], mod=mod))
        else:
            temp_vec = np.mat(tensors_to_vec(tensor[i], mod=mod))
            x_mat = np.vstack((x_mat, temp_vec))
    return x_mat

def tensors_to_vec(tensor, mod):
    if mod==0:
        vec = tl.tensor_to_vec(tensor)
    else:
        tmat = tl.unfold(tensor, mode=mod)
        vec = tmat.flatten()
    return vec

def tensor_normalization(core_tensor, factors, len_core_rank):
    tensor = tl.tucker_to_tensor((core_tensor, factors))
    
    #signs = np.sign(tensor)  
    #for i in np.zeros(len_core_rank, dtype=int):
    #    signs = signs[i]
        
    tensor_norm = np.linalg.norm(tensor)

    normalized_tensor = tensor/tensor_norm
    
    return (normalized_tensor)

def covariate_syn(xtilde, beta):
    for i in range(len(beta)):
        xtilde[i,:] = float(beta[i]) * xtilde[i,:]
    
    return xtilde

def factor_kroneker(core_tensor, factors, exclude):
    maxs = len(factors)
    if exclude < 0 or exclude >= maxs:
        factorskro = tl.tenalg.kronecker(factors)
        
    else:
        factors_rev = []
        for i in range(maxs):
            if i != exclude:
                factors_rev.append(factors[i])
        
        b1kro = np.array(np.mat(tl.tenalg.kronecker(factors_rev))*np.mat(tl.unfold(core_tensor, mode=exclude)).T)
        
        m,_ = factors[exclude].shape
        
        xhat = list([np.eye(m)])
        xhat.extend([b1kro])

        factorskro = tl.tenalg.kronecker(xhat)
    
    return factorskro


# # Basic function for local weighted linear regression

# In[34]:


def single_LWregression(Bhat, xmat, yvec, hopt = 0.3):
    
    coef_vec = tl.tensor_to_vec(Bhat)
    
    design = np.mat(xmat)*np.mat(coef_vec.reshape(-1,1))
    design = np.hstack(((np.mat(np.ones(len(design)))).T, design)) 

    _, Beta = LocalWeightRegression(design, design, np.mat(yvec), h = hopt)
    
    return (Beta, design[:,1])

      
def LocalWeightRegression(xvec, xmat, ymat, h):
    m,n = np.shape(xvec)
    ypred = np.zeros(m)
    for i in range(m):
        if i == 0:
            coef = localWeight(xvec[i], xmat, ymat, h)
            coef_mat = copy.copy(coef)
        else:
            coef = localWeight(xvec[i], xmat, ymat, h)
            coef_mat = np.hstack((coef_mat, coef))
        
        ypred[i] = xvec[i]*coef
    return (ypred, coef_mat)

def localWeight(xvec, xmat, ymat, h):

    wt = kernel(xvec, xmat, h)
    W = np.linalg.pinv(xmat.T * wt * xmat) * (xmat.T * wt * ymat.T)
    return W

def kernel(xvec, xmat, h):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    
    for j in range(m):
        diff = xvec - xmat[j]
        weights[j, j] = np.exp(diff * diff.T/(-2.0 * h**2))
    return weights


# # Optimization functions

# In[35]:


def coef_approx_TSIM(core_tensor, factors, y, x_k,
                  core_rank, num_fac, Beta, design):

    ytilde = y - np.ravel(Beta[0,:]) + np.multiply(np.ravel(Beta[1,:]), np.ravel(design))
    
    for k in range(num_fac):
        b_k = factor_kroneker(core_tensor, factors, exclude=k)
        
        xtildek = x_k[k] * np.mat(b_k)
        xtildek = covariate_syn(xtilde=xtildek, beta=np.ravel(Beta[1,:]))
        
        lrk = (xtildek.T * xtildek).I * xtildek.T * ytilde.reshape(-1,1)

        uk = lrk.reshape(factors[k].shape)
        
        factors[k] = np.array(uk)
    
    b_u = factor_kroneker(core_tensor = core_tensor, factors = factors, exclude = -1)
    
    xtildeu = x_k[0]*np.mat(b_u)
    xtildeu = covariate_syn(xtilde=xtildeu, beta=np.ravel(Beta[1,:]))
    
    lru = np.linalg.pinv(xtildeu.T * xtildeu) * xtildeu.T * ytilde.reshape(-1,1)

    core_tensor = tl.vec_to_tensor(np.array(lru), shape = core_rank)
    
    
    Bhat1 = tensor_normalization(core_tensor = core_tensor, 
                                              factors=factors, 
                                              len_core_rank=num_fac)    
    return (Bhat1, core_tensor, factors)


def coef_opt_TSIM(X, y, core_rank, maxitr=100, hopt=-1, tol = 1e-4):
    XX = copy.copy(X)
    x = tl.unfold(X, mode=0)
    if hopt == -1:
        hopt = np.power(x.shape[0], -7/24)
        print('Default h in TSIM:', hopt)
        
    tenshape = X.shape[1:]
    nsample = X.shape[0]
    num_fac = len(core_rank)
    x_k = []
    for k in range(num_fac):
        x_k.append(sample_vectorization(X, mod=k, samplesize = nsample))
        
    xkk = copy.copy(x_k)
    for itr in range(maxitr):
        if (np.sum(xkk[1] - x_k[1]))!= 0:
            print('WRONG: xkk changed')
        if (np.sum(XX - X)) != 0:
            print('WRONG: X changed')
        if itr == 0:
            model = LinearRegression(fit_intercept=True)
            lr = model.fit(x,y)
            Bhat = tl.vec_to_tensor(lr.coef_, shape=tenshape)
            Bhat = Bhat/np.linalg.norm(Bhat)

            Beta, design = single_LWregression(Bhat, x, y, hopt=hopt)
            core_tensor, factors = tl.decomposition.tucker(Bhat, rank=core_rank)
            Bold = copy.copy(Bhat)

        else:
            Beta, design = single_LWregression(Bhat_opt, x, y, hopt=hopt)
            Bold = copy.copy(Bhat_opt)       
            
        Bhat_opt, core_tensor, factors = coef_approx_TSIM(core_tensor, factors,
                                                     y, x_k, core_rank, num_fac, 
                                                                        Beta, design)
        
        norm = np.linalg.norm(Bhat_opt - Bold)

        if norm < tol:
            print('Tolerance achieved')
            Bhat_opt = Bhat_opt/np.linalg.norm(Bhat_opt)
            Beta,_ = single_LWregression(Bhat_opt, x, y, hopt=hopt)
            break
        if itr == maxitr-1:
            print('Caution!, maxitr arrived')
            Bhat_opt = Bhat_opt/np.linalg.norm(Bhat_opt)
            Beta,_ = single_LWregression(Bhat_opt, x, y, hopt=hopt)
            
    return (Bhat_opt, Beta)


# # Plot function

# In[36]:


def plot_fun(xtensor, y, u_est, Beta, fres, title=False, save=False):
    xmat = tl.unfold(xtensor, mode=0)
    xarr= np.ravel(np.mat(xmat) * np.mat(tl.tensor_to_vec(u_est).reshape(-1,1)))

    yarr = np.ravel(Beta[0,:]) + np.multiply(np.ravel(Beta[1,:]), xarr)
    
    plt.rcParams["figure.figsize"] = [20.0, 10.0]
    plt.rcParams["figure.autolayout"] = True
    plt.figure()
    plt.subplots_adjust(hspace=0.01, wspace=0.1)
    plt.subplot(121)
    plt.scatter(fres, y, color='gray', label='observation')
    plt.plot(sorted((xarr)), yarr[(xarr).argsort(0)], color='red', label='estimates')
    plt.plot(sorted(fres), np.sin(sorted(fres)), color='black', label='underlying')
    if title:
        plt.title('link function estimation')
        plt.legend()
    
    plt.subplot(122)
    plt.plot(y,y,color='red', label='underlying')
    plt.scatter(yarr, y, color='gray', label='estimates v.s. observations')
    if title:
        plt.title('fitting results')
        plt.legend()   
    
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    
    if save:
        plt.savefig('lowtoy.eps')
        plt.savefig('lowtoy')

    plt.show()
    return yarr


# # Generate pseudo-samples

# In[37]:


def gen_pseudo_sample(nsample = 400, core_size = 2, dims = 4, sigma = 0.1, seed=0):

    core_rank = (core_size*np.ones(3, dtype=int)).tolist()
    lemx = np.array(range(core_size**3))
    emx = tl.vec_to_tensor(lemx, shape = core_rank)
    u1 = np.array(range(dims*core_size), dtype=float).reshape(dims,core_size); 
    u2 = np.array(range(dims*core_size), dtype=float).reshape(dims,core_size); 
    u3 = np.array(range(dims*core_size), dtype=float).reshape(dims,core_size)
    B = tl.tucker_to_tensor((emx, [u1,u2,u3]))
    B = B/np.linalg.norm(B)


    tensor_rank = list(B.shape)
    sample_size = [nsample]
    sample_size.extend(tensor_rank)

    np.random.seed(seed)

    X = tl.tensor(np.random.uniform(-2, 2, size=sample_size))


    fres = gen_samp(reg_X=X, coe_B=B,samplesize=nsample)

    eps = np.random.normal(loc=0, scale = 1, size = nsample)
    #y = np.sin(fres) + sigma*eps
    y = np.sin(fres) + sigma*eps
    # Note that, once the original tendency of the observations are decreasing, 
    # the index will be the same as the initialization refers to the OLS estimate.
    
    return X, y, fres, B


# In[ ]:




