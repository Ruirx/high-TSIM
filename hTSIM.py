#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
from tensorly import decomposition
from sklearn.linear_model import LinearRegression, LassoCV
import copy
import warnings
warnings.filterwarnings('ignore')


# # Basic functions for TSIM

# In[9]:


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

def tensor_normalization(core_tensor, factors):
    tensor = tl.tucker_to_tensor((core_tensor, factors))
    tensor_norm = np.linalg.norm(tensor)

    normalized_tensor = tensor/tensor_norm
    
    return (normalized_tensor)

def covariate_syn(xtilde, beta):
    for i in range(len(beta)):
        xtilde[i,:] = float(beta[i]) * xtilde[i,:]
    
    return xtilde

def factor_kroneker(core_tensor, factors, exclude):
    core_tensor[np.isnan(core_tensor)] = 0.1
    maxs = len(factors)
    if exclude < 0 or exclude >= maxs:
        factorskro = tl.tenalg.kronecker(factors)
        
    else:
        factors_rev = []
        for i in range(maxs):
            if np.sum(np.isnan(factors[i])):
                print('sum of nan in factors: ', np.sum(np.isnan(factors[i])))
                factors[i][np.isnan(factors[i])] = 0.01
            
            if i != exclude:
                factors_rev.append(factors[i])
        
        b1kro = np.array(np.mat(tl.tenalg.kronecker(factors_rev))*np.mat(tl.unfold(core_tensor, mode=exclude)).T)
        
        m,_ = factors[exclude].shape
        
        xhat = list([np.eye(m)])
        xhat.extend([b1kro])

        factorskro = tl.tenalg.kronecker(xhat)
    
    return factorskro

def tensor_det(core_tensor, factors):
    num_fac = len(factors)
    core_tensor[np.isnan(core_tensor)] = 0.1
    for i in range(num_fac):
        factors[i][np.isnan(factors[i])] = 0.1
        _,col = factors[i].shape
        mat = np.mat(factors[i][:col, ])
        factemp = np.mat(factors[i])*np.mat(np.linalg.pinv(mat))
        factemp = np.vstack((np.eye(col), np.array(factemp[col:, ])))
        factors[i] = factemp
        core_tensor = tl.tenalg.mode_dot(core_tensor, np.array(mat), mode=i)
    Bnorm  = tensor_normalization(core_tensor, factors)
    return (core_tensor, factors, Bnorm)


# # Basic functions for local weighted linear regression

# In[10]:


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

# In[11]:


def coef_approx_TSIM(core_tensor, factors, y, x_k,
                  core_rank, num_fac, Beta, design, kfold):

    design[np.isnan(design)] = 0.01
    Beta[np.isnan(Beta)] = 0.01
    ytilde = y - np.ravel(Beta[0,:]) + np.multiply(np.ravel(Beta[1,:]), np.ravel(design))
    
    for k in range(num_fac):
        row,col = factors[k].shape
        b_k = factor_kroneker(core_tensor, factors, exclude=k)
        xtildek = x_k[k] * np.mat(b_k)
        xtildek = covariate_syn(xtilde=xtildek, beta=np.ravel(Beta[1,:]))

        lrk = LassoCV(cv=kfold).fit(xtildek, ytilde)
        uk = lrk.coef_.reshape(-1, col)
        if np.sum(np.isnan(uk))>0:
            print('Caution, NaN generated')
            print(uk)
            uk[np.isnan(uk)] = 0.01
            print(uk)
            break

        factors[k] = np.array(uk)

    b_u = factor_kroneker(core_tensor = core_tensor, factors = factors, exclude = -1)

    xtildeu = x_k[0]*np.mat(b_u)
    xtildeu = covariate_syn(xtilde=xtildeu, beta=np.ravel(Beta[1,:]))

    model = LinearRegression(fit_intercept=False)
    lru = model.fit(xtildeu, ytilde)
    coef = lru.coef_

    core_tensor = tl.vec_to_tensor(np.array(coef), shape = core_rank)

    core_tensor, factors, Bhat1 = tensor_det(core_tensor, factors)

    return (Bhat1, core_tensor, factors)

def coef_opt_TSIM(X, y, core_rank, maxitr=100, hopt=-1, tol = 1e-4, kfold=5):
    XX = copy.copy(X)
    x = tl.unfold(X, mode=0)
    tenshape = X.shape[1:]
    nsample = X.shape[0]
    
    if hopt == -1:
        hopt = np.power(nsample, -7/24)
        print('Default h in TSIM:', hopt)
    x_k = []
    num_fac = len(core_rank)
    for k in range(num_fac):
        x_k.append(sample_vectorization(X, mod=k, samplesize = nsample))
        
    for itr in range(maxitr):
        if itr == 0:
            model = LinearRegression(fit_intercept=True)
            lr = model.fit(x,y)
            Bhat = tl.vec_to_tensor(lr.coef_, shape=tenshape)
            Bhat = Bhat/np.linalg.norm(Bhat)*(np.sign(lr.coef_)[0])
             
            Beta, design = single_LWregression(Bhat, x, y, hopt=hopt)
            core_tensor, factors = tl.decomposition.tucker(Bhat, rank=core_rank)
            core_tensor, factors,_ = tensor_det(core_tensor, factors)
            Bold = copy.copy(Bhat)
            
            #yy = np.ravel(Beta[0,:]) + np.multiply(np.ravel(Beta[1,:]), np.ravel(design))
            #plt.plot(sorted(np.ravel(design)), yy[(design).argsort(0)], color='red')            
            #plt.scatter(sorted(fres), y[fres.argsort(0)])
            #plt.plot(sorted(fres),np.cos(sorted(fres)))
            #plt.show() 

        else:
            if np.isnan(Bhat_opt).all():
                print(f'All elements in B is nan in step {itr}, stop optimization.')
                Bhat_opt = Bold
                break
            else:
                Beta, design = single_LWregression(Bhat_opt, x, y, hopt=hopt)
                Bold = copy.copy(Bhat_opt)

            #ys = np.ravel(Beta[0,:]) + np.multiply(np.ravel(Beta[1,:]), np.ravel(design))
            #plt.plot(sorted(np.ravel(design)), ys[(design).argsort(0)], color='red')
            #plt.scatter(sorted(fres), y[fres.argsort(0)])
            #plt.plot(sorted(fres),np.cos(sorted(fres)))
            #plt.plot(sorted(np.ravel(fres)), yy[(design).argsort(0)], color='green')            
            #plt.show()        
        

            
        Bhat_opt, core_tensor, factors = coef_approx_TSIM(core_tensor, factors,
                                                     y, x_k, core_rank, num_fac, 
                                                                        Beta, design, kfold)
        
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
    
    sign_opt = np.sign(tl.tensor_to_vec(Bhat_opt)[0])
    Bhat_opt = Bhat_opt*sign_opt
    Beta = Beta*sign_opt
    return (Bhat_opt, Beta)


# # Plot function for fitting results

# In[37]:


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
        plt.savefig('hightoy.eps')
        plt.savefig('hightoy')
        
    plt.show()
    return yarr


# # Pseudo-random sample generation function

# In[26]:


def gen_pseudo_sample(nsample = 400, core_size = 2, psize = 2, nzero = 2, sigma = 0.1, seed=101):

    core_rank = (np.hstack((core_size*np.ones(2, dtype=int), 3))).tolist()
    lemx = np.array(range(1, core_size**2*3+1))
    emx = tl.vec_to_tensor(lemx, shape = core_rank)
    u1 = np.vstack((np.eye(core_size), np.random.normal(0, 1, size=psize*core_size).reshape(psize,core_size), np.zeros((nzero, core_size)))); 
    u2 = np.vstack((np.eye(core_size), np.random.normal(0, 1, size=psize*core_size).reshape(psize,core_size), np.zeros((nzero, core_size)))); 
    u3 = np.eye(3); 
    B = tl.tucker_to_tensor((emx, [u1,u2,u3]))
    B = B/np.linalg.norm(B)


    tensor_dim = list(B.shape)

    sample_size = [nsample]
    sample_size.extend(tensor_dim)

    np.random.seed(seed)

    X = tl.tensor(np.random.uniform(-1, 1, size=sample_size))


    fres = gen_samp(reg_X=X, coe_B=B,samplesize=nsample)

    eps = np.random.normal(loc=0, scale = 1, size = nsample)
    #y = np.sin(fres) + sigma*eps
    y = np.sin(fres) + sigma*eps
    # Note that, once the original tendency of the observations are decreasing, 
    # the index will be the same as the initialization refers to the OLS estimate.

    return X, y, fres, B


# In[ ]:





# In[ ]:




