# In the name of ALLAH
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:23:24 2024

@author: amidzam1
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.func import vmap, grad, functional_call, jacfwd
from collections import OrderedDict

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

seed = 10
torch.manual_seed(seed)
# torch.manual_seed(667)




    
# Parameters
# Parameters of OU-rocess
N_dim_X     = 50
N_dim_A_row = 100

Theta = 0.1 * torch.rand(size=[N_dim_X,N_dim_X])
Mu = 0.02 * torch.rand(size=[N_dim_X,1])
Sigma = torch.tensor([4]) * torch.rand(size=[N_dim_X,1])
    
# Parameters
t_start   = torch.tensor(0.001).to(device)
t_end     = torch.tensor(1.).to(device)
T         = t_end - t_start  # time horizon
N         = 40              # number of time steps 80
M         = 100              # 100number of sample paths 2000
dt        = T / N            # time step size
n_episode = 1

 
A = 1 * torch.randn(size=[N_dim_A_row , N_dim_X])
Theta_prim = A @ Theta @ torch.linalg.inv(A.T@A) @ A.T
Mu_prim    = A @ Mu
Sigma_prim = A @ Sigma

Func = []
# Model
Norm_grad= []
# Optimizer
# with torch.autograd.set_detect_anomaly(True):
    
LR = 6e-3

for i in range(n_episode):
    # initialiation
    
    X     =  torch.ones(size = (M, N_dim_X, N+1), requires_grad=False).to(device)
    Y     =  0 * torch.ones(size = (M, N_dim_X, N+1), requires_grad=False).to(device)
    Z     =  0 * torch.ones(size = (M, N_dim_A_row, N+1), requires_grad=False).to(device)
    DX    = torch.zeros(size = (M, N), requires_grad=True).to(device)
    W     = torch.zeros(size=(M, N+1), requires_grad=False).to(device)
    
    beta_sum, alpha_sum = 0, 0
    state_list = []
    func_t = []
    func_opt_t = []
    
    for n in range(N):
        time_value = t_start + n * dt
        time       = time_value * (torch.ones(size=[M], requires_grad=True).to(device))
        
        #X = X.detach().clone().requires_grad_(True)
        State = torch.cat( (X[:,:,n], time[:,None]), dim=1)
        state_list.append(State)
        
        dW_t    = torch.randn(size=[M,1]).to(device) * torch.sqrt(dt)
        
        ''' Without cloning: '''
        Wnew = torch.zeros_like(W)
        Wnew[:, 0:n+1] = W[:, 0:n+1]  # Copy existing values up to t
        Wnew[:, n+1]   = Wnew[:, n] + dW_t[:,0]      # Update specific step
        W = Wnew
        dY =  ( - Theta[None,...] @ Y[:, :, n][...,None] + (Theta @ Mu)[None,...]) * dt + Sigma[None,...]  * dW_t[:,None,:] 
        Y[:, :, n+1] = Y[:, :, n] + dY.squeeze(-1)      # Update specific step
        dZ =  ( - Theta_prim[None,...] @ Z[:, :, n][...,None] + (Theta_prim @ Mu_prim)[None,...]) * dt + Sigma_prim[None,...]  * dW_t[:,None,:] 
        Z[:, :, n+1] = Z[:, :, n] + dZ.squeeze(-1)      # Update specific step
        
        ''' Stochastic Gradient Descent: '''
        noise = 1*torch.randn(size=[M, N_dim_A_row])
        X[:, :, n+1] = X[:, :, n] - LR * ((A.T) @ (A @ X[:,:,n].T - Z[:,:,n].T - noise.T)).T
    
        
        func      = ((A @ X[:,:,n].T - Z[:,:,n].T - noise.T).norm(dim=0)**2).mean()/2
        func_t.append((A @ X[:,:,n].T - Z[:,:,n].T - noise.T).norm(dim=0)**2/2)
    
        func_opt  = ((A @ Y[:,:,n].T - Z[:,:,n].T - noise.T).norm(dim=0)**2).mean()/2
        func_opt_t.append((A @ Y[:,:,n].T - Z[:,:,n].T - noise.T).norm(dim=0)**2/2)
        
    Func.append(func.mean().detach().cpu())

    
    

        
    if i%5 == 0:
        print("Iteration step = ", i, "func = ",func.mean())
        sns.set()
        plt.figure(1)
        plt.plot(Func), plt.title('func')
        # plt.ylim([0., 2])
    #     plt.savefig(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.png")
    #     np.save(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.npy", Func)
    #     # np.save(f"results/Fnc2_v4_alpha_N{N}_M{M}_2.npy", alpha_sum.detach())
    #     # np.save(f"results/Fnc2_v4_Langevin_N{N}_M{M}_2.npy", beta_sum.detach())
        plt.show()
        
        
    if i%5 == 0:
        plt.figure(2)
        plt.plot(X[0:M:,0, :].cpu().T)
        plt.figure(3)
        plt.plot(X[0:M:,1, :].cpu().T)
        plt.figure(5)
        Delta_f = (torch.stack(func_t) - torch.stack(func_opt_t)).abs().mean(dim=1).detach().cpu()
        plt.plot( Delta_f ), plt.title('E|f(X)-f(X*)|')
        plt.figure(4)
        Delta_x = ((X-Y).norm(dim=1)**2).mean(dim=0)
        plt.plot( Delta_x ), plt.title('E|X-X*|^2')
        plt.show()

        np.save(f"results/LS_SGD__diff_X_N{N}_M{M}_{seed}_long.npy", Delta_x[:N] )
        np.save(f"results/LS_SGD__diff_F_N{N}_M{M}_{seed}_long.npy", Delta_f[:N] )