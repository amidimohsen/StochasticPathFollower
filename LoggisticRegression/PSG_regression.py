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
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cpu")

torch.manual_seed(10)



# Parameters of OU-rocess
N_dim_X     = 5 
N_dim_A_row = 10
    
Theta = 0.1 * torch.rand(size=[N_dim_A_row,N_dim_A_row])
Mu    = 0.02 * torch.rand(size=[N_dim_A_row,1])
Sigma = 4 * torch.rand(size=[N_dim_A_row,1])
    
# Parameters
t_start   = torch.tensor(0.001).to(device)
t_end     = torch.tensor(1.).to(device)
T         = t_end - t_start  # time horizon
N         = 40              # number of time steps 80
M         = 400              # 100number of sample paths 2000
dt        = T / N            # time step size
n_episode = 5

 
A = torch.randn(size=[N_dim_A_row , N_dim_X])
S0 = 0 * torch.ones(size = (N_dim_A_row, 1) )
b0 = torch.sigmoid(S0)
mu = 1e-4
LR = 4e-1

# x_ = torch.zeros(size=[N_dim_X], requires_grad= True)
# Optimizer = torch.optim.Adam(params=[x_], lr=1e-1)
# for e in range(2000):
#     loss =  ( torch.log(1 + torch.exp(A@x_)).sum() - b0.T @ A @ x_ ) + mu/2 * x_.norm()**2
#     # loss = objective(x)
#     Optimizer.zero_grad()
#     loss.backward()
#     Optimizer.step()
    
    
gen = torch.Generator()
Func = []
# Model
Norm_grad= []
# Optimizer
# with torch.autograd.set_detect_anomaly(True):
for i in range(n_episode):
    # initialiation
    
    # X   =   x_[None,:,None].repeat([M,1,N]).requires_grad_(False)#$ * torch.ones(size = (M, N_dim_X, N), requires_grad=True).to(device)
    X   =   1*torch.ones(size = (M, N_dim_X, N), requires_grad=False).to(device)
    S   =   0 * torch.ones(size = (M, N_dim_A_row, N), requires_grad=False).to(device)
    W   = torch.zeros(size=(M, N), requires_grad=False).to(device)
    
    beta_sum, alpha_sum = 0, 0
    state_list = []
    alpha_list = []
    beta_list  = []
    dW_list    = []
    func_t     = []
    
    
    for n in range(N-1):
        time_value = t_start + n * dt
        time       = time_value * (torch.ones(size=[M], requires_grad=False).to(device))
        
        #X = X.detach().clone().requires_grad_(True)
        State = torch.cat( (X[:,:,n], time[:,None]), dim=1)
        state_list.append(State)
        
            
        gen.manual_seed(n)
        dW_t    = torch.randn(size=[M,1], generator=gen).to(device) * torch.sqrt(dt)
        dW_list.append(dW_t)
        
        ''' Without cloning: '''
        Wnew = torch.zeros_like(W)
        Wnew[:, 0:n+1] = W[:, 0:n+1]  # Copy existing values up to t
        Wnew[:, n+1]   = Wnew[:, n] + dW_t[:,0]      # Update specific step
        W = Wnew
        dS =  ( - Theta[None,...] @ S[:, :, n][...,None] + (Theta @ Mu)[None,...] ) * dt + Sigma[None,...]  * dW_t[:,None,:] 
        S[:, :, n+1] = S[:, :, n] + dS.squeeze(-1)      # Update specific step

        # bt = 1/(1+torch.exp(-S[:,:,n]))
        bt = torch.sigmoid(S[:, :, n])
        ct = torch.exp(-S[:,:,n]) / (1+torch.exp(-S[:,:,n]))**2
        
        # func      =  ( torch.log( 1 + torch.exp( A@X[:,:,n].T) ).sum(dim=0) - ((bt@A)*X[:,:,n]).sum(dim=1) )\
        #             + mu * X[:,:,n].norm(dim=1)**2/2
        # func_X    =  ( torch.einsum("Mn, nd -> Md", (torch.exp( A@X[:,:,n].T)/(1+torch.exp( A@X[:,:,n].T))).T, A) -(bt @ A) ) \
        #             + mu * X[:,:,n]
        func      =  ( -torch.nn.functional.logsigmoid( -A@X[:,:,n].T ).sum(dim=0) - ((bt@A)*X[:,:,n]).sum(dim=1) )\
                    + mu * X[:,:,n].norm(dim=1)**2/2
        func_X   =  ( torch.einsum("Mn, nd -> Md", torch.sigmoid( A@X[:,:,n].T).T, A) -(bt @ A) ) \
                    + mu * X[:,:,n]
                    
                    
        ''' Without cloning: '''
        dX =  - LR  * func_X
        Xnew = torch.zeros_like(X)
        Xnew[:, :, 0:n+1] = X[:, :, 0:n+1]  # Copy existing values up to t
        Xnew[:, :, n+1] = Xnew[:, :, n] + dX      # Update specific step
        X = Xnew#.detach()
        
    
    ''' obtaining the optimal solution and optimal objective '''
    if i==0:
        try:
            X_optim    = torch.load("X_optim3.npy")
            func_opt_t = torch.load("func_opt_t3.npy")
        except:
            X_optim    = []
            func_opt_t = []
            for n in range(N):
                print(n)
                # bt = 1/(1+torch.exp(-S[:,:,n]))
                bt = torch.sigmoid(S[:, :, n])
                x_opt = torch.ones(size = (M, N_dim_X), requires_grad=True)
                Optimizer_ = torch.optim.Adam(params=[x_opt], lr=8e-2)
                for e in range(2000):
                    loss0 =  ( torch.log(1 + torch.exp(x_opt@A.T)).sum(dim=1) - (bt@A*x_opt).sum(dim=1) ) + mu/2 * x_opt.norm(dim=1)**2
                    loss_ = loss0.mean()
                    # loss = objective(x)
                    Optimizer_.zero_grad()
                    loss_.backward()
                    Optimizer_.step()
                X_optim.append(x_opt)
                
                func_opt   =  ( torch.log( 1 + torch.exp( A@x_opt.T) ).sum(dim=0) - ((bt@A)*x_opt).sum(dim=1) )\
                            + mu * x_opt.norm(dim=1)**2/2
                func_opt_t.append(func_opt)
                            
            X_optim    = torch.stack(X_optim, dim=2)
            func_opt_t = torch.stack(func_opt_t, dim=1)
            torch.save(X_optim,"X_optim3.npy")        
            torch.save(func_opt_t,"func_opt_t3.npy")         
        
        
    for n in range(N):
        # bt = 1/(1+torch.exp(-S[:,:,n]))
        bt = torch.sigmoid(S[:, :, n])
        # func      =  ( torch.log( 1 + torch.exp( A@X[:,:,n].T) ).sum(dim=0) - ((bt@A)*X[:,:,n]).sum(dim=1) ) \
        #             + mu * X[:,:,n].norm(dim=1)**2/2
        func      =  ( -torch.nn.functional.logsigmoid( -A@X[:,:,n].T ).sum(dim=0) - ((bt@A)*X[:,:,n]).sum(dim=1) )\
                    + mu * X[:,:,n].norm(dim=1)**2/2
        func_t.append(func)

    Func.append(func.mean().detach().cpu())
        
    
    if i%5 == 0:
        print("Iteration step = ", i, "func = ",func.mean().detach())
        sns.set()
        plt.figure(1)
        # plt.plot(Func)
        # plt.ylim([3., 9])
    #     plt.savefig(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.png")
    #     np.save(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.npy", Func)
    #     # np.save(f"results/Fnc2_v4_alpha_N{N}_M{M}_2.npy", alpha_sum.detach())
    #     # np.save(f"results/Fnc2_v4_Langevin_N{N}_M{M}_2.npy", beta_sum.detach())
        # plt.show()
        
        
    if i%5 == 0:
        plt.figure(2)
        plt.plot(X[0:M:,0, :].cpu().detach().T)
        plt.figure(3)
        plt.plot(X[0:M:,-1, :].cpu().detach().T)
        plt.figure(5)
        Delta_f = (torch.stack(func_t,dim=1) - func_opt_t).abs().mean(dim=0).detach().cpu()
        plt.plot( Delta_f ), plt.title('E|f(X)-f(X*)|')
        plt.show()
        plt.figure(4)
        Delta_x = ((X_optim - X).norm(dim=1)**2).mean(dim=0).detach()
        plt.plot( Delta_x  ), plt.title('E|X-X*|^2')
        # plt.ylim([0,1])
        plt.show()

        np.save(f"results/LogReg_SGD__diff_F_N{N}_M{M}.npy", Delta_f)
        np.save(f"results/LogReg_SGD__diff_X_N{N}_M{M}.npy", Delta_x)
