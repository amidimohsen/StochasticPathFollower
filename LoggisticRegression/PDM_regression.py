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

torch.manual_seed(0)


def cumprod_mat(A, dim=-2, right_multiply=False):
    """
    Cumulative matrix product along a time axis for a batch of matrices.
    Out[..., t, :, :] = A[..., 0, :, :] @ A[..., 1, :, :] @ ... @ A[..., t, :, :]
    (when right_multiply=False).

    Parameters
    ----------
    A : torch.Tensor
        Tensor of shape (..., T, n, n) (time axis T at position `dim`).
    dim : int
        Axis index of the time dimension in A (default -2: second-last, i.e. (..., T, n, n)).
    right_multiply : bool
        If True, compute Out[..., t] = A[..., t] @ A[..., t-1] @ ... @ A[..., 0]
        (i.e. cumulative product with A[t] on left each step).
        If False (default), compute forward product Out[..., t] = A[..., 0] @ A[..., 1] @ ... @ A[..., t].

    Returns
    -------
    Out : torch.Tensor
        Tensor of same shape as A containing cumulative matrix products.
    """
    # move time dim to index 0 for convenience
    A = A.movedim(dim, 0)   # now shape (T, ..., n, n)
    T = A.shape[0]
    rest_shape = A.shape[1:-2]  # batch dims (may be empty)
    n1, n2 = A.shape[-2], A.shape[-1]
    if n1 != n2:
        raise ValueError("cumprod_mat requires square matrices (n x n) at the trailing dims")

    out_list = [A[0]]  # Start with first element (creates new reference, not in-place)
    # scan forward
    if not right_multiply:
        for t in range(1, T):
            # out[t] = out[t-1] @ A[t]
            out_list.append(torch.matmul(out_list[t-1], A[t]))
    else:
        # out[t] = A[t] @ out[t-1]
        for t in range(1, T):
            out_list.append(torch.matmul(A[t], out_list[t-1]))

    out = torch.stack(out_list, dim=0)  # shape (T, ..., n, n)
    out = out.movedim(0, dim)
    return out


def LipSwish(x):
    return  0.909 * x * torch.nn.functional.sigmoid(x)

def Tanh(x):
    return  torch.nn.functional.tanh(x)

def Gelu(x):
    return  torch.nn.functional.gelu(x)

# Flatten per-sample Jacobians (OrderedDict of [M, out_dim, *param_shape]) -> [M, out_dim, DIm]
def flatten_per_sample_jacobian(jac_pytree: OrderedDict):
    # Keep [M, out_dim] intact; flatten only param dims and concat across params
    flat_chunks = []
    for g in jac_pytree.values():
        # g: [M, out_dim, *param_shape]
        flat_chunks.append(g.flatten(start_dim=2))  # -> [M, out_dim, numel(param)]
    return torch.cat(flat_chunks, dim=2)  # -> [M, out_dim, DIm]

class Dynamics_NN(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.hidden = 8
        self.Lay1_drift   = torch.nn.Linear(in_features = N_dim_X + 1, out_features = self.hidden)
        self.Lay2_drift   = torch.nn.Linear(in_features = self.hidden, out_features = self.hidden)
        self.Lay3_drift   = torch.nn.Linear(in_features = self.hidden, out_features = self.hidden)
        self.Lay_alpha   = torch.nn.Linear(in_features=self.hidden, out_features = N_dim_X)

        self.Lay1_diffusion   = torch.nn.Linear(in_features = N_dim_X + 1, out_features = self.hidden)
        self.Lay2_diffusion   = torch.nn.Linear(in_features = self.hidden, out_features = self.hidden)
        self.Lay3_diffusion   = torch.nn.Linear(in_features = self.hidden, out_features = self.hidden)
        self.Lay_beta    = torch.nn.Linear(in_features=self.hidden, out_features = N_dim_X)
        
        self.Lay1_norm = torch.nn.LayerNorm(normalized_shape = self.hidden)
        self.Lay2_norm = torch.nn.LayerNorm(normalized_shape = self.hidden)
        self.Lay3_norm = torch.nn.LayerNorm(normalized_shape = self.hidden)
        self.Lay4_norm = torch.nn.LayerNorm(normalized_shape = self.hidden)
            
    
    def forward(self, State):
        hidden1_drift = Gelu( self.Lay1_drift(State) ) 
        hidden2_drift = Gelu( self.Lay2_drift(hidden1_drift) ) 
        hidden3_drift = Gelu( self.Lay3_drift(hidden2_drift) ) 
        Alpha         = self.Lay_alpha(hidden3_drift)
        
        hidden1_diffusion = Gelu( self.Lay1_diffusion(State) )
        hidden2_diffusion = Gelu( self.Lay2_diffusion(hidden1_diffusion) ) 
        hidden3_diffusion = Gelu( self.Lay3_diffusion(hidden2_diffusion) )  
        Beta              = self.Lay_beta(hidden3_diffusion)
        
        return Alpha, Beta
                
  
    
  # Parameters of OU-rocess
N_dim_X     = 5 
N_dim_A_row = 10
    
Theta = 0.1 * torch.rand(size=[N_dim_A_row,N_dim_A_row])
Mu = 0.02 * torch.rand(size=[N_dim_A_row,1])
Sigma = torch.tensor([4]) * torch.rand(size=[N_dim_A_row,1])
    
# Parameters
t_start   = torch.tensor(0.001).to(device)
t_end     = torch.tensor(1.).to(device)
T         = t_end - t_start  # time horizon
N         = 40              # number of time steps 80
M         = 400              # 100number of sample paths 2000
dt        = T / N            # time step size
n_episode = 20000

 
A = torch.randn(size=[N_dim_A_row , N_dim_X])
S0 = 0 * torch.ones(size = (N_dim_A_row, 1) )
b0 = 1/(1+torch.exp(-S0))
mu = 1e-4

x_ = torch.zeros(size=[N_dim_X], requires_grad= True)
Optimizer = torch.optim.Adam(params=[x_], lr=1e-1)
for e in range(2000):
    # loss =  (torch.log(1 + torch.exp(A@x_)).sum() - b0.T @ A @ x_ ) + mu/2 * x_.norm()**2
    loss = ( -torch.nn.functional.logsigmoid( -A@x_).sum() - b0.T @ A @ x_ ) + mu/2 * x_.norm()**2
    # loss = objective(x)
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()
    
    
gen = torch.Generator()
Func = []
# Model
Dynamics = Dynamics_NN(dim=1).to(device)
Norm_grad= []
# Optimizer
LR = 8e-3
Optimizer = torch.optim.Adam(params = Dynamics.parameters(), lr = LR)
# with torch.autograd.set_detect_anomaly(True):
for i in range(n_episode):
    # initialiation
    params = OrderedDict(Dynamics.named_parameters())
    buffers = OrderedDict(Dynamics.named_buffers())
    DIm = sum(p.numel() for p in params.values())

    X   =   x_[None,:,None].repeat([M,1,N]).requires_grad_(True)#$ * torch.ones(size = (M, N_dim_X, N), requires_grad=True).to(device)
    VarX = torch.zeros(size = (M, N_dim_X, N, DIm), requires_grad=True ).to(device)
    S    =   0 * torch.ones(size = (M, N_dim_A_row, N), requires_grad=False).to(device)
    DX   = torch.zeros(size = (M, N), requires_grad=True).to(device)
    W    = torch.zeros(size=(M, N), requires_grad=False).to(device)
    
    
    beta_sum, alpha_sum = 0, 0
    state_list = []
    alpha_list = []
    beta_list  = []
    dW_list  = []
    func_t   = []
    for n in range(N-1):
        time_value = t_start + n * dt
        time       = time_value * (torch.ones(size=[M], requires_grad=True).to(device))
        
        #X = X.detach().clone().requires_grad_(True)
        State = torch.cat( (X[:,:,n], time[:,None]), dim=1)
        state_list.append(State)
        
        alpha_Xt, beta_Xt = Dynamics.forward( State )
        
        # /////
        # Functions that return the per-sample outputs (vector of size N_dim_X)
        def alpha_vec(p, b, x_single):
            a, _ = functional_call(Dynamics, (p, b), (x_single.unsqueeze(0),))
            return a.squeeze(0)  # [N_dim_X]
        
        def beta_vec(p, b, x_single):
            _, bt = functional_call(Dynamics, (p, b), (x_single.unsqueeze(0),))
            return bt.squeeze(0)  # [N_dim_X]
        
        # Per-sample Jacobians w.r.t. params, vectorized across the batch
        # Each entry in grads_*_tree: [M, N_dim_X, *param.shape]
        grads_alpha_tree = vmap(jacfwd(alpha_vec, argnums=0), in_dims=(None, None, 0))(params, buffers, State.detach())
        grads_beta_tree  = vmap(jacfwd(beta_vec,  argnums=0), in_dims=(None, None, 0))(params, buffers, State.detach())
        
        # Flatten only parameter dims -> [M, N_dim_X, DIm]
        diff_theta_alpha = flatten_per_sample_jacobian(grads_alpha_tree)
        diff_theta_beta  = flatten_per_sample_jacobian(grads_beta_tree)
        
        # Optional sanity checks
        assert diff_theta_alpha.shape[0] == M and diff_theta_alpha.shape[1] == N_dim_X and diff_theta_alpha.shape[2] == DIm
        assert diff_theta_beta.shape == diff_theta_alpha.shape
        # ///// 
        diff_X_alpha = vmap(jacfwd(lambda x: Dynamics(x)[0]))(State)[:, :, :N_dim_X]  # (M, d, d)
        diff_X_beta  = vmap(jacfwd(lambda x: Dynamics(x)[1]))(State)[:, :, :N_dim_X]
        
        beta_list.append(beta_Xt)
        beta_sum  = beta_sum + beta_Xt.mean().abs()
        alpha_sum = alpha_sum + alpha_Xt.mean().abs()
            
        gen.manual_seed(n)
        dW_t    = torch.randn(size=[M,1], generator=gen).to(device) * torch.sqrt(dt)
        dW_list.append(dW_t)
        
        ''' Without cloning: '''
        Wnew = torch.zeros_like(W)
        Wnew[:, 0:n+1] = W[:, 0:n+1]  # Copy existing values up to t
        Wnew[:, n+1]   = Wnew[:, n] + dW_t[:,0]      # Update specific step
        W = Wnew
        dS =  ( - Theta[None,...] @ S[:, :, n][...,None] + (Theta @ Mu)[None,...]) * dt + Sigma[None,...]  * dW_t[:,None,:] 
        S[:, :, n+1] = S[:, :, n] + dS.squeeze(-1)      # Update specific step

        d_VarX         = (diff_theta_alpha + diff_X_alpha @ VarX[:, :, n, :])*dt +\
                         (diff_theta_beta + diff_X_beta @ VarX[:, :, n, :])*dW_t[...,None]
        # VarX           = VarX.clone()
        ''' Without cloning: '''
        VarXnew = torch.zeros_like(VarX)
        VarXnew[:, :, 0:n+1, :] = VarX[:, :, 0:n+1, :]  # Copy existing values up to t
        VarXnew[:, :, n+1, :]   = VarXnew[:, :, n, :] + d_VarX      # Update specific step
        VarX = VarXnew.detach() 
        
        dX =  (alpha_Xt * dt + beta_Xt * dW_t)
        ''' Without cloning: '''
        Xnew = torch.zeros_like(X)
        Xnew[:, :, 0:n+1] = X[:, :, 0:n+1]  # Copy existing values up to t
        Xnew = Xnew.clone() 
        Xnew[:, :, n+1] = Xnew[:, :, n] + dX      # Update specific step
        X = Xnew#.detach()
        
    
    ''' obtaining the optimal solution and optimal objective '''
    if i==0:
        try:
            X_optim    = torch.load("X_optim@.npy")
            func_opt_t = torch.load("func_opt_t@.npy")
        except:
            X_optim    = []
            func_opt_t = []
            for n in range(N):
                print(n)
                bt = 1/(1+torch.exp(-S[:,:,n]))
                x_opt = torch.ones(size = (M, N_dim_X), requires_grad=True)
                Optimizer_ = torch.optim.Adam(params=[x_opt], lr=8e-2)
                for e in range(2000):
                    # loss0 = ( torch.log(1 + torch.exp(x_opt@A.T)).sum(dim=1) - (bt@A*x_opt).sum(dim=1) ) + mu/2 * x_opt.norm(dim=1)**2
                    loss0 = ( -torch.nn.functional.logsigmoid( -x_opt@A.T ).sum(dim=1) - (bt@A*x_opt).sum(dim=1) ) + mu/2 * x_opt.norm(dim=1)**2
                    loss_ = loss0.mean()
                    # loss = objective(x)
                    Optimizer_.zero_grad()
                    loss_.backward()
                    Optimizer_.step()
                X_optim.append(x_opt)
                
                # func_opt   = ( torch.log( 1 + torch.exp( A@x_opt.T) ).sum(dim=0) - ((bt@A)*x_opt).sum(dim=1) )\
                #             + mu * x_opt.norm(dim=1)**2/2
                func_opt   =  ( -torch.nn.functional.logsigmoid( -x_opt@A.T ).sum(dim=1) - ((bt@A)*x_opt).sum(dim=1) )\
                            + mu * x_opt.norm(dim=1)**2/2
                func_opt_t.append(func_opt)
                            
            X_optim    = torch.stack(X_optim, dim=2)
            func_opt_t = torch.stack(func_opt_t, dim=1)
            torch.save(X_optim,"X_optim.npy")        
            torch.save(func_opt_t,"func_opt_t.npy")       
        
        
    ''' Computation for the last time step '''
    n = N-1
    time_value = t_start + n * dt
    time       = time_value * (torch.ones(size=[M], requires_grad=True).to(device))
    dW_t    = torch.randn(size=[M,1]).to(device) * torch.sqrt(dt)
    dW_list.append(dW_t)
    dW_all = torch.cat(dW_list, dim=1)
    State = torch.cat( (X[:,:,n], time[:,None]), dim=1)
    alpha_Xt, beta_Xt = Dynamics.forward( State )
    beta_list.append(beta_Xt)
    state_list.append(State)
    State_all = torch.cat(state_list, dim=0)
    alpha_all, beta_all = Dynamics.forward( State_all )

    Jacob_alpha_all = torch.zeros(size=[N*M, N_dim_X, N_dim_X])
    for l in range(N_dim_X):
        Jacob_alpha_all[:,l,:]  = torch.autograd.grad(outputs=alpha_all[:,l], inputs=State_all,\
                                             grad_outputs=torch.ones_like(alpha_all[:,l]),\
                                             retain_graph=True, create_graph=True)[0][:,:N_dim_X] 
    Jacob_beta_all = torch.zeros(size=[N*M, N_dim_X, N_dim_X])
    for l in range(N_dim_X):
        Jacob_beta_all[:,l,:]  = torch.autograd.grad(outputs=beta_all[:,l], inputs=State_all,\
                                             grad_outputs=torch.ones_like(beta_all[:,l]),\
                                             retain_graph=True, create_graph=True)[0][:,:N_dim_X] 
        
    Jacob_alpha_all  = Jacob_alpha_all.reshape([N, M, N_dim_X, N_dim_X]).permute([1, 0, 2, 3])                 # size = [N_path, N_t, N_dim_U, N_dim_X]
    Jacob_beta_all   = Jacob_beta_all.reshape([N, M, N_dim_X, N_dim_X]).permute([1, 0, 2, 3])                 # size = [N_path, N_t, N_dim_U, N_dim_X]
    

        
    Loss = 0
    for n in range(N):
        bt = 1/(1+torch.exp(-S[:,:,n]))
        ct = torch.exp(-S[:,:,n]) / (1+torch.exp(-S[:,:,n]))**2
        # func      =  ( torch.log( 1 + torch.exp( A@X[:,:,n].T) ).sum(dim=0) - (bt@A*X[:,:,n]).sum(dim=1) )\
        #             + mu * X[:,:,n].norm(dim=1)**2/2
        func   =  ( -torch.nn.functional.logsigmoid( -A@X[:,:,n].T ).sum(dim=0) - ((bt@A)*X[:,:,n]).sum(dim=1) )\
                + mu * X[:,:,n].norm(dim=1)**2/2
        func_t.append(func)
        # func_X    =  ( torch.einsum("nM, nd -> Md", torch.exp( A@X[:,:,n].T)/(1+torch.exp( A@X[:,:,n].T)), A) -(bt @ A) ) \
        #             + mu * X[:,:,n]
        func_X =  ( torch.einsum("nM, nd -> Md", torch.sigmoid( A@X[:,:,n].T), A) -(bt @ A) )\
            + mu * X[:,:,n]
        Loss = Loss + (func_X[...,None]*VarX[:,:,n,:]).sum(dim=1).mean(dim=0).norm()**2
            # Loss = Loss + ( A.T @ (A * DX_t[n][:, s] - A * DY_t[n][:, s] - 1) ).mean()**2 + (func_X).mean()**2
            
    # print(DX)        
    Func.append(func.mean().detach().cpu())

    Loss2 =  func.mean()
    Optimizer.zero_grad()
    Loss.backward()
    # torch.nn.utils.clip_grad_norm_(Dynamics.parameters(), max_norm=1e6)  # Apply gradient clipping
    Optimizer.step()
    
    

        
    if i%5 == 0:
        print("Iteration step = ", i, ", Loss = ", Loss.detach(),\
              "func = ",func.mean().detach(), "Beta_sum = ", beta_sum.detach(),\
              "Alpha_sum = ", alpha_sum.detach())
        sns.set()
        plt.figure(1)
        plt.plot(Func)
        # plt.ylim([3., 9])
    #     plt.savefig(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.png")
    #     np.save(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.npy", Func)
    #     # np.save(f"results/Fnc2_v4_alpha_N{N}_M{M}_2.npy", alpha_sum.detach())
    #     # np.save(f"results/Fnc2_v4_Langevin_N{N}_M{M}_2.npy", beta_sum.detach())
        plt.show()
        
        
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
        plt.show()

        np.save(f"results/LogReg_PDM__diff_F_N{N}_M{M}.npy", Delta_f)
        np.save(f"results/LogReg_PDM__diff_X_N{N}_M{M}.npy", Delta_x)