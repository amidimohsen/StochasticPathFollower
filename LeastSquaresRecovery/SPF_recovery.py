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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

torch.manual_seed(2)
torch.manual_seed(667)


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



class Dynamics_NN(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.hidden = 32
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
        hidden1_drift =   LipSwish( self.Lay1_norm(self.Lay1_drift(State) ) )
        hidden2_drift =   LipSwish( self.Lay2_norm(self.Lay2_drift(hidden1_drift) ) ) #best:tanh
        hidden3_drift = LipSwish( self.Lay3_drift(hidden2_drift) )  #best:tanh
        Alpha         = self.Lay_alpha(hidden3_drift)
        
        hidden1_diffusion =   LipSwish( self.Lay3_norm(self.Lay1_diffusion(State) ))
        hidden2_diffusion =  LipSwish( self.Lay4_norm(self.Lay2_diffusion(hidden1_diffusion) ) ) #best:tanh
        hidden3_diffusion = LipSwish( self.Lay3_diffusion(hidden2_diffusion) )  #best:tanh
        Beta              = self.Lay_beta(hidden3_diffusion)
        
        return Alpha, Beta
                
  
    
  # Parameters of OU-rocess
Mu = 0.02
theta = 0.1
Sigma = torch.tensor([0.4])
    
N_dim_X     = 5
N_dim_A_row = 10
    
    
# Parameters
t_start   = torch.tensor(0.001).to(device)
t_end     = torch.tensor(1.).to(device)
T         = t_end - t_start  # time horizon
N         = 40              # number of time steps 80
M         = 400              # 100number of sample paths 2000
dt        = T / N            # time step size
n_episode = 20000

 
A = 4 * torch.rand(size=[N_dim_A_row , N_dim_X])
e = torch.rand(size=[N_dim_X]).repeat([M,1])
Func = []
# Model
Dynamics = Dynamics_NN(dim=1).to(device)
Norm_grad= []
# Optimizer
Optimizer = torch.optim.Adam(params = Dynamics.parameters(), lr = 8e-3)
# with torch.autograd.set_detect_anomaly(True):
for i in range(n_episode):
    # initialiation
    
    X     =  0 * torch.ones(size = (M, N_dim_X, N), requires_grad=True).to(device)
    Y     =  0 * torch.ones(size = (M, N), requires_grad=False).to(device)
    DX    = torch.zeros(size = (M, N), requires_grad=True).to(device)
    W     = torch.zeros(size=(M, N), requires_grad=False).to(device)
    

    beta_sum, alpha_sum = 0, 0
    state_list = []
    alpha_list = []
    beta_list  = []
    dW_list  = []
    
    for n in range(N-1):
        time_value = t_start + n * dt
        time       = time_value * (torch.ones(size=[M], requires_grad=True).to(device))
        
        #X = X.detach().clone().requires_grad_(True)
        State = torch.cat( (X[:,:,n], time[:,None]), dim=1)
        state_list.append(State)
        
        alpha_Xt, beta_Xt = Dynamics.forward( State )
        beta_list.append(beta_Xt)
        beta_sum  = beta_sum + beta_Xt.mean().abs()
        alpha_sum = alpha_sum + alpha_Xt.mean().abs()
            
        dW_t    = torch.randn(size=[M,1]).to(device) * torch.sqrt(dt)
        dW_list.append(dW_t)
        
        ''' Without cloning: '''
        Wnew = torch.zeros_like(W)
        Wnew[:, 0:n+1] = W[:, 0:n+1]  # Copy existing values up to t
        Wnew[:, n+1]   = Wnew[:, n] + dW_t[:,0]      # Update specific step
        W = Wnew
        dY =  (Mu * time_value - theta * Y[:, n]) * dt + Sigma * dW_t[:,0] 
        Y[:, n+1] = Y[:, n] + dY      # Update specific step


        dX =  (alpha_Xt * dt + beta_Xt * dW_t)
        ''' Without cloning: '''
        Xnew = torch.zeros_like(X)
        Xnew[:, :, 0:n+1] = X[:, :, 0:n+1]  # Copy existing values up to t
        Xnew = Xnew.clone() 
        Xnew[:, :, n+1] = Xnew[:, :, n] + dX      # Update specific step
        X = Xnew#.detach()
        
    
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
    

    ''' Simulating Malliavin Derivative, forwards in time  for all t'''    
    Gamma_s = []
    I_matrix = torch.eye(N_dim_X, device=device, dtype=Jacob_alpha_all.dtype, requires_grad=False)
    I_all = I_matrix.unsqueeze(0).unsqueeze(0).expand(M, N, -1, -1).clone() 
    diffusion_term = Jacob_beta_all * dW_all[...,None,None]
    F_aux = I_all + Jacob_alpha_all * dt + diffusion_term
    for s in range(N):
        F_aux2    = F_aux[:,s:-1,:,:] # 
        I_start = I_matrix.unsqueeze(0).expand(M, -1, -1).unsqueeze(1) 
        F_aux3    = torch.cat((I_start, F_aux2), dim=1)
        F_cumprod = cumprod_mat(F_aux3, dim=1, right_multiply=True)
        Gamma_s.append( F_cumprod )
        
    Zeros2 = torch.zeros(M, N, N_dim_X, N_dim_X, device=device, dtype=Jacob_alpha_all.dtype, requires_grad=False)
    Gamma_s_t = []
    for s in range(N):
        gamma_s = Gamma_s[s]
        zeros_prefix = Zeros2[:, :s, :, :] 
        # DsXt.append( torch.einsum("Mjkt, Mkdt -> Mjdt", torch.cat([Gamma_t[t], Zeros2[:,:,:,0:N_t-t-1] ], dim=3), torch.cat([ b_func(X, u)[0][:,:,:,:t+1], Zeros1[:,:,:,0:N_t-t-1] ], dim=3)) )
        Gamma_s_t.append( torch.cat([zeros_prefix, gamma_s ], dim=1)  )
    Gamma_s_t = torch.stack(Gamma_s_t, dim=1)                            # Dimension: [N_path, N_t with indices $s$, N_t with indices $t$ , dim_X, dim_X]
    DsXt      = torch.einsum("Mstij, Msj -> Msti", Gamma_s_t, torch.stack(beta_list, dim=1))  # Dimension: [N_path, N_t with indices $s$, N_t with indices $t$ , dim_X]
    


    ''' Malliavin Derivative for OT-process (Y_t),  for all t'''    
    DY_t = []
    for t in range(N):
        DY  = []
        Aux = Sigma*torch.ones(size=[M])
        for s in range(t+1):
            new_val = Aux * torch.exp( - theta * dt)
            DY.append(Aux)            
            Aux = new_val
        DY_t.append(torch.stack(DY, dim=1).fliplr())
        
        
        
    
    Loss = 0
    for n in range(N):
        t = 1# n*dt
        func      = (A @ X[:,:,n].T - A @ (Y[:,n][:,None]*e).T - W[:,n][None,:]/np.sqrt(t+1e-12)).norm(dim=0).mean()
        func_X    = (A.T) @ (A @ X[:,:,n].T - A @ (Y[:,n][:,None]*e).T - W[:,n][None,:]/np.sqrt(t+1e-12)) 
        
        for s in range(n+1):
            Loss = Loss + 0*( A.T @ (A @ DsXt[:, s, n,:].T - A @ (DY_t[n][:, s][:,None]*e).T - 1/np.sqrt(t+1e-12)) ).mean(dim=1).norm()**2 +\
                          (func_X).mean(dim=1).norm()**2
            # Loss = Loss + ( A.T @ (A * DX_t[n][:, s] - A * DY_t[n][:, s] - 1) ).mean()**2 + (func_X).mean()**2
            
    # print(DX)        
    Func.append(func.mean().detach().cpu())


    # Norm_grad = 0
    # Optimizer.zero_grad()
    # Loss.backward(retain_graph=True)             
    # for (_, params) in enumerate(Dynamics.parameters()):
    #     if (params.grad != None):
    #         Norm_grad += params.grad.norm()
            
            
    # print(Norm_grad/1e4)        
    Loss2 =  func.mean()
    Optimizer.zero_grad()
    Loss.backward(retain_graph=True)
    # torch.nn.utils.clip_grad_norm_(Dynamics.parameters(), max_norm=1e6)  # Apply gradient clipping
    Optimizer.step()
    
    

        
    if i%10 == 0:
        print("Iteration step = ", i, ", Loss = ", Loss.detach(),\
              "func = ",func.mean().detach(), "Beta_sum = ", beta_sum.detach(),\
              "Alpha_sum = ", alpha_sum.detach())
        sns.set()
        plt.figure(1)
        plt.plot(Func)
        plt.ylim([0., 2])
    #     plt.savefig(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.png")
    #     np.save(f"Fnc2_OU_Malliavin_func_N{N}_M{M}.npy", Func)
    #     # np.save(f"results/Fnc2_v4_alpha_N{N}_M{M}_2.npy", alpha_sum.detach())
    #     # np.save(f"results/Fnc2_v4_Langevin_N{N}_M{M}_2.npy", beta_sum.detach())
        plt.show()
        
        
    if i%10 == 0:
        plt.figure(2)
        plt.plot(X[0:M:,0, :].cpu().detach().T)
        plt.figure(3)
        plt.plot(X[0:M:,1, :].cpu().detach().T)
        plt.figure(4)
        plt.plot( ((X-e[:,:,None]@Y[:,None,:]).norm(dim=1)**2).mean(dim=0).detach() )
        plt.show()
    #     plt.savefig(f"Fnc2_OU_Malliavin_paths_N{N}_M{M}.png")
        plt.show()
    #     np.save(f"Fnc2_OU_Malliavin_paths_N{N}_M{M}.npy", X.cpu().detach().T)