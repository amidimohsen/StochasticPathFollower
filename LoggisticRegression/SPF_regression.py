# In the name of ALLAH
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:23:24 2024

@author: amidzam1
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

torch.manual_seed(0)


def cumprod_mat(A, dim=-2, right_multiply=False):
    """
    Cumulative matrix product along a time axis for a batch of matrices.
    Out[..., t, :, :] = A[..., 0, :, :] @ A[..., 1, :, :] @ ... @ A[..., t, :, :]
    (when right_multiply=False).
    """
    A = A.movedim(dim, 0)  # (T, ..., n, n)
    T = A.shape[0]
    n1, n2 = A.shape[-2], A.shape[-1]
    if n1 != n2:
        raise ValueError("cumprod_mat requires square matrices (n x n) at the trailing dims")

    out_list = [A[0]]
    if not right_multiply:
        for t in range(1, T):
            out_list.append(out_list[t - 1] @ A[t])
    else:
        for t in range(1, T):
            out_list.append(A[t] @ out_list[t - 1])

    out = torch.stack(out_list, dim=0)
    out = out.movedim(0, dim)
    return out


def LipSwish(x):
    return 0.909 * x * torch.sigmoid(x)


def Tanh(x):
    return torch.tanh(x)


def Gelu(x):
    return F.gelu(x)


# Dimensions
N_dim_X = 5
N_dim_A_row = 10

class Dynamics_NN(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden = 32
        self.Lay1_drift = torch.nn.Linear(in_features=N_dim_X + 1, out_features=self.hidden)
        self.Lay2_drift = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.Lay3_drift = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.Lay_alpha  = torch.nn.Linear(in_features=self.hidden, out_features=N_dim_X)

        self.Lay1_diffusion = torch.nn.Linear(in_features=N_dim_X + 1, out_features=self.hidden)
        self.Lay2_diffusion = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.Lay3_diffusion = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.Lay_beta = torch.nn.Linear(in_features=self.hidden, out_features=N_dim_X)

        self.Lay1_norm = torch.nn.LayerNorm(normalized_shape=self.hidden)
        self.Lay2_norm = torch.nn.LayerNorm(normalized_shape=self.hidden)
        self.Lay3_norm = torch.nn.LayerNorm(normalized_shape=self.hidden)
        self.Lay4_norm = torch.nn.LayerNorm(normalized_shape=self.hidden)

    def forward(self, State):
        hidden1_drift = Tanh(self.Lay1_drift(State))
        hidden2_drift = Tanh(self.Lay2_drift(hidden1_drift))
        hidden3_drift = Tanh(self.Lay3_drift(hidden2_drift))
        Alpha = self.Lay_alpha(hidden3_drift)

        hidden1_diffusion = Tanh(self.Lay1_diffusion(State))
        hidden2_diffusion = Tanh(self.Lay2_diffusion(hidden1_diffusion))
        hidden3_diffusion = Tanh(self.Lay3_diffusion(hidden2_diffusion))
        Beta = self.Lay_beta(hidden3_diffusion)
        return Alpha, Beta


# OU parameters (matrix form)
Theta = 0.1 * torch.randn(size=[N_dim_A_row, N_dim_A_row], device=device)
Mu_vec = .02 * torch.randn(size=[N_dim_A_row, 1], device=device)  # column vector
Sigma_vec = 4.0 * torch.randn(size=[N_dim_A_row, 1], device=device)  # column vector

# Time discretization
t_start = torch.tensor(0.001, device=device)
t_end = torch.tensor(1.0, device=device)
T_horizon = t_end - t_start
N = 40
M = 400
dt = T_horizon / N
n_episode = 20000

# Logistic model
A = torch.randn(size=[N_dim_A_row, N_dim_X], device=device)
S0 = torch.zeros(size=(N_dim_A_row, 1), device=device)
b0 = torch.sigmoid(S0)  # soft label at t=0
mu = 1e-3  # L2 regularization (logistic)

# Pre-optimize a static x_ at t=0 for initialization
x_ = torch.zeros(size=[N_dim_X], requires_grad=True, device=device)
opt_x = torch.optim.Adam(params=[x_], lr=1e-1)
for e in range(2000):  # fewer iters suffice
    z0 = A @ x_  # (n,)
    loss =  ( F.softplus(z0).sum() - (b0.squeeze(-1) * z0).sum() ) + 0.5 * mu * x_.pow(2).sum()
    opt_x.zero_grad()
    loss.backward()
    opt_x.step()

gen = torch.Generator(device=device)
Func = []

# Model
Dynamics = Dynamics_NN(dim=1).to(device)
opt_dyn = torch.optim.Adam(params=Dynamics.parameters(), lr=8e-3)

for i in range(n_episode):
    # Initialize paths
    X = x_[None, :, None].repeat(M, 1, N).requires_grad_(True)        # (M, d, N)
    S = torch.zeros(size=(M, N_dim_A_row, N), device=device)          # (M, nA, N), labels latent
    W = torch.zeros(size=(M, N), device=device)                       # (M, N)

    beta_sum, alpha_sum = 0.0, 0.0
    state_list, beta_list, dW_list = [], [], []

    # time marching
    for n in range(N - 1):
        time_value = t_start + n * dt
        time = time_value * torch.ones(size=[M], device=device, requires_grad=True)
        State = torch.cat((X[:, :, n], time[:, None]), dim=1)  # (M, d+1)
        state_list.append(State)

        alpha_Xt, beta_Xt = Dynamics(State)  # (M, d), (M, d)
        beta_list.append(beta_Xt)
        beta_sum += beta_Xt.mean().abs()
        alpha_sum += alpha_Xt.mean().abs()

        gen.manual_seed(n)
        dW_t = torch.randn(size=[M, 1], generator=gen, device=device) * torch.sqrt(dt)
        dW_list.append(dW_t)

        # Brownian path update
        W[:, n + 1] = W[:, n] + dW_t[:, 0]

        # OU update for S: dS = ( - S @ Theta^T + (Theta @ Mu)^T ) dt + dW_t @ Sigma^T
        drift_S = (- S[:, :, n] @ Theta.T) + (Theta @ Mu_vec).T  # (M, nA)
        diff_S = dW_t @ Sigma_vec.T                             # (M, nA)
        S[:, :, n + 1] = S[:, :, n] + drift_S * dt + diff_S

        # State X update
        dX = alpha_Xt * dt + beta_Xt * dW_t  # (M, d)
        Xnew = torch.zeros_like(X)
        Xnew[:, :, :n + 1] = X[:, :, :n + 1]
        Xnew[:, :, n + 1] = X[:, :, n] + dX
        X = Xnew

    ''' obtaining the optimal solution and optimal objective '''
    if i==0:
        try:
            X_optim    = torch.load("X_optim2.npy")
            func_opt_t = torch.load("func_opt_t2.npy")
        except:
            X_optim    = []
            func_opt_t = []
            for n in range(N):
                print(n)
                bt = 1/(1+torch.exp(-S[:,:,n]))
                x_opt = torch.ones(size = (M, N_dim_X), requires_grad=True)
                Optimizer_ = torch.optim.Adam(params=[x_opt], lr=1e-1)
                for e in range(2000):
                    loss0 =  ( torch.log(1 + torch.exp(x_opt@A.T)).sum(dim=1) - (bt@A*x_opt).sum(dim=1) ) + mu/2 * x_opt.norm(dim=1)**2
                    loss_ = loss0.mean()
                    # loss = objective(x)
                    Optimizer_.zero_grad()
                    loss_.backward()
                    Optimizer_.step()
                    # print(loss_)
                X_optim.append(x_opt)
                
                func_opt   =  ( torch.log( 1 + torch.exp( A@x_opt.T) ).sum(dim=0) - ((bt@A)*x_opt).sum(dim=1) )\
                            + mu * x_opt.norm(dim=1)**2/2
                func_opt_t.append(func_opt)
                            
            X_optim    = torch.stack(X_optim, dim=2)
            func_opt_t = torch.stack(func_opt_t, dim=1)
            torch.save(X_optim,"X_optim.npy")        
            torch.save(func_opt_t,"func_opt_t.npy")  
            
            

    # Final time step for derivatives
    n = N - 1
    time_value = t_start + n * dt
    time = time_value * torch.ones(size=[M], device=device, requires_grad=True)
    dW_t = torch.randn(size=[M, 1], device=device) * torch.sqrt(dt)
    dW_list.append(dW_t)
    dW_all = torch.cat(dW_list, dim=1)  # (M, N)

    State = torch.cat((X[:, :, n], time[:, None]), dim=1)
    alpha_Xt, beta_Xt = Dynamics(State)
    beta_list.append(beta_Xt)
    state_list.append(State)

    State_all = torch.cat(state_list, dim=0)  # (M*N, d+1)
    alpha_all, beta_all = Dynamics(State_all)

    # Jacobians wrt X (not time)
    Jacob_alpha_all = torch.zeros(size=[N * M, N_dim_X, N_dim_X], device=device)
    Jacob_beta_all = torch.zeros(size=[N * M, N_dim_X, N_dim_X], device=device)
    for l in range(N_dim_X):
        Ja = torch.autograd.grad(alpha_all[:, l], State_all,
                                 grad_outputs=torch.ones_like(alpha_all[:, l]),
                                 retain_graph=True, create_graph=True)[0][:, :N_dim_X]
        Jb = torch.autograd.grad(beta_all[:, l], State_all,
                                 grad_outputs=torch.ones_like(beta_all[:, l]),
                                 retain_graph=True, create_graph=True)[0][:, :N_dim_X]
        Jacob_alpha_all[:, l, :] = Ja
        Jacob_beta_all[:, l, :] = Jb

    Jacob_alpha_all = Jacob_alpha_all.reshape(N, M, N_dim_X, N_dim_X).permute(1, 0, 2, 3)  # (M, N, d, d)
    Jacob_beta_all = Jacob_beta_all.reshape(N, M, N_dim_X, N_dim_X).permute(1, 0, 2, 3)   # (M, N, d, d)

    # Malliavin for X
    I_d = torch.eye(N_dim_X, device=device)
    I_all = I_d.unsqueeze(0).unsqueeze(0).expand(M, N, -1, -1).clone()
    diffusion_term = Jacob_beta_all * dW_all[..., None, None]
    F_aux = I_all + Jacob_alpha_all * dt + diffusion_term

    Gamma_s = []
    for s in range(N):
        F_aux2 = F_aux[:, s:-1, :, :]    # (M, N-s-1, d, d)
        I_start = I_d.unsqueeze(0).expand(M, -1, -1).unsqueeze(1)  # (M, 1, d, d)
        F_aux3 = torch.cat((I_start, F_aux2), dim=1)  # (M, N-s, d, d)
        F_cumprod = cumprod_mat(F_aux3, dim=1, right_multiply=True)  # (M, N-s, d, d)
        Gamma_s.append(F_cumprod)

    Zeros2 = torch.zeros(M, N, N_dim_X, N_dim_X, device=device)
    Gamma_s_t = []
    for s in range(N):
        gamma_s = Gamma_s[s]
        zeros_prefix = Zeros2[:, :s, :, :]
        Gamma_s_t.append(torch.cat([zeros_prefix, gamma_s], dim=1))
    Gamma_s_t = torch.stack(Gamma_s_t, dim=1)  # (M, N_s, N_t, d, d)
    DsXt = torch.einsum("Mstij, Msj -> Msti", Gamma_s_t, torch.stack(beta_list, dim=1))  # (M, N, N, d)

    # Malliavin for S (linear OU: Jacob_beta_S_all = 0, Jacob_alpha_S_all = -Theta)
    Jacob_beta_S_all = torch.zeros(size=[M, N, N_dim_A_row, N_dim_A_row], device=device)
    Jacob_alpha_S_all = -Theta.unsqueeze(0).unsqueeze(0).expand(M, N, -1, -1).contiguous()
    betaS_all = Sigma_vec[:, 0].unsqueeze(0).unsqueeze(0).expand(M, N, -1)  # (M, N, nA)

    I_A = torch.eye(N_dim_A_row, device=device)
    I_allA = I_A.unsqueeze(0).unsqueeze(0).expand(M, N, -1, -1).clone()
    diffusion_term_S = Jacob_beta_S_all * dW_all[..., None, None]
    F_auxS = I_allA + Jacob_alpha_S_all * dt + diffusion_term_S

    GammaS_s = []
    for s in range(N):
        F_aux2S = F_auxS[:, s:-1, :, :]
        I_startS = I_A.unsqueeze(0).expand(M, -1, -1).unsqueeze(1)
        F_aux3S = torch.cat((I_startS, F_aux2S), dim=1)
        GammaS_s.append(cumprod_mat(F_aux3S, dim=1, right_multiply=True))

    Zeros2A = torch.zeros(M, N, N_dim_A_row, N_dim_A_row, device=device)
    GammaS_s_t = []
    for s in range(N):
        gammaS_s = GammaS_s[s]
        zeros_prefix = Zeros2A[:, :s, :, :]
        GammaS_s_t.append(torch.cat([zeros_prefix, gammaS_s], dim=1))
    GammaS_s_t = torch.stack(GammaS_s_t, dim=1)  # (M, N, N, nA, nA)
    DsSt = torch.einsum("Mstij, Msj -> Msti", GammaS_s_t, betaS_all)  # (M, N, N, nA)

    # Precompute A⊗A for Hessian assembly
    AA = torch.einsum('ni,nj->nij', A, A)  # (nA, d, d)

    Loss = 0.0
    func_for_plot = None

    for n in range(N):
        bt = torch.sigmoid(S[:, :, n])          # (M, nA), soft labels
        ct = bt * (1.0 - bt)                    # (M, nA), derivative d bt / d S

        z = A @ X[:, :, n].T                    # (nA, M)
        splus = F.softplus(z)                   # stable log(1+exp)
        sigma = torch.sigmoid(z)                # (nA, M)

        # per-path objective (M,)
        func_n = ( splus.sum(dim=0) - (bt * z.T).sum(dim=1) ) + 0.5 * mu * X[:, :, n].pow(2).sum(dim=1)
        # print(func_n.mean())
        # per-path gradient wrt x_n (M, d): grad = (A^T(σ - b))^T + μ x
        grad_n = (A.T @ (sigma - bt.T)).T + mu * X[:, :, n]

        # per-path Hessian wrt x_n (M, d, d): H = sum_i w_i a_i a_i^T, w_i = σ(1-σ)
        w = sigma * (1.0 - sigma)               # (nA, M)
        H_n = torch.einsum('nM, nij -> Mij', w, AA)  # (M, d, d); μI added separately below in residual

        # Malliavin residual sum over s <= n
        mall_term = 0.0
        for s in range(n + 1):
            v = DsXt[:, s, n, :].unsqueeze(-1)               # (M, d, 1)
            Dsbt = ct * DsSt[:, s, n, :]                     # (M, nA)
            rhs = (Dsbt @ A).unsqueeze(-1)                   # (M, d, 1)
            # residual: (H + μI) v - (∂b/∂S DsS) A
            residual = (H_n @ v + mu * v) - rhs              # (M, d, 1)
            mall_term = mall_term + residual.mean(dim=0).norm()**2+ grad_n.mean(dim=0).norm()**2

        # add gradient term once per n
        Loss = Loss + mall_term 

        # store for plotting
        if n == N - 1:
            func_for_plot = func_n.mean().detach().cpu()

    Func.append(func_for_plot)
    Loss2 =  func_n.mean()
    
    opt_dyn.zero_grad()
    Loss.backward()  # no retain_graph to avoid leaks
    # Optional gradient clipping:
    # torch.nn.utils.clip_grad_norm_(Dynamics.parameters(), max_norm=1e3)
    opt_dyn.step()

    if i % 5 == 0:
        print(f"Iter {i:5d} | Loss={Loss.item():.4e} | func={Func[-1].item():.4e} "
              f"| Beta_sum={float(beta_sum):.3e} | Alpha_sum={float(alpha_sum):.3e}")
        sns.set()
        plt.figure(1); plt.clf(); plt.plot(Func); plt.title("Mean objective at final time"); plt.show()

        if 'X_optim' in locals():
            plt.figure(2); plt.clf()
            plt.plot(((X_optim.detach() - X.detach()).norm(dim=1)**2).mean(dim=0).cpu())
            plt.title("E|X-X*|^2")
            plt.show()
            