# In the name of ALLAH
# -*- coding: utf-8 -*-
"""
Memory-optimized version:
- O(1) memory in time (no [N] axis for states/sensitivities)
- In-place updates, no per-step zeros_like clones
- Optional jacobian micro-batching to cap peak memory
- Uses jacrev (reverse-mode) for better perf/memory when DIm >> N_dim_X
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.func import vmap, functional_call, jacrev  # use jacrev instead of jacfwd
from collections import OrderedDict
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cpu")

seed = 1
torch.manual_seed(seed)

def flatten_per_sample_jacobian(jac_pytree: OrderedDict):
    # OrderedDict of [M_chunk, out_dim, *param_shape] -> [M_chunk, out_dim, DIm]
    flat_chunks = []
    for g in jac_pytree.values():
        flat_chunks.append(g.flatten(start_dim=2))
    return torch.cat(flat_chunks, dim=2)

def LipSwish(x):
    return 0.909 * x * torch.nn.functional.sigmoid(x)

# Problem sizes and constants
N_dim_X     = 5
N_dim_A_row = 10

class Dynamics_NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 16
        self.Lay1_drift = torch.nn.Linear(N_dim_X + 1, hidden)
        self.Lay2_drift = torch.nn.Linear(hidden, hidden)
        self.Lay3_drift = torch.nn.Linear(hidden, hidden)
        self.Lay_alpha  = torch.nn.Linear(hidden, N_dim_X)

        self.Lay1_diffusion = torch.nn.Linear(N_dim_X + 1, hidden)
        self.Lay2_diffusion = torch.nn.Linear(hidden, hidden)
        self.Lay3_diffusion = torch.nn.Linear(hidden, hidden)
        self.Lay_beta       = torch.nn.Linear(hidden, N_dim_X)

    def forward(self, State):
        h1 = LipSwish(self.Lay1_drift(State))
        h2 = LipSwish(self.Lay2_drift(h1))
        h3 = LipSwish(self.Lay3_drift(h2))
        Alpha = self.Lay_alpha(h3)

        h1d = LipSwish(self.Lay1_diffusion(State))
        h2d = LipSwish(self.Lay2_diffusion(h1d))
        h3d = LipSwish(self.Lay3_diffusion(h2d))
        Beta  = self.Lay_beta(h3d)
        return Alpha, Beta

# OU parameters
Mu = 0.02
theta = 0.1
Sigma = torch.tensor([0.4], device=device)

# Time/grid
t_start   = torch.tensor(0.001, device=device)
t_end     = torch.tensor(1.0, device=device)
T         = t_end - t_start
N         = 40     # time steps
M         = 100    # batch size
dt        = T / N
n_episode = 20000

# Data / matrices
A = (1 * torch.randn(size=[N_dim_A_row, N_dim_X], device=device))
e = torch.rand(size=[N_dim_X], device=device).repeat([M, 1])

# Model/opt
Dynamics = Dynamics_NN().to(device)
Optimizer = torch.optim.Adam(params=Dynamics.parameters(), lr=8e-3)

# Derived sizes
DIm = sum(p.numel() for p in Dynamics.parameters())
print("Dim = ", DIm)
# Controls
DO_PLOTS = True         # plotting uses extra memory
JAC_CHUNK = None         # e.g., set to 25 to compute jacobians in micro-batches over M
VARX_DTYPE = torch.float32  # can try torch.bfloat16/float16 (on supported hw) to halve VarX memory

# Per-sample vector outputs for jac computation
def alpha_vec(mod, p, b, x_single):
    a, _ = functional_call(mod, (p, b), (x_single.unsqueeze(0),))
    return a.squeeze(0)  # [N_dim_X]

def beta_vec(mod, p, b, x_single):
    _, bt = functional_call(mod, (p, b), (x_single.unsqueeze(0),))
    return bt.squeeze(0)  # [N_dim_X]

def per_sample_param_jacobians(mod, params, buffers, State, chunk=None):
    """
    Returns:
      diff_theta_alpha, diff_theta_beta: [M, N_dim_X, DIm]
    Computes per-sample Jacobians in chunks across batch to limit peak memory.
    """
    Mtot = State.shape[0]
    if chunk is None or chunk >= Mtot:
        grads_alpha_tree = vmap(jacrev(alpha_vec, argnums=1), in_dims=(None, None, None, 0))(mod, params, buffers, State)
        grads_beta_tree  = vmap(jacrev(beta_vec,  argnums=1), in_dims=(None, None, None, 0))(mod, params, buffers, State)
        diff_alpha = flatten_per_sample_jacobian(grads_alpha_tree)  # [M, N_dim_X, DIm]
        diff_beta  = flatten_per_sample_jacobian(grads_beta_tree)
        return diff_alpha, diff_beta
    else:
        out_alpha = []
        out_beta = []
        for start in range(0, Mtot, chunk):
            end = min(start + chunk, Mtot)
            St_chunk = State[start:end]
            g_alpha = vmap(jacrev(alpha_vec, argnums=1), in_dims=(None, None, None, 0))(mod, params, buffers, St_chunk)
            g_beta  = vmap(jacrev(beta_vec,  argnums=1), in_dims=(None, None, None, 0))(mod, params, buffers, St_chunk)
            out_alpha.append(flatten_per_sample_jacobian(g_alpha))
            out_beta.append(flatten_per_sample_jacobian(g_beta))
        return torch.cat(out_alpha, dim=0), torch.cat(out_beta, dim=0)

Func = []
t0 = time.time()

for i in range(n_episode):
    # Important: refresh params so jacobians reflect the current model weights
    params  = OrderedDict(Dynamics.named_parameters())
    buffers = OrderedDict(Dynamics.named_buffers())

    # States at current time (no [N] dimension)
    X    = torch.zeros(M, N_dim_X, device=device)             # [M, N_dim_X]
    VarX = torch.zeros(M, N_dim_X, DIm, device=device, dtype=VARX_DTYPE)  # [M, N_dim_X, DIm]
    Y    = torch.zeros(M, device=device)                      # [M]
    W    = torch.zeros(M, device=device)                      # [M]

    Loss = torch.tensor(0.0, device=device)

    # If you want to plot sample paths, you can store thin history (on CPU) for a few samples
    if DO_PLOTS:
        X_hist0 = []
        X_hist1 = []

    # We compute the per-time loss term at the beginning of each step (n=0..N-1).
    # That matches the original sum over n using X[:,:,n], VarX[:,:,n,:].
    for n in range(N-1):
        # Time scalar and broadcast column
        Time_value = t_start + n * dt
        Time = torch.full((M, 1), fill_value=Time_value.item(), device=device)

        # Loss term for this time index n, using current X, Y, W, VarX
        # func: scalar
        # func_X: [N_dim_X, M]
        t_scalar = 1.0  # as in your code (you had t=1 instead of n*dt)
        rhs = (A @ X.T) - (A @ (Y[:, None] * e).T) - (W[None, :] / np.sqrt(t_scalar + 1e-12))
        func = rhs.norm(dim=0).mean()
        func_X = (A.T @ rhs)                               # [N_dim_X, M]
        # Contract across state dim, mean over batch -> [DIm], then L2^2
        Loss = Loss + torch.einsum("mn, mnd -> md", func_X.T, VarX).mean(dim=0).norm()**2

        # Forward step: build State, forward pass, per-sample param Jacobians
        State = torch.cat((X, Time), dim=1)                # [M, N_dim_X+1]
        alpha_Xt, beta_Xt = Dynamics(State)                # [M, N_dim_X] each

        diff_theta_alpha, diff_theta_beta = per_sample_param_jacobians(
            Dynamics, params, buffers, State, chunk=JAC_CHUNK
        )  # [M, N_dim_X, DIm] each

        # dW and state updates (in-place)
        dW_t = torch.randn(M, device=device) * torch.sqrt(dt)
        # Y and W don't require grad; update in-place
        Y.add_((Mu * Time_value - theta * Y) * dt + Sigma * dW_t)
        W.add_(dW_t)

        # Sensitivity and state updates
        # VarX dtype may be lower precision; cast for arithmetic then cast back
        d_VarX = diff_theta_alpha * dt + diff_theta_beta * dW_t[:, None, None]
        VarX = VarX + d_VarX.to(VarX.dtype)

        dX = alpha_Xt * dt + beta_Xt * dW_t[:, None]
        X.add_(dX)

        if DO_PLOTS:
            X_hist0.append(X[:min(M, 50), 0].detach().cpu())
            X_hist1.append(X[:min(M, 50), 1].detach().cpu())

    # Add the final time index (n = N-1) loss with the current state
    rhs = (A @ X.T) - (A @ (Y[:, None] * e).T) - (W[None, :] / np.sqrt(1.0 + 1e-12))
    func = rhs.norm(dim=0).mean()
    func_X = (A.T @ rhs)
    Loss = Loss + torch.einsum("mn, mnd -> md", func_X.T, VarX).mean(dim=0).norm()**2

    Func.append(func.detach().cpu())

    # Optimize
    Optimizer.zero_grad()
    # Avoid retain_graph unless you need multiple backward passes
    Loss.backward()
    Optimizer.step()

    if i == 0:
        print("average-runtime for first iteration =", time.time() - t0, "sec", "DIm =", DIm)

    if i % 5 == 0:
        print(f"Iter {i} | Loss={Loss.item():.6f} | func={func.item():.6f}")
        if DO_PLOTS:
            sns.set()
            plt.figure(1); plt.plot(Func); plt.ylim([0., 2]); plt.show()
            plt.figure(2); plt.plot(torch.stack(X_hist0, dim=1).T); plt.show()
            plt.figure(3); plt.plot(torch.stack(X_hist1, dim=1).T); plt.show()