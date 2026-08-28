"""
PINN for networked vector-host model using a NORMALIZED RANDOM-WALK network.

This script is the original PINN code with:
 - corrected movement operator (uses u @ M for (M^T u))
 - analytic per-node R02 and steady-state initial conditions integrated
 - the NumPy RK4 diagnostic retained for quick ODE checks

Run the script; the diagnostic prints initial dI_h/dt, I_h after the short sim,
and a growth ratio for the seeded node. After verifying the ODE diagnostic,
you can proceed to train the PINN (training loop is included and active).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
print(f"Using device: {device}")

# ---------------------------
# Random graph builder (Erdős-Rényi with min-degree guarantee)
# ---------------------------
def random_adj(n: int, p: float = 0.103, seed: int = None, min_degree: int = 3) -> np.ndarray:
    rng = np.random.RandomState(seed)
    A = (rng.rand(n, n) < p).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    for i in range(n):
        deg = int(A[i].sum())
        if deg < min_degree:
            candidates = [j for j in range(n) if j != i and A[i, j] == 0]
            rng.shuffle(candidates)
            for j in candidates[:min_degree - deg]:
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A

# ---------------------------
# Build graph and normalized random-walk matrices
# ---------------------------
N_nodes = 30
A = random_adj(N_nodes, p=0.103, seed=1)
G = nx.from_numpy_array(A)

deg   = A.sum(axis=1)                          # (N,) node degrees
P     = A / deg[:, np.newaxis]                 # (N, N)  P_ij = A_ij / d_i  (row-stochastic)

# Diffusion scalars per compartment
c_Sh = 1.0   # for S_h
c_Ih = 1.0   # for I_h
c_Sv = 0.0  # for S_v
c_Iv = 0.0  # for I_v

# Movement rate matrices  M_ij = c * P_ij  (normalized by source degree)
M_sh     = c_Sh * P
Sigma_ih = c_Ih * P
Nu_sv    = c_Sv * P
Nu_iv    = c_Iv * P

# torch tensors
A_torch      = torch.tensor(A,        dtype=torch.float32, device=device)
M_sh_t       = torch.tensor(M_sh,     dtype=torch.float32, device=device)
Sigma_ih_t   = torch.tensor(Sigma_ih, dtype=torch.float32, device=device)
Nu_sv_t      = torch.tensor(Nu_sv,    dtype=torch.float32, device=device)
Nu_iv_t      = torch.tensor(Nu_iv,    dtype=torch.float32, device=device)

# Print graph statistics
print(f"\nRandom graph  |  N={N_nodes}, p=0.08, min_degree=3, seed=1")
print(f"  Edges                         : {G.number_of_edges()}")
print(f"  Average degree                : {np.mean(deg):.2f}  (min={deg.min():.0f}, max={deg.max():.0f})")
print(f"  Average clustering coefficient: {nx.average_clustering(G):.4f}")
print(f"  Average shortest path length  : {nx.average_shortest_path_length(G):.4f}")
print(f"\n  Row-sum of P (should all be 1): min={P.sum(axis=1).min():.6f}, "
      f"max={P.sum(axis=1).max():.6f}")

# ---------------------------
# Model parameters (per-node vectors)
# ---------------------------
beta_h  = np.full(N_nodes, 0.022,    dtype=np.float32)
beta_v  = np.full(N_nodes, 0.48,     dtype=np.float32)
mu_h    = np.full(N_nodes, 0.000016,    dtype=np.float32)   # human death rate
Lambda_h = np.full(N_nodes, 0.033,   dtype=np.float32)
Lambda_v = np.full(N_nodes, 0.13,    dtype=np.float32)
gamma   = 0.000055
theta   = 0.00009
mu_v    = 0.033
delta_v = 0.0025   # vector extra mortality

eps = 1e-12
d = mu_h + gamma + theta                     # (N,)
beta_h = (1.0 + 10.0) * d * (mu_v + delta_v)**2 * Lambda_h / (beta_v * mu_h * Lambda_v + eps)
# torch tensors
beta_h_t   = torch.tensor(beta_h,   dtype=torch.float32, device=device)
beta_v_t   = torch.tensor(beta_v,   dtype=torch.float32, device=device)
mu_h_t     = torch.tensor(mu_h,     dtype=torch.float32, device=device)
Lambda_h_t = torch.tensor(Lambda_h, dtype=torch.float32, device=device)
Lambda_v_t = torch.tensor(Lambda_v, dtype=torch.float32, device=device)
gamma_t    = torch.tensor(float(gamma),   dtype=torch.float32, device=device)
theta_t    = torch.tensor(float(theta),   dtype=torch.float32, device=device)
mu_v_t     = torch.tensor(float(mu_v),    dtype=torch.float32, device=device)
delta_v_t  = torch.tensor(float(delta_v), dtype=torch.float32, device=device)

# ---------------------------
# Analytic per-node R02 and steady-state initial conditions
# ---------------------------
# Formulas provided by the user; vectorized implementation with safeguards.

# eps = 1e-12

mu = mu_v + delta_v                          # # scalar mu (vector for consistency with vectorized ops)# scalar
# d = mu_h + gamma + theta                   # # d is vector (depends on mu_h) # (N,)
# beta_h = (1 + 0.01) * d * (mu_v + delta_v)**2 * Lambda_h / (beta_v * mu_h * Lambda_v + eps)

# compute b (vectorized)
b = d * ( (beta_v * beta_h * (mu_h + theta) / (d + eps)) - 2.0 * (mu**2) * Lambda_h - mu * Lambda_h * beta_v )

# R02 (per-node)
R02 = (beta_h * beta_v * mu_h * Lambda_v) / ( (mu_h + gamma + theta) * (mu + eps)**2 * Lambda_h + eps )

# discriminant for lambda_h
disc = b**2 - 4.0 * mu**3 * d**2 * Lambda_h**2 * (1.0 - R02) * (mu + beta_v)
disc = np.maximum(disc, 0.0)   # clamp negative numerical noise

# lambda_h (choose the root with + sqrt as in user's formula)
den_lambda_h = 2.0 * mu * Lambda_h * (mu + beta_v) + eps
lambda_h  = (b + np.sqrt(disc)) / den_lambda_h
# lambda_h = (b - np.sqrt(disc)) / den_lambda_h

# lambda_v
lambda_v = beta_v * lambda_h / (d + lambda_h + eps)

# Now compute steady-state compartments (vectorized)
S_h = Lambda_h / (mu_h + lambda_h * (1.0 - gamma / (d + eps)) + eps)
I_h = lambda_h * Lambda_h / ( (mu_h + lambda_h * (1.0 - gamma / (d + eps)) + eps) * (d + eps) )
S_v = Lambda_v / (mu + lambda_v + eps)
I_v = Lambda_v * lambda_v / (mu * (mu + lambda_v + eps) + eps)

# Replace any tiny negative numerical artifacts with zero
S_h = np.maximum(S_h, 0.0)
I_h = np.maximum(I_h, 0.0)
S_v = np.maximum(S_v, 0.0)
I_v = np.maximum(I_v, 0.0)

# Snapshot of all-node EE values — used only for var_scales below.
# The actual PINN initial condition (y0_full) is built further down
# with infecteds seeded at one node only.
ee_snapshot = np.stack([S_h, I_h, S_v, I_v], axis=1).astype(np.float32)

# Update R0 summary using R02 (per-node). Use mean for scalar summary, but print min/max.
R0 = float(np.mean(R02))
print(f"\nR0 (from R02 formula) mean ≈ {R0:.3f}  (per-node R02: min={float(np.min(R02)):.3f}, max={float(np.max(R02)):.3f})")
print("EE compartments (uniform across nodes):", ee_snapshot[0])

# ---------------------------
# Characteristic scales for loss normalization
# ---------------------------
# Use the EE compartment values as normalization scales for the ODE residual.
# These are the natural magnitudes of each state variable at steady state.
# A small floor avoids division by zero (e.g. if a compartment is near zero).
_min_I_scale = 1e-3   # floor for infected compartments
var_scales = np.stack([np.maximum(S_h, eps),
                       np.maximum(I_h, _min_I_scale),
                       np.maximum(S_v, eps),
                       np.maximum(I_v, _min_I_scale)], axis=1).astype(np.float32)  # (N, 4)

# ---------------------------
# PINN network
# ---------------------------
class PINNNet(nn.Module):
    def __init__(self, n_nodes: int, hidden=(128, 128), T_final: float = 200.0,
                 y0_anchor: np.ndarray = None):
        super().__init__()
        self.n_nodes = n_nodes
        self.T_final = float(T_final)
        self.node_embed = nn.Linear(n_nodes, 16, bias=False)
        self.time_fc    = nn.Linear(1, 16)
        layers  = []
        in_dim  = 32
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))   # unrestricted output; positivity via relu below
        self.mlp = nn.Sequential(*layers)
        # Hard IC: register y0 as a non-trainable buffer of shape (N_nodes, 4).
        # At t=0 the network returns y0[node] exactly, for every node.
        if y0_anchor is not None:
            self.register_buffer('y0_anchor',
                                 torch.tensor(y0_anchor, dtype=torch.float32))
        else:
            self.register_buffer('y0_anchor', None)

    def forward(self, t: torch.Tensor, node_onehot: torch.Tensor) -> torch.Tensor:
        t_norm = t / self.T_final                        # in [0, 1]
        te = torch.tanh(self.time_fc(t_norm))
        ne = torch.tanh(self.node_embed(node_onehot))
        x  = torch.cat([te, ne], dim=1)
        correction = self.mlp(x)                         # unrestricted (B, 4)
        # Hard IC: u(t=0, node_i) = y0[i] exactly.
        # one_hot_i @ y0_anchor selects row i of y0_anchor (B, 4).
        y0_i = node_onehot @ self.y0_anchor
        # relu clamps to [0, ∞); at t=0 output = relu(y0_i) = y0_i since y0 ≥ 0.
        return torch.relu(y0_i + t_norm * correction)

# ---------------------------
# Normalized random-walk movement operator
#   move_i = sum_j M_ij * u_i  -  sum_j M_ji * u_j
#          = (M @ 1)_i * u_i   -  (M^T u)_i
# For M = c*P:  sum_j M_ij = c (constant), so  move = c*u - c*(P^T u)
# ---------------------------
def movement_term(u: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    u : (B, N) batched field values
    M : (N, N) movement rate matrix  (M_ij = rate from i to j)
    returns net outflow (B, N)
    """
    row_sum = M.sum(dim=1).unsqueeze(0)   # (1, N)  = c for normalized P
    # Correct implementation: (M^T u)_i = sum_j M_{j,i} u_j is computed as u @ M
    return row_sum * u - u @ M            # (B, N)

# ---------------------------
# RHS of ODEs
# ---------------------------
def rhs_graph(states: torch.Tensor) -> torch.Tensor:
    """states: (B, N, 4)  →  rhs: (B, N, 4)"""
    S_h = states[:, :, 0]   # (B, N)
    I_h = states[:, :, 1]
    S_v = states[:, :, 2]
    I_v = states[:, :, 3]

    denom = S_h + I_h + 1e-8
    inf_h = beta_h_t.unsqueeze(0) * (S_h * I_v / denom)
    inf_v = beta_v_t.unsqueeze(0) * (S_v * I_h / denom)

    dS_h_local = Lambda_h_t.unsqueeze(0) - inf_h - mu_h_t.unsqueeze(0) * S_h + gamma_t * I_h
    dI_h_local = inf_h - (gamma_t + mu_h_t.unsqueeze(0) + theta_t) * I_h
    dS_v_local = Lambda_v_t.unsqueeze(0) - inf_v - (mu_v_t + delta_v_t) * S_v
    dI_v_local = inf_v - (mu_v_t + delta_v_t) * I_v

    dS_h = dS_h_local - movement_term(S_h, M_sh_t)
    dI_h = dI_h_local - movement_term(I_h, Sigma_ih_t)
    dS_v = dS_v_local - movement_term(S_v, Nu_sv_t)
    dI_v = dI_v_local - movement_term(I_v, Nu_iv_t)

    return torch.stack([dS_h, dI_h, dS_v, dI_v], dim=2)   # (B, N, 4)

# ---------------------------
# Helpers
# ---------------------------
def make_node_onehots(n_nodes: int, dev: torch.device) -> torch.Tensor:
    return torch.tensor(np.eye(n_nodes, dtype=np.float32), device=dev)

def pinn_loss(model: nn.Module, t_colloc: torch.Tensor, node_onehots: torch.Tensor,
              y0_full: np.ndarray, var_scales_np: np.ndarray) -> Tuple[torch.Tensor, dict]:
    N_t     = t_colloc.shape[0]
    N_nodes = node_onehots.shape[0]

    t_rep    = t_colloc.repeat_interleave(N_nodes, dim=0).requires_grad_(True)
    node_rep = node_onehots.repeat(N_t, 1)

    pred      = model(t_rep, node_rep)                        # (N_t*N, 4)
    pred_full = pred.view(N_t, N_nodes, 4)

    dpred_dt_cols = []
    for k in range(4):
        gk = torch.autograd.grad(pred[:, k].sum(), t_rep,
                                 create_graph=True, retain_graph=True)[0]
        dpred_dt_cols.append(gk)
    dpred_dt = torch.cat(dpred_dt_cols, dim=1).view(N_t, N_nodes, 4)

    rhs_full = rhs_graph(pred_full)

    scales   = torch.tensor(var_scales_np, dtype=torch.float32, device=device)  # (N, 4)
    res_norm = (dpred_dt - rhs_full) / (scales.unsqueeze(0) + 1e-8)
    res_loss = torch.mean(res_norm ** 2)

    # IC is hard-enforced by the network architecture:
    #   u(t=0, node_i) = y0[i] exactly  (see PINNNet.forward).
    # No separate ic_loss term is needed or computed.
    loss    = res_loss
    metrics = {"res_loss": res_loss.item(), "ic_loss": 0.0}
    return loss, metrics

# ---------------------------
# Replace initial_condition and y0_full creation so susceptibles are uniform
# across nodes and infected states are seeded at seed_node using analytic I_h/I_v.
# ---------------------------
# def initial_condition(N_nodes: int, seed_node: int = 0,
#                       use_analytic_seed: bool = True,
#                       S_h_uniform_from: str = "mean") -> np.ndarray:
#     """
#     Build initial condition y0_full (N,4) with:
#       - Susceptibles uniform across all nodes (S_h and S_v).
#         S_h_uniform_from: "mean" -> use mean(EE S_h); "seed" -> use S_h[seed_node];
#                           "dfe"  -> use DFE S_h (Lambda_h/mu_h).
#       - Infecteds zero everywhere except at seed_node where they are set to the
#         analytic EE infected values I_h[seed_node], I_v[seed_node] if
#         use_analytic_seed=True, otherwise a small numeric seed is used.
#     """
#     # Preconditions: analytic EE vectors S_h, I_h, S_v, I_v must exist
#     required = ('S_h' in globals() and 'I_h' in globals() and 'S_v' in globals() and 'I_v' in globals())
#     if not required:
#         raise RuntimeError("S_h, I_h, S_v, I_v must be defined before calling initial_condition().")

#     if not (0 <= seed_node < N_nodes):
#         raise ValueError(f"seed_node must be in [0, {N_nodes-1}]")
    
#     eps = 1e-12
#     mu = mu_v + delta_v
#     d = mu_h + gamma + theta
#     beta_h = (1.0 + 10.0) * d * (mu_v + delta_v)**2 * Lambda_h / (beta_v * mu_h * Lambda_v + eps)
#     b = d * ( (beta_v * beta_h * (mu_h + theta) / (d + eps)) - 2.0 * mu**2 * Lambda_h - mu * Lambda_h * beta_v )
#     R02 = (beta_h * beta_v * mu_h * Lambda_v) / ( (mu_h + gamma + theta) * mu**2 * Lambda_h + eps )
#     disc = b**2 - 4.0 * mu**3 * d**2 * Lambda_h**2 * (1.0 - R02) * (mu + beta_v)
#     disc = np.maximum(disc, 0.0)   # clamp negative numerical noise
#     den_lambda_h = 2.0 * mu * Lambda_h * (mu + beta_v) + eps
#     lambda_h = (b + np.sqrt(disc)) / den_lambda_h
#     # lambda_h = (b - np.sqrt(disc)) / den_lambda_h
#     lambda_v = beta_v * lambda_h / (d + lambda_h + eps)
#     S_h = Lambda_h / (mu_h + lambda_h * (1.0 - gamma / (d + eps)) + eps)
#     I_h = (lambda_h * Lambda_h) / ( (mu_h + lambda_h * (1.0 - gamma / (d + eps)) + eps) * (d + eps) )
#     S_v = Lambda_v / (mu + lambda_v + eps)
#     I_v = (Lambda_v * lambda_v) / (mu * (mu + lambda_v + eps) + eps)
    
#     # Determine uniform susceptible values
#     if S_h_uniform_from == "mean":
#         S_h_uniform = float(np.mean(S_h))
#         S_v_uniform = float(np.mean(S_v))
#     elif S_h_uniform_from == "seed":
#         S_h_uniform = float(S_h[seed_node])
#         S_v_uniform = float(S_v[seed_node])
#     elif S_h_uniform_from == "EE":
#         S_h_uniform = float(np.mean(S_h))
#         S_v_uniform = float(np.mean(S_v))
#     else:
#         raise ValueError("S_h_uniform_from must be one of: 'mean','seed','EE'")

#     # Build uniform susceptibles arrays
#     S_h0 = np.full(N_nodes, S_h_uniform, dtype=np.float32)
#     S_v0 = np.full(N_nodes, S_v_uniform, dtype=np.float32)

#     # Infecteds: zero everywhere except seed_node
#     I_h0 = np.zeros(N_nodes, dtype=np.float32)
#     I_v0 = np.zeros(N_nodes, dtype=np.float32)

#     if use_analytic_seed:
#         # use analytic EE infected values at seed_node
#         I_h0[seed_node] = float(I_h[seed_node])
#         I_v0[seed_node] = float(I_v[seed_node])
#     else:
#         # small numeric seed if requested
#         I_h0[seed_node] = max(1e-6, float(I_h[seed_node]))
#         I_v0[seed_node] = max(1e-6, float(I_v[seed_node]))

#     # Ensure non-negativity
#     S_h0 = np.maximum(S_h0, 0.0)
#     I_h0 = np.maximum(I_h0, 0.0)
#     S_v0 = np.maximum(S_v0, 0.0)
#     I_v0 = np.maximum(I_v0, 0.0)

#     y0 = np.stack([S_h0, I_h0, S_v0, I_v0], axis=1)  # (N,4)
#     return y0.astype(np.float32)

# # --- Replace previous y0_full assignment with the new initial_condition call ---
# # Choose seed_node and options as desired:
# seed_node = 0
# # Options:
# #   use_analytic_seed=True  -> seed infected at seed_node with analytic I_h/I_v
# #   S_h_uniform_from: "mean" | "seed" | "dfe"
# y0_full = initial_condition(N_nodes, seed_node=seed_node,
#                             use_analytic_seed=True,
#                             S_h_uniform_from="mean")

# # Debug print to confirm
# print("Initial condition (seed node):", y0_full[seed_node])
# print("Uniform S_h used:", float(y0_full[0,0]), "Uniform S_v used:", float(y0_full[0,2]))
# print("Initial I_h at seed node:", float(y0_full[seed_node,1]), "Initial I_v at seed node:", float(y0_full[seed_node,3]))

# ---------------------------
# Initial condition: uniform susceptibles, infected seeded at one node using analytic EE
# ---------------------------
def initial_condition_seeded(N_nodes: int, seed_node: int = 0,
                             S_h_uniform_from: str = "mean",
                             use_analytic_seed: bool = True,
                             min_seed: float = 1e-12) -> np.ndarray:
    """
    Build y0_full (N,4) with:
      - Susceptibles uniform across nodes (S_h and S_v).
        S_h_uniform_from: "mean" -> mean(EE S_h); "seed" -> S_h[seed_node]; "dfe" -> DFE mean.
      - Infecteds zero everywhere except at seed_node where they are set to analytic EE I_h/I_v
        (or to min_seed if analytic value is numerically zero).
    """
    # Preconditions: S_h, I_h, S_v, I_v must exist (computed earlier)
    required = ('S_h' in globals() and 'I_h' in globals() and 'S_v' in globals() and 'I_v' in globals())
    if not required:
        raise RuntimeError("Analytic EE vectors S_h, I_h, S_v, I_v must be defined before calling this function.")

    if not (0 <= seed_node < N_nodes):
        raise ValueError(f"seed_node must be in [0, {N_nodes-1}]")

    # Choose uniform susceptible values
    if S_h_uniform_from == "mean":
        S_h_uniform = float(np.mean(S_h))
        S_v_uniform = float(np.mean(S_v))
    elif S_h_uniform_from == "seed":
        S_h_uniform = float(S_h[seed_node])
        S_v_uniform = float(S_v[seed_node])
    elif S_h_uniform_from == "dfe":
        # DFE fallback
        S_h_uniform = float(np.mean(Lambda_h / (mu_h + 1e-12)))
        S_v_uniform = float(np.mean(Lambda_v / (mu_v + delta_v + 1e-12)))
    else:
        raise ValueError("S_h_uniform_from must be one of: 'mean','seed','dfe'")

    # Build arrays
    S_h0 = np.full(N_nodes, S_h_uniform, dtype=np.float32)
    S_v0 = np.full(N_nodes, S_v_uniform, dtype=np.float32)
    I_h0 = np.zeros(N_nodes, dtype=np.float32)
    I_v0 = np.zeros(N_nodes, dtype=np.float32)

    # Seed infected at seed_node using analytic EE (or min_seed if analytic is zero)
    if use_analytic_seed:
        I_h0[seed_node] = float(max(I_h[seed_node], min_seed))
        I_v0[seed_node] = float(max(I_v[seed_node], min_seed))
    else:
        # small numeric seed
        I_h0[seed_node] = float(min_seed)
        I_v0[seed_node] = float(min_seed)

    # Ensure non-negativity
    S_h0 = np.maximum(S_h0, 0.0)
    I_h0 = np.maximum(I_h0, 0.0)
    S_v0 = np.maximum(S_v0, 0.0)
    I_v0 = np.maximum(I_v0, 0.0)

    y0 = np.stack([S_h0, I_h0, S_v0, I_v0], axis=1)  # (N,4)
    return y0.astype(np.float32)

# --- Build seeded initial condition directly from the EE arrays (lines 143-146) ---
# S_h (line 143) and S_v (line 145): uniform EE susceptibles across ALL nodes.
# I_h (line 144) and I_v (line 146): EE infected only at seed_node; zero elsewhere.
seed_node = 0   # change to the node you want to seed

S_h0 = np.full(N_nodes, float(S_h[seed_node]), dtype=np.float32)   # EE S_h, uniform
S_v0 = np.full(N_nodes, float(S_v[seed_node]), dtype=np.float32)   # EE S_v, uniform
I_h0 = np.zeros(N_nodes, dtype=np.float32)                          # I_h = 0 everywhere
I_v0 = np.zeros(N_nodes, dtype=np.float32)                          # I_v = 0 everywhere
I_h0[seed_node] = float(I_h[seed_node])                             # seed EE I_h
I_v0[seed_node] = float(I_v[seed_node])                             # seed EE I_v

y0_full = np.stack([S_h0, I_h0, S_v0, I_v0], axis=1).astype(np.float32)  # (N, 4)

print(f"IC — seed node {seed_node} : S_h={S_h0[seed_node]:.4g}  I_h={I_h0[seed_node]:.4g}  "
      f"S_v={S_v0[seed_node]:.4g}  I_v={I_v0[seed_node]:.4g}")
print(f"IC — non-seed node 1      : S_h={S_h0[1]:.4g}  I_h={I_h0[1]:.4g}  "
      f"S_v={S_v0[1]:.4g}  I_v={I_v0[1]:.4g}")

# ---------------------------
# Training setup
# ---------------------------
torch.manual_seed(0)

t0      = 0.0
T_final = 100.0

model        = PINNNet(N_nodes, hidden=(128, 128), T_final=T_final,
                       y0_anchor=y0_full).to(device)   # y0_full: seeded IC (N, 4)
node_onehots = make_node_onehots(N_nodes, device)
seed_node    = 0

# y0_full already set by analytic steady-state block above

t_colloc = torch.tensor(
    np.linspace(t0, T_final, 250, dtype=np.float32).reshape(-1, 1),
    device=device
)

optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=400, min_lr=1e-6
)

n_epochs    = 10000
print_every = 1000

hist_loss = []
hist_res  = []
hist_ic   = []

print(f"\nTraining PINN on Normalized Random-Walk network  (N={N_nodes}) ...")

# ---------------------------
# NumPy RK4 diagnostic (standalone ODE test)
# ---------------------------
def movement_term_np(u, M):
    # u: (N,) field values
    # M: (N,N) movement rate matrix (M_ij = rate from i to j)
    # row_sum = sum_j M_ij  (equals c for normalized P)
    row_sum = M.sum(axis=1)            # (N,)
    # Correct: (M^T u)_i = sum_j M_{j,i} u_j is computed as u @ M
    return row_sum * u - (u @ M)       # (N,)

def rhs_graph_numpy(y, beta_h, beta_v, Lambda_h, Lambda_v, mu_h, mu_v,
                    gamma, theta, delta_v, M_sh, Sigma_ih, Nu_sv, Nu_iv):
    """
    y: (N,4) state [S_h, I_h, S_v, I_v] per node
    returns dy/dt: (N,4)
    """
    S_h = y[:, 0]
    I_h = y[:, 1]
    S_v = y[:, 2]
    I_v = y[:, 3]

    denom = S_h + I_h + 1e-12
    inf_h = beta_h * (S_h * I_v / denom)
    inf_v = beta_v * (S_v * I_h / denom)

    dS_h_local = Lambda_h - inf_h - mu_h * S_h + gamma * I_h
    dI_h_local = inf_h - (gamma + mu_h + theta) * I_h
    dS_v_local = Lambda_v - inf_v - (mu_v + delta_v) * S_v
    dI_v_local = inf_v - (mu_v + delta_v) * I_v

    dS_h = dS_h_local - movement_term_np(S_h, M_sh)
    dI_h = dI_h_local - movement_term_np(I_h, Sigma_ih)
    dS_v = dS_v_local - movement_term_np(S_v, Nu_sv)
    dI_v = dI_v_local - movement_term_np(I_v, Nu_iv)

    dy = np.stack([dS_h, dI_h, dS_v, dI_v], axis=1)
    return dy

def rk4_step(y, dt, rhs, *args):
    k1 = rhs(y, *args)
    k2 = rhs(y + 0.5*dt*k1, *args)
    k3 = rhs(y + 0.5*dt*k2, *args)
    k4 = rhs(y + dt*k3, *args)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def sweep_movement_and_run(m_vals):
    results = []
    for c_Sh_val, c_Ih_val, c_Sv_val, c_Iv_val in m_vals:
        M_sh_tmp = c_Sh_val * P
        Sigma_ih_tmp = c_Ih_val * P
        Nu_sv_tmp = c_Sv_val * P
        Nu_iv_tmp = c_Iv_val * P

        y = y0_full.copy()
        dt = 0.1
        steps = 300
        Ih_trace = np.zeros(steps+1)
        Ih_trace[0] = y[seed_node,1]
        for k in range(steps):
            y = rk4_step(y, dt, rhs_graph_numpy, beta_h, beta_v, Lambda_h, Lambda_v,
                        mu_h, mu_v, gamma, theta, delta_v, M_sh_tmp, Sigma_ih_tmp, Nu_sv_tmp, Nu_iv_tmp)
            Ih_trace[k+1] = y[seed_node,1]
        results.append({
            "c_Sh": c_Sh_val, "c_Ih": c_Ih_val, "c_Sv": c_Sv_val, "c_Iv": c_Iv_val,
            "Ih_final": Ih_trace[-1], "Ih_max": Ih_trace.max()
        })
    return results

# Example sweep: vary human movement from 0 to 5, keep vectors immobile
m_vals = [
    (0.0, 0.0, 0.0, 0.0),
    (2.5, 2.5, 0.0, 0.0),
    (5.0, 5.0, 0.0, 0.0),
    (10.0, 10.0, 0.0, 0.0),
]
res = sweep_movement_and_run(m_vals)
for r in res:
    print(r)

# Diagnostic run parameters
dt = 0.1
n_steps = 500  # simulate dt * n_steps time units
y = y0_full.copy()   # (N,4) initial condition from analytic block
seed = seed_node

# Print initial derivative at seed node
dy0 = rhs_graph_numpy(y, beta_h, beta_v, Lambda_h, Lambda_v, mu_h, mu_v,
                      gamma, theta, delta_v, M_sh, Sigma_ih, Nu_sv, Nu_iv)
print("\n--- NumPy RK4 diagnostic ---")
print("Initial dI_h/dt at seed node:", float(dy0[seed,1]))

# Short RK4 forward simulation
Ih_trace = np.zeros(n_steps+1)
Ih_trace[0] = y[seed,1]
for k in range(n_steps):
    y = rk4_step(y, dt, rhs_graph_numpy, beta_h, beta_v, Lambda_h, Lambda_v,
                 mu_h, mu_v, gamma, theta, delta_v, M_sh, Sigma_ih, Nu_sv, Nu_iv)
    Ih_trace[k+1] = y[seed,1]

# Report results
print(f"I_h at seed node after {dt*n_steps:.1f} time units: {Ih_trace[-1]:.6e}")
growth_ratio = Ih_trace[-1] / (Ih_trace[0] + 1e-12)
print(f"Growth ratio (I_final / I_initial): {growth_ratio:.3f}")

# Optional quick check: print first 10 values
print("First 10 I_h values at seed node:", Ih_trace[:10])

# ---------------------------
# PINN training loop (active)
# ---------------------------
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()
    loss, metrics = pinn_loss(model, t_colloc, node_onehots, y0_full, var_scales)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.detach())

    hist_loss.append(loss.item())
    hist_res.append(metrics['res_loss'])
    hist_ic.append(metrics['ic_loss'])

    if epoch % print_every == 0 or epoch == 1:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:5d} | Loss {loss.item():.6e} | res {metrics['res_loss']:.3e}"
              f" | ic {metrics['ic_loss']:.3e} | lr {lr_now:.1e}")

# ---------------------------
# Training-curve plot
# ---------------------------
plt.figure(figsize=(9, 4))
plt.semilogy(np.arange(1, n_epochs + 1), hist_loss, label='Total loss',    color='tab:green')
plt.semilogy(np.arange(1, n_epochs + 1), hist_res,  label='Residual loss', color='tab:orange', ls='--')
plt.semilogy(np.arange(1, n_epochs + 1), hist_ic,   label='IC loss',       color='tab:blue',   ls=':')
plt.xlabel('Epoch', fontsize=13)
plt.ylabel('Normalised loss (log scale)', fontsize=13)
plt.title('PINN training — Normalized Random-Walk network', fontsize=14)
plt.legend(fontsize=12, framealpha=0.9)
plt.grid(True, which='both', alpha=0.35)
plt.tight_layout()
plt.show()

# ---------------------------
# Evaluate on fine time grid
# ---------------------------
t_test  = np.linspace(t0, T_final, 10001, dtype=np.float32)
Tt      = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32, device=device)
node_onehots_eval = make_node_onehots(N_nodes, device)

with torch.no_grad():
    preds = []
    for i in range(len(t_test)):
        t_i   = Tt[i].unsqueeze(0).repeat(N_nodes, 1)
        out_i = model(t_i, node_onehots_eval)
        preds.append(out_i.cpu().numpy())
    preds = np.stack(preds, axis=0)   # (T, N, 4)

S_h = preds[:, :, 0];  I_h = preds[:, :, 1]
S_v = preds[:, :, 2];  I_v = preds[:, :, 3]

human_prev  = I_h / (S_h + I_h + 1e-12)
vector_prev = I_v / (S_v + I_v + 1e-12)

# ---------------------------
# Spatio-temporal prevalence heatmaps
# ---------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

vmin_h = float(np.nanmin(human_prev));  vmax_h = float(np.nanmax(human_prev))
im0 = axes[0].imshow(human_prev.T, aspect='auto', origin='lower',
                     extent=[t_test[0], t_test[-1], 0, N_nodes - 1],
                     cmap='viridis', vmin=vmin_h, vmax=vmax_h)
axes[0].set_title("Human prevalence — Normalized Random-Walk network")
axes[0].set_ylabel("Node index")
plt.colorbar(im0, ax=axes[0]).set_label('Human prevalence')

vmin_v = float(np.nanmin(vector_prev));  vmax_v = float(np.nanmax(vector_prev))
im1 = axes[1].imshow(vector_prev.T, aspect='auto', origin='lower',
                     extent=[t_test[0], t_test[-1], 0, N_nodes - 1],
                     cmap='magma', vmin=vmin_v, vmax=vmax_v)
axes[1].set_title("Vector prevalence — Normalized Random-Walk network")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Node index")
plt.colorbar(im1, ax=axes[1]).set_label('Vector prevalence')

plt.tight_layout()
plt.show()

# ---------------------------
# Network snapshots (human prevalence)
# ---------------------------
snap_times   = [0.0, 25.0, 50.0, 100.0]
snap_indices = [int(np.argmin(np.abs(t_test - t))) for t in snap_times]
pos = nx.spring_layout(G, seed=2)

fig = plt.figure(figsize=(14, 4))
for i, idx in enumerate(snap_indices):
    ax = fig.add_subplot(1, len(snap_indices), i + 1)
    node_vals = human_prev[idx]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=200,
                                   node_color=node_vals, cmap='viridis',
                                   vmin=vmin_h, vmax=vmax_h, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
    ax.set_title(f"Human prev  t = {t_test[idx]:.0f}")
    ax.axis('off')
    plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("Human prevalence snapshots — Normalized Random-Walk network")
plt.tight_layout()
plt.show()

# ---------------------------
# Network snapshots (vector prevalence)
# ---------------------------
fig2 = plt.figure(figsize=(14, 4))
for i, idx in enumerate(snap_indices):
    ax = fig2.add_subplot(1, len(snap_indices), i + 1)
    node_vals = vector_prev[idx]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=200,
                                   node_color=node_vals, cmap='magma',
                                   vmin=vmin_v, vmax=vmax_v, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
    ax.set_title(f"Vector prev  t = {t_test[idx]:.0f}")
    ax.axis('off')
    plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
plt.suptitle("Vector prevalence snapshots — Normalized Random-Walk network")
plt.tight_layout()
plt.show()
