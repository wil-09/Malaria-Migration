"""
V6_PINNs_NRW_OptCtrl.py
=======================
Python translation of V6_PINNs_NRW_OptCtrl.m

Malaria Optimal Control — Normalized Random-Walk Two-Patch Network.
Implements a forward-backward sweep (Pontryagin maximum principle) to
find three time-dependent controls u1 (personal protection), u2
(treatment) and u3 (vector control) that minimise a weighted cost
functional combining infections and control effort.

Includes:
  - 8 control-strategy scenarios
  - Parametric bootstrap (N=200, cv=10 %) for 95 % confidence intervals
    on ΔJ, ΔI_h, ICER and Δpeak for both patches
  - Tabular console output mirroring the MATLAB script
  - Two publication-quality matplotlib figures

Requirements: numpy, scipy (for np.trapz alias only), matplotlib.
"""

from __future__ import annotations
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from types import SimpleNamespace

matplotlib.rcParams.update({"font.size": 9})


# =========================================================================
# Helper: format a confidence interval as "[lo,hi]" in WIDTH characters
# =========================================================================

def fmtci(lo: float, hi: float, width: int = 9) -> str:
    """Return '[lo,hi]' right-justified to *width* characters."""
    for nd in (3, 2, 1):
        fmt = f"[{{:.{nd}g}},{{:.{nd}g}}]"
        raw = fmt.format(lo, hi)
        if len(raw) <= width:
            break
    return raw.rjust(width)


# =========================================================================
# Model right-hand side
# =========================================================================

def state_rhs(y: np.ndarray, u: np.ndarray,
              p: SimpleNamespace, active: np.ndarray) -> np.ndarray:
    """Evaluate the 8-component state ODE RHS at a single time point."""
    Sh1 = max(y[0], 0.0);  Ih1 = max(y[1], 0.0)
    Sv1 = max(y[2], 0.0);  Iv1 = max(y[3], 0.0)
    Sh2 = max(y[4], 0.0);  Ih2 = max(y[5], 0.0)
    Sv2 = max(y[6], 0.0);  Iv2 = max(y[7], 0.0)

    u1 = u[0] * active[0]
    u2 = u[1] * active[1]
    u3 = u[2] * active[2]

    Nh1 = Sh1 + Ih1 + 1e-12
    infh1 = (1 - u1) * p.beta_h1 * Sh1 * Iv1 / Nh1
    infv1 = p.beta_v1 * Sv1 * Ih1 / Nh1
    d1 = p.gamma + p.mu_h + p.theta + p.sigma_12
    mu1 = p.mu_v + p.delta_v + p.nu_12

    Nh2 = Sh2 + Ih2 + 1e-12
    infh2 = p.beta_h2 * Sh2 * Iv2 / Nh2
    infv2 = p.beta_v2 * Sv2 * Ih2 / Nh2
    d2 = p.gamma + p.mu_h + p.theta + p.sigma_21
    mu2 = p.mu_v + p.nu_21

    dSh1 = p.Lambda_h1 - infh1 - (p.mu_h + p.m_12) * Sh1 + (p.gamma + u2) * Ih1 + p.m_21 * Sh2
    dIh1 = infh1 - (d1 + u2) * Ih1 + p.sigma_21 * Ih2
    dSv1 = p.Lambda_v1 - infv1 - (mu1 + u3) * Sv1 + p.nu_21 * Sv2
    dIv1 = infv1 - (mu1 + u3) * Iv1 + p.nu_21 * Iv2

    dSh2 = p.Lambda_h2 - infh2 - (p.mu_h + p.m_21) * Sh2 + p.gamma * Ih2 + p.m_12 * Sh1
    dIh2 = infh2 - d2 * Ih2 + p.sigma_12 * Ih1
    dSv2 = p.Lambda_v2 - infv2 - mu2 * Sv2 + p.nu_12 * Sv1
    dIv2 = infv2 - mu2 * Iv2 + p.nu_12 * Iv1

    return np.array([dSh1, dIh1, dSv1, dIv1, dSh2, dIh2, dSv2, dIv2])


# =========================================================================
# Adjoint right-hand side
# =========================================================================

def adjoint_rhs(lam: np.ndarray, y: np.ndarray, u: np.ndarray,
                p: SimpleNamespace, active: np.ndarray) -> np.ndarray:
    """Evaluate the 8-component adjoint ODE RHS at a single time point."""
    Sh1 = max(y[0], 0.0);  Ih1 = max(y[1], 0.0)
    Sv1 = max(y[2], 0.0);  Iv1 = max(y[3], 0.0)
    Sh2 = max(y[4], 0.0);  Ih2 = max(y[5], 0.0)
    Sv2 = max(y[6], 0.0);  Iv2 = max(y[7], 0.0)

    u1 = u[0] * active[0]
    u2 = u[1] * active[1]
    u3 = u[2] * active[2]

    l1, l2, l3, l4 = lam[0], lam[1], lam[2], lam[3]
    l5, l6, l7, l8 = lam[4], lam[5], lam[6], lam[7]

    Nh1 = Sh1 + Ih1 + 1e-12
    Nh2 = Sh2 + Ih2 + 1e-12
    d1 = p.gamma + p.mu_h + p.theta + p.sigma_12
    d2 = p.gamma + p.mu_h + p.theta + p.sigma_21
    mu1 = p.mu_v + p.delta_v + p.nu_12
    mu2 = p.mu_v + p.nu_21

    dlam1 = (l1 * (p.mu_h + p.m_12) - p.m_12 * l5
             + ((1 - u1) * p.beta_h1 * Iv1 * (l2 - l1)
                + (1 - u1) * p.beta_v1 * Ih1 * (l4 - l3)) / Nh1**2)

    dlam2 = (-p.A1 + l2 * (d1 + u2) - p.sigma_21 * l6
             + ((1 - u1) * p.beta_h1 * Sh1 * Iv1 * (l2 - l1)
                + (1 - u1) * p.beta_v1 * Sv1 * Sh1 * (l4 - l3)) / Nh1**2)

    dlam3 = (-p.A2 + l3 * (mu1 + u3) - p.nu_12 * l7
             - p.beta_v1 * Ih1 * (1 - u1) * (l4 - l3) / Nh1)

    dlam4 = (-p.A2 + l4 * (mu1 + u3) - p.nu_12 * l8
             - p.beta_h1 * Sh1 * (1 - u1) * (l2 - l1) / Nh1)

    dlam5 = (-p.m_21 * l1 + l5 * (p.mu_h + p.m_21)
             + (p.beta_h2 * Iv2 * (l6 - l5)
                + p.beta_v2 * Ih2 * (l8 - l7)) / Nh2**2)

    dlam6 = (-p.A1 + l6 * d2 - p.sigma_12 * l2
             + (p.beta_h2 * Sh2 * Iv2 * (l6 - l5)
                + p.beta_v2 * Sv2 * Sh2 * (l8 - l7)) / Nh2**2)

    dlam7 = (-p.A2 + l7 * mu2 - p.nu_21 * l3
             - p.beta_v2 * Ih2 * (l8 - l7) / Nh2)

    dlam8 = (-p.A2 + l8 * mu2 - p.nu_21 * l4
             - p.beta_h2 * Sh2 * (l6 - l5) / Nh2)

    return np.array([dlam1, dlam2, dlam3, dlam4, dlam5, dlam6, dlam7, dlam8])


# =========================================================================
# Optimal controls (Pontryagin optimality conditions)
# =========================================================================

def optimal_controls(Y: np.ndarray, LAM: np.ndarray,
                     p: SimpleNamespace, active: np.ndarray) -> np.ndarray:
    """Compute optimal controls from state and adjoint arrays (N × 3)."""
    N = Y.shape[0]
    U = np.zeros((N, 3))

    Sh1 = np.maximum(Y[:, 0], 0.0);  Ih1 = np.maximum(Y[:, 1], 0.0)
    Sv1 = np.maximum(Y[:, 2], 0.0);  Iv1 = np.maximum(Y[:, 3], 0.0)
    Sh2 = np.maximum(Y[:, 4], 0.0);  Ih2 = np.maximum(Y[:, 5], 0.0)
    Sv2 = np.maximum(Y[:, 6], 0.0);  Iv2 = np.maximum(Y[:, 7], 0.0)

    l1 = LAM[:, 0];  l2 = LAM[:, 1];  l3 = LAM[:, 2];  l4 = LAM[:, 3]
    l5 = LAM[:, 4];  l6 = LAM[:, 5];  l7 = LAM[:, 6];  l8 = LAM[:, 7]

    Nh1 = Sh1 + Ih1 + 1e-12

    if active[0]:
        num1 = ((l2 - l1) * p.beta_h1 * Sh1 * Iv1
                + (l4 - l3) * p.beta_v1 * Sv1 * Ih1)
        U[:, 0] = np.clip(num1 / (p.B1 * Nh1), 0.0, p.u1max)

    if active[1]:
        U[:, 1] = np.clip((l2 - l1) * Ih1 / p.B2, 0.0, p.u2max)

    if active[2]:
        U[:, 2] = np.clip((l3 * Sv1 + l4 * Iv1) / p.B3, 0.0, p.u3max)

    return U


# =========================================================================
# RK4 forward integration
# =========================================================================

def rk4_forward(y0: np.ndarray, U: np.ndarray, t: np.ndarray,
                p: SimpleNamespace, active: np.ndarray) -> np.ndarray:
    """Integrate the state equations forward in time using RK4."""
    N = len(t)
    Y = np.zeros((N, 8))
    Y[0] = y0
    for k in range(N - 1):
        h = t[k + 1] - t[k]
        yk = Y[k]
        uk = U[k];   uk1 = U[k + 1];   um = 0.5 * (uk + uk1)
        K1 = state_rhs(yk,            uk,  p, active)
        K2 = state_rhs(yk + h / 2 * K1, um,  p, active)
        K3 = state_rhs(yk + h / 2 * K2, um,  p, active)
        K4 = state_rhs(yk + h * K3,     uk1, p, active)
        Y[k + 1] = np.maximum(yk + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4), 0.0)
    return Y


# =========================================================================
# RK4 backward integration (adjoint)
# =========================================================================

def rk4_backward(Y: np.ndarray, U: np.ndarray, t: np.ndarray,
                 p: SimpleNamespace, active: np.ndarray) -> np.ndarray:
    """Integrate the adjoint equations backward in time using RK4."""
    N = len(t)
    LAM = np.zeros((N, 8))
    for k in range(N - 2, -1, -1):
        h = t[k + 1] - t[k]
        yk = Y[k];   yk1 = Y[k + 1];   ym = 0.5 * (yk + yk1)
        uk = U[k];   uk1 = U[k + 1];   um = 0.5 * (uk + uk1)
        lk1 = LAM[k + 1]
        K1 = -adjoint_rhs(lk1,             yk1, uk1, p, active)
        K2 = -adjoint_rhs(lk1 + h / 2 * K1, ym,  um,  p, active)
        K3 = -adjoint_rhs(lk1 + h / 2 * K2, ym,  um,  p, active)
        K4 = -adjoint_rhs(lk1 + h * K3,     yk,  uk,  p, active)
        LAM[k] = lk1 + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return LAM


# =========================================================================
# Forward-backward sweep (iterative optimal control solver)
# =========================================================================

def fwd_bwd_sweep(y0: np.ndarray, t: np.ndarray,
                  p: SimpleNamespace, active: np.ndarray,
                  tol: float = 1e-5, maxiter: int = 300):
    """Solve the optimal control problem via forward-backward sweep.

    Returns
    -------
    U_opt : ndarray, shape (N, 3)
    Y_opt : ndarray, shape (N, 8)
    """
    N = len(t)
    U = np.zeros((N, 3))

    for _ in range(maxiter):
        U_old = U.copy()
        Y = rk4_forward(y0, U, t, p, active)
        LAM = rk4_backward(Y, U, t, p, active)
        U_new = optimal_controls(Y, LAM, p, active)
        U = 0.5 * (U_old + U_new)
        if np.max(np.abs(U - U_old)) < tol:
            break

    Y_opt = rk4_forward(y0, U, t, p, active)
    return U, Y_opt


# =========================================================================
# Basic reproduction number R0
# =========================================================================

def compute_R0(p: SimpleNamespace) -> float:
    """Compute R0 from parameter namespace p."""
    EPS = sys.float_info.epsilon

    sigma_12 = p.sigma_12;  sigma_21 = p.sigma_21
    nu_12 = p.nu_12;        nu_21 = p.nu_21
    mu_h = p.mu_h;          mu_v = p.mu_v
    delta_v = p.delta_v;    theta = p.theta;  gamma = p.gamma

    Sh1 = p.Sh1_star;  Sh2 = p.Sh2_star
    Sv1 = p.Sv1_star;  Sv2 = p.Sv2_star

    d1 = mu_h + theta + gamma + sigma_12
    d2 = mu_h + theta + gamma + sigma_21
    mu1 = mu_v + delta_v + nu_12
    mu2 = mu_v + nu_21

    den2 = Sh2 * (d1 * d2 - sigma_12 * sigma_21) + EPS

    K_23 = p.beta_v2 * Sv2 / den2
    K_41 = p.beta_h2 / (mu1 * mu2 - nu_12 * nu_21 + EPS)
    Psi = (p.beta_v1 * Sv1 * Sh2) / (p.beta_v2 * Sh1 * Sv2 + EPS)
    tau = p.beta_h1 / p.beta_h2

    A = (Psi * (tau * d2 * mu2 + nu_12 * sigma_21)
         + tau * nu_21 * sigma_12 + d1 * mu1)

    sqrt_arg = max(
        1.0 - (4 * Psi * tau
               * (d1 * d2 - sigma_12 * sigma_21)
               * (mu1 * mu2 - nu_12 * nu_21)) / (A**2 + EPS),
        0.0,
    )

    return math.sqrt(0.5 * (K_41 * K_23 * A) * (1.0 + math.sqrt(sqrt_arg)))


# =========================================================================
# Compute DFE susceptible equilibria
# =========================================================================

def compute_equilibria(p: SimpleNamespace) -> None:
    """Update p.Sh1_star, Sh2_star, Sv1_star, Sv2_star in-place."""
    dh = p.mu_h * (p.mu_h + p.m_12 + p.m_21)
    p.Sh1_star = (p.Lambda_h1 * (p.mu_h + p.m_21) + p.m_21 * p.Lambda_h2) / dh
    p.Sh2_star = (p.Lambda_h2 * (p.mu_h + p.m_12) + p.m_12 * p.Lambda_h1) / dh
    dv = p.mu_v * (p.mu_v + p.delta_v + p.nu_12 + p.nu_21) + p.nu_21 * p.delta_v
    p.Sv1_star = (p.Lambda_v1 * (p.mu_v + p.nu_21) + p.nu_21 * p.Lambda_v2) / dv
    p.Sv2_star = (p.Lambda_v2 * (p.mu_v + p.nu_12) + p.nu_12 * p.Lambda_v1) / dv


# =========================================================================
# ICER computation (used for both patches)
# =========================================================================

def compute_icer(res_J: np.ndarray, res_AC: np.ndarray, n_scen: int):
    """Return (AC_srt, J_srt, ord_idx, ICER_val, ICER_flag, best_k, dominated)."""
    ord_idx = np.argsort(res_AC)
    AC_srt = res_AC[ord_idx]
    J_srt = res_J[ord_idx]

    dominated = np.zeros(n_scen, dtype=bool)
    for i in range(n_scen):
        for j in range(n_scen):
            if i != j and not dominated[j]:
                if AC_srt[j] >= AC_srt[i] and J_srt[j] < J_srt[i]:
                    dominated[i] = True
                    break

    ICER_val = np.full(n_scen, np.nan)
    ICER_flag = ["   "] * n_scen
    for k in range(1, n_scen):
        if dominated[k]:
            ICER_flag[k] = "DOM"
            continue
        prev_candidates = [ii for ii in range(k) if not dominated[ii]]
        if not prev_candidates:
            prev_J = J_srt[0];   prev_AC = AC_srt[0]
        else:
            prev_J = J_srt[prev_candidates[-1]]
            prev_AC = AC_srt[prev_candidates[-1]]
        dAC = AC_srt[k] - prev_AC
        dJ = J_srt[k] - prev_J
        if dAC < 1e-10:
            ICER_flag[k] = "DOM";  dominated[k] = True
        else:
            ICER_val[k] = dJ / dAC
            if dJ < 0:
                ICER_flag[k] = "DOMINANT"

    dom_set = [k for k in range(n_scen)
               if not dominated[k] and not np.isnan(ICER_val[k]) and ICER_val[k] < 0]
    if dom_set:
        tmp = int(np.argmax(AC_srt[dom_set]))
        best_k = dom_set[tmp]
    else:
        front = [k for k in range(n_scen)
                 if not dominated[k] and not np.isnan(ICER_val[k]) and ICER_val[k] >= 0]
        if not front:
            front = [k for k in range(n_scen) if not dominated[k]]
        tmp = int(np.argmin(ICER_val[front]))
        best_k = front[tmp]

    ICER_flag[best_k] = "*** BEST ***"
    return AC_srt, J_srt, ord_idx, ICER_val, ICER_flag, best_k, dominated


# =========================================================================
# Bootstrap ICER CIs (fixed pairing)
# =========================================================================

def bootstrap_icer_ci(boot_J: np.ndarray, boot_AC: np.ndarray,
                      ord_idx: np.ndarray, dominated: np.ndarray,
                      n_s: int, plo: float, phi: float) -> np.ndarray:
    """Return (n_s × 2) CI array [lo, hi] for ICER values."""
    N_boot = boot_J.shape[0]
    boot_ICER = np.full((N_boot, n_s), np.nan)
    for b in range(N_boot):
        Jbs = boot_J[b, ord_idx]
        ACbs = boot_AC[b, ord_idx]
        for k in range(1, n_s):
            if dominated[k]:
                continue
            prev_candidates = [ii for ii in range(k) if not dominated[ii]]
            if not prev_candidates:
                pvJ = Jbs[0];  pvAC = ACbs[0]
            else:
                pvJ = Jbs[prev_candidates[-1]]
                pvAC = ACbs[prev_candidates[-1]]
            dAC = ACbs[k] - pvAC
            if abs(dAC) > 1e-10:
                boot_ICER[b, k] = (Jbs[k] - pvJ) / dAC
    return np.column_stack([
        np.nanpercentile(boot_ICER, plo, axis=0),
        np.nanpercentile(boot_ICER, phi, axis=0),
    ])


# =========================================================================
# Print ICER table
# =========================================================================

def print_icer_table(title: str,
                     lbl_srt: list[str],
                     J_srt: np.ndarray,
                     AC_srt: np.ndarray,
                     ICER_val: np.ndarray,
                     ICER_flag: list[str],
                     best_k: int,
                     dominated: np.ndarray,
                     ci_J_sorted: np.ndarray,
                     ci_AC_sorted: np.ndarray,
                     ci_ICER: np.ndarray,
                     ac_header: str = "Inf. Averted") -> None:
    sep = "=" * 84
    row = "-" * 84
    print(f"\n{sep}")
    print(f" {title}")
    print(f" ICER(B vs A) = [J(B) - J(A)] / [AC(B) - AC(A)]")
    print(sep)
    print(f"{'Strategy':<18} | {'Cost J':>10} | {ac_header:>12} | {'ICER':>15} | {'Status':<13}")
    print(row)
    for k, lbl in enumerate(lbl_srt):
        iv = ICER_val[k]
        icer_str = f"{'—':>15}" if np.isnan(iv) else f"{iv:>15.4f}"
        marker = "  <<<" if ICER_flag[k] == "*** BEST ***" else ""
        print(f"{lbl:<18} | {J_srt[k]:>10.4f} | {AC_srt[k]:>12.4f} | {icer_str} | {ICER_flag[k]:<13}{marker}")
        # 95 % CI sub-row
        os = k  # already sorted index; pass pre-sorted CI arrays
        if k == 0 or np.all(np.isnan(ci_ICER[k])):
            ici_str = f"{'—':>15}"
        elif dominated[k]:
            ici_str = f"{'DOM':>15}"
        else:
            ici_str = f"{('[' + f'{ci_ICER[k,0]:.3g}' + ',' + f'{ci_ICER[k,1]:.3g}' + ']'):>15}"
        print(f"  {'[95% CI]':<16} | {fmtci(ci_J_sorted[k,0], ci_J_sorted[k,1], 10)} "
              f"| {fmtci(ci_AC_sorted[k,0], ci_AC_sorted[k,1], 12)} | {ici_str} | {'':13}")
    print(sep)
    print("\nLegend:")
    print("  DOMINANT  : cheaper AND averts more cases than its comparator (ICER<0)")
    print("  DOM       : dominated — costlier and less effective than another strategy")
    print("  *** BEST  : most cost-effective strategy on the efficiency frontier\n")


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:
    print("=========================================================")
    print("  Malaria Optimal Control — Normalized Random-Walk Net  ")
    print("=========================================================\n")

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------
    p = SimpleNamespace()

    p.epsilon  = 0.8
    p.theta    = 9e-5
    p.gamma    = 5.5e-4
    p.Lambda_h = 0.033
    p.Lambda_v = 0.13
    p.mu_h     = 1.6e-5
    p.beta_h   = 0.022
    p.beta_v   = 0.48
    p.mu_v     = 0.033
    p.delta_v  = 0.25

    p.Lambda_h1 = p.Lambda_h
    p.Lambda_h2 = p.Lambda_h
    p.Lambda_v1 = p.Lambda_v * (1 - p.epsilon)
    p.Lambda_v2 = p.Lambda_v
    p.beta_h1   = p.beta_h * (1 - p.epsilon)
    p.beta_h2   = p.beta_h
    p.beta_v1   = p.beta_v * (1 - p.epsilon)
    p.beta_v2   = p.beta_v

    p.m_12     = 0.001;   p.m_21     = 0.1
    p.sigma_12 = 0.9;     p.sigma_21 = 0.1
    p.nu_12    = 0.0;     p.nu_21    = 0.0

    compute_equilibria(p)

    # Cost-function weights
    p.A1 = 30;   p.A2 = 10
    p.B1 = 50;   p.B2 = 20;   p.B3 = 20

    # Control bounds
    p.u1max = 1.0;  p.u2max = 1.0;  p.u3max = 1.0

    R0 = compute_R0(p)
    print(f"DFE:  S_h1* = {p.Sh1_star:.4f}   S_v1* = {p.Sv1_star:.4f}")
    epi = "(> 1 → epidemic)" if R0 > 1 else "(< 1 → extinction)"
    print(f"R0   = {R0:.4f}  {epi}\n")

    # ------------------------------------------------------------------
    # Time grid & initial conditions
    # ------------------------------------------------------------------
    T_fin = 1000
    N     = 10001
    t     = np.linspace(0, T_fin, N)

    y0 = np.array([p.Sh1_star, 0.001, p.Sv1_star, 0.001,
                   p.Sh2_star, 0.1,   p.Sv2_star, 0.1])

    # ------------------------------------------------------------------
    # Baseline (no control)
    # ------------------------------------------------------------------
    U_zero  = np.zeros((N, 3))
    Y_base  = rk4_forward(y0, U_zero, t, p, np.zeros(3, dtype=bool))
    Ih_base    = Y_base[:, 1]
    Ih_base_p2 = Y_base[:, 5]

    # ------------------------------------------------------------------
    # 8 control strategies
    # ------------------------------------------------------------------
    active_mat = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ], dtype=bool)

    labels = ["No control", "u_1 only", "u_2 only", "u_3 only",
              "u_1+u_2", "u_1+u_3", "u_2+u_3", "u_1+u_2+u_3"]
    n_scen = len(labels)

    res_J       = np.zeros(n_scen)
    res_AC      = np.zeros(n_scen)
    res_AC2     = np.zeros(n_scen)
    res_peakIh  = np.zeros(n_scen)
    res_peakIh2 = np.zeros(n_scen)
    res_intU    = np.zeros((n_scen, 3))

    Y_store: list[np.ndarray] = [None] * n_scen
    U_store: list[np.ndarray] = [None] * n_scen

    for s in range(n_scen):
        act = active_mat[s]
        print(f"Running scenario {s+1}/{n_scen}:  {labels[s]:<18}  ...", end="", flush=True)

        if not np.any(act):
            U_opt = U_zero.copy()
            Y_opt = Y_base.copy()
        else:
            U_opt, Y_opt = fwd_bwd_sweep(y0, t, p, act)

        Y_store[s] = Y_opt
        U_store[s] = U_opt

        Ih1 = Y_opt[:, 1];  Nv1 = Y_opt[:, 2] + Y_opt[:, 3]
        Ih2 = Y_opt[:, 5]
        u1 = U_opt[:, 0];  u2 = U_opt[:, 1];  u3 = U_opt[:, 2]

        res_J[s]       = np.trapz(p.A1 * Ih1 + p.A2 * Nv1
                                  + 0.5 * p.B1 * u1**2 + 0.5 * p.B2 * u2**2
                                  + 0.5 * p.B3 * u3**2, t)
        res_AC[s]      = np.trapz(Ih_base    - Ih1, t)
        res_AC2[s]     = np.trapz(Ih_base_p2 - Ih2, t)
        res_peakIh[s]  = float(np.max(Ih1))
        res_peakIh2[s] = float(np.max(Ih2))
        res_intU[s, 0] = np.trapz(u1, t)
        res_intU[s, 1] = np.trapz(u2, t)
        res_intU[s, 2] = np.trapz(u3, t)

        print(f"  J = {res_J[s]:9.4f}   AC_P1 = {res_AC[s]:9.4f}   AC_P2 = {res_AC2[s]:9.4f}")

    # ==================================================================
    # PARAMETRIC BOOTSTRAP — 95 % Confidence Intervals
    # ==================================================================
    N_boot  = 200
    cv_boot = 0.10
    plo     = 2.5
    phi_pct = 97.5

    pf_boot = ["beta_h1", "beta_h2", "beta_v1", "beta_v2",
               "Lambda_h1", "Lambda_h2", "Lambda_v1", "Lambda_v2",
               "mu_h", "mu_v", "gamma", "theta", "delta_v"]

    boot_J         = np.zeros((N_boot, n_scen))
    boot_AC        = np.zeros((N_boot, n_scen))
    boot_AC2       = np.zeros((N_boot, n_scen))
    boot_peakIh    = np.zeros((N_boot, n_scen))
    boot_peakIh2   = np.zeros((N_boot, n_scen))
    boot_peakBase  = np.zeros(N_boot)
    boot_peakBase2 = np.zeros(N_boot)
    boot_intBase   = np.zeros(N_boot)
    boot_intBase2  = np.zeros(N_boot)

    print(f"\nParametric bootstrap ({N_boot} replicates, cv = {100*cv_boot:.0f}%) ...")
    rng = np.random.default_rng(42)

    for b in range(N_boot):
        pb = SimpleNamespace(**vars(p))
        for f in pf_boot:
            setattr(pb, f, getattr(p, f) * math.exp(cv_boot * rng.standard_normal()))
        compute_equilibria(pb)
        y0_b = np.array([pb.Sh1_star, 0.001, pb.Sv1_star, 0.001,
                         pb.Sh2_star, 0.1,   pb.Sv2_star, 0.1])

        Yb0   = rk4_forward(y0_b, U_zero, t, pb, np.zeros(3, dtype=bool))
        Ihb0  = Yb0[:, 1];  Ih2b0 = Yb0[:, 5]
        boot_peakBase[b]  = float(np.max(Ihb0))
        boot_peakBase2[b] = float(np.max(Ih2b0))
        boot_intBase[b]   = float(np.trapz(Ihb0, t))
        boot_intBase2[b]  = float(np.trapz(Ih2b0, t))

        for s in range(n_scen):
            Us  = U_store[s]
            Ys  = rk4_forward(y0_b, Us, t, pb, active_mat[s])
            Ih1s = Ys[:, 1];  Ih2s = Ys[:, 5]
            Nv1s = Ys[:, 2] + Ys[:, 3]
            boot_J[b, s] = np.trapz(
                pb.A1 * Ih1s + pb.A2 * Nv1s
                + 0.5 * pb.B1 * Us[:, 0]**2
                + 0.5 * pb.B2 * Us[:, 1]**2
                + 0.5 * pb.B3 * Us[:, 2]**2, t)
            boot_AC[b, s]      = float(np.trapz(Ihb0  - Ih1s, t))
            boot_AC2[b, s]     = float(np.trapz(Ih2b0 - Ih2s, t))
            boot_peakIh[b, s]  = float(np.max(Ih1s))
            boot_peakIh2[b, s] = float(np.max(Ih2s))

        if (b + 1) % 50 == 0:
            print(f"  Replicate {b+1:3d} / {N_boot}")

    # Derived bootstrap statistics
    boot_deltaJ    = boot_J[:, 0:1] - boot_J
    boot_pct_J     = 100 * boot_deltaJ / (boot_J[:, 0:1] + 1e-12)
    boot_dPeak     = boot_peakBase[:, None]  - boot_peakIh
    boot_dPeak2    = boot_peakBase2[:, None] - boot_peakIh2
    boot_pct_peak  = 100 * boot_dPeak  / (boot_peakBase[:, None]  + 1e-12)
    boot_pct_peak2 = 100 * boot_dPeak2 / (boot_peakBase2[:, None] + 1e-12)
    boot_pct_AC    = 100 * boot_AC  / (boot_intBase[:, None]  + 1e-12)
    boot_pct_AC2   = 100 * boot_AC2 / (boot_intBase2[:, None] + 1e-12)

    def _ci(arr):
        return np.column_stack([np.percentile(arr, plo, axis=0),
                                np.percentile(arr, phi_pct, axis=0)])

    ci_J       = _ci(boot_J)
    ci_deltaJ  = _ci(boot_deltaJ)
    ci_pct_J   = _ci(boot_pct_J)
    ci_AC      = _ci(boot_AC)
    ci_AC2     = _ci(boot_AC2)
    ci_pct_AC  = _ci(boot_pct_AC)
    ci_pct_AC2 = _ci(boot_pct_AC2)
    ci_peak    = _ci(boot_peakIh)
    ci_peak2   = _ci(boot_peakIh2)
    ci_pPeak   = _ci(boot_pct_peak)
    ci_pPeak2  = _ci(boot_pct_peak2)

    print("Bootstrap replicates complete.\n")

    # ==================================================================
    # TABLE 1 — Cost, averted cases, peak, control efforts
    # ==================================================================
    hdr_sep = "=" * 122
    row_sep = "-" * 122
    print()
    print(hdr_sep)
    print(f" Optimal Control Results  (T_fin={T_fin}, R0={R0:.3f})"
          f"  [95% CI (bootstrap N={N_boot}, cv={100*cv_boot:.0f}%) in sub-rows]")
    print(hdr_sep)
    print(f"{'Strategy':<18} | {'Cost J':>9} | {'Avrt P1':>9} | {'Avrt P2':>9} | "
          f"{'Peak P1':>9} | {'Peak P2':>9} | {'int(u_1)':>9} | {'int(u_2)':>9} | "
          f"{'int(u_3)':>9} | {'% reduc.':>8}")
    print(row_sep)
    for s in range(n_scen):
        pct = 100 * (res_J[0] - res_J[s]) / res_J[0]
        print(f"{labels[s]:<18} | {res_J[s]:9.4f} | {res_AC[s]:9.4f} | {res_AC2[s]:9.4f} | "
              f"{res_peakIh[s]:9.4f} | {res_peakIh2[s]:9.4f} | "
              f"{res_intU[s,0]:9.4f} | {res_intU[s,1]:9.4f} | {res_intU[s,2]:9.4f} | "
              f"{pct:7.2f}%")
        pct_ci = f"[{ci_pct_J[s,0]:.0f},{ci_pct_J[s,1]:.0f}]%"
        print(f"  {'[95% CI]':<16} | {fmtci(ci_J[s,0],ci_J[s,1])} | "
              f"{fmtci(ci_AC[s,0],ci_AC[s,1])} | {fmtci(ci_AC2[s,0],ci_AC2[s,1])} | "
              f"{fmtci(ci_peak[s,0],ci_peak[s,1])} | {fmtci(ci_peak2[s,0],ci_peak2[s,1])} | "
              f"{'':>9} | {'':>9} | {'':>9} | {pct_ci:>9}")
    print(hdr_sep)

    # ==================================================================
    # TABLE 2 — Averted cases & peak reduction
    # ==================================================================
    sep2 = "-" * 104
    print()
    print(sep2)
    print(" Averted Cases & Peak I_h Reduction  (vs no-control baseline) [95% CI in sub-rows]")
    print(sep2)
    print(f"{'Strategy':<18} | {'Avrt int(P1)':>12} | {'% avrt P1':>9} | {'% peak P1':>10} | "
          f"{'Avrt int(P2)':>12} | {'% avrt P2':>9} | {'% peak P2':>10}")
    print(sep2)
    int_Ih_base     = float(np.trapz(Ih_base, t))
    peak_Ih_base    = float(np.max(Ih_base))
    int_Ih_base_p2  = float(np.trapz(Ih_base_p2, t))
    peak_Ih_base_p2 = float(np.max(Ih_base_p2))
    for s in range(1, n_scen):
        pct_ac    = 100 * res_AC[s]  / (int_Ih_base    + 1e-12)
        pct_peak  = 100 * (peak_Ih_base   - res_peakIh[s])  / (peak_Ih_base   + 1e-12)
        pct_ac2   = 100 * res_AC2[s] / (int_Ih_base_p2 + 1e-12)
        pct_peak2 = 100 * (peak_Ih_base_p2 - res_peakIh2[s]) / (peak_Ih_base_p2 + 1e-12)
        print(f"{labels[s]:<18} | {res_AC[s]:12.4f} | {pct_ac:8.2f}% | {pct_peak:9.2f}% | "
              f"{res_AC2[s]:12.4f} | {pct_ac2:8.2f}% | {pct_peak2:9.2f}%")
        print(f"  {'[95% CI]':<16} | "
              f"{fmtci(ci_AC[s,0],ci_AC[s,1],12)} | "
              f"[{ci_pct_AC[s,0]:.0f},{ci_pct_AC[s,1]:.0f}]% | "
              f"[{ci_pPeak[s,0]:.0f},{ci_pPeak[s,1]:.0f}]% | "
              f"{fmtci(ci_AC2[s,0],ci_AC2[s,1],12)} | "
              f"[{ci_pct_AC2[s,0]:.0f},{ci_pct_AC2[s,1]:.0f}]% | "
              f"[{ci_pPeak2[s,0]:.0f},{ci_pPeak2[s,1]:.0f}]%")
    print(sep2)

    # ==================================================================
    # ICER — Patch 1
    # ==================================================================
    AC_srt, J_srt, ord_idx, ICER_val, ICER_flag, best_k, dominated = \
        compute_icer(res_J, res_AC, n_scen)
    lbl_srt = [labels[i] for i in ord_idx]
    best_orig_idx = int(ord_idx[best_k])

    ci_ICER = bootstrap_icer_ci(boot_J, boot_AC, ord_idx, dominated, n_scen, plo, phi_pct)
    ci_J_srt  = ci_J[ord_idx]
    ci_AC_srt = ci_AC[ord_idx]

    print_icer_table(
        "INCREMENTAL COST-EFFECTIVENESS RATIO (ICER) TABLE",
        lbl_srt, J_srt, AC_srt, ICER_val, ICER_flag, best_k, dominated,
        ci_J_srt, ci_AC_srt, ci_ICER, "Inf. Averted")
    print(f"Most cost-effective strategy: {lbl_srt[best_k]}\n")

    # ==================================================================
    # ICER — Patch 2
    # ==================================================================
    AC2_srt, J2_srt, ord2_idx, ICER2_val, ICER2_flag, best_k2, dominated2 = \
        compute_icer(res_J, res_AC2, n_scen)
    lbl2_srt = [labels[i] for i in ord2_idx]

    ci_ICER2  = bootstrap_icer_ci(boot_J, boot_AC2, ord2_idx, dominated2, n_scen, plo, phi_pct)
    ci_J_srt2  = ci_J[ord2_idx]
    ci_AC2_srt = ci_AC2[ord2_idx]

    print_icer_table(
        "INCREMENTAL COST-EFFECTIVENESS RATIO (ICER) TABLE — PATCH 2",
        lbl2_srt, J2_srt, AC2_srt, ICER2_val, ICER2_flag, best_k2, dominated2,
        ci_J_srt2, ci_AC2_srt, ci_ICER2, "Avrt P2")
    print(f"Most cost-effective strategy (patch 2): {lbl2_srt[best_k2]}\n")

    # ==================================================================
    # FIGURE 1 — Overview
    # ==================================================================
    cmap = plt.cm.get_cmap("tab10", n_scen)
    clr  = [cmap(i) for i in range(n_scen)]
    lsty = ["-", "-.", ":", "--", "-", "-.", ":", "--"]
    lw   = 1.6

    fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig1.canvas.manager.set_window_title("Optimal Control Analysis")

    # (1) I_h all strategies
    ax = axes[0, 0]
    for s in range(n_scen):
        ax.plot(t, Y_store[s][:, 1], lsty[s], color=clr[s], lw=lw, label=labels[s])
    ax.set_xlabel("Time");  ax.set_ylabel("I_h(t)")
    ax.set_title("Infected Humans — all strategies")
    ax.legend(loc="upper right", fontsize=7);  ax.grid(True)

    # (2) I_v all strategies
    ax = axes[0, 1]
    for s in range(n_scen):
        ax.plot(t, Y_store[s][:, 3], lsty[s], color=clr[s], lw=lw, label=labels[s])
    ax.set_xlabel("Time");  ax.set_ylabel("I_v(t)")
    ax.set_title("Infected Vectors — all strategies")
    ax.legend(loc="upper right", fontsize=7);  ax.grid(True)

    # (3) Optimal control profiles (all-control scenario)
    ax = axes[0, 2]
    s_all = n_scen - 1
    ax.plot(t, U_store[s_all][:, 0], "b-",  lw=lw, label="u_1 (protection)")
    ax.plot(t, U_store[s_all][:, 1], "r--", lw=lw, label="u_2 (treatment)")
    ax.plot(t, U_store[s_all][:, 2], "g:",  lw=lw, label="u_3 (vector ctrl)")
    ax.set_xlabel("Time");  ax.set_ylabel("Control intensity")
    ax.set_title("Optimal controls  (u_1+u_2+u_3 scenario)")
    ax.set_ylim([0, 1.05]);  ax.legend();  ax.grid(True)

    # (4) Total cost J
    ax = axes[1, 0]
    bars = ax.bar(range(n_scen), res_J, color=clr)
    ax.set_xticks(range(n_scen))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cost  J");  ax.set_title("Total Cost per Strategy");  ax.grid(True)

    # (5) Averted cases
    ax = axes[1, 1]
    bars2 = ax.bar(range(1, n_scen), res_AC[1:], color=clr[1:])
    ax.set_xticks(range(1, n_scen))
    ax.set_xticklabels(labels[1:], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("∫(I_h_base − I_h_ctrl) dt")
    ax.set_title("Total Averted Cases");  ax.grid(True)

    # (6) % cost reduction
    ax = axes[1, 2]
    pct_J_all = 100 * (res_J[0] - res_J[1:]) / res_J[0]
    bars3 = ax.bar(range(1, n_scen), pct_J_all, color=clr[1:])
    ax.set_xticks(range(1, n_scen))
    ax.set_xticklabels(labels[1:], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cost reduction (%)")
    ax.set_title("% Cost Reduction vs No-Control");  ax.grid(True)

    fig1.suptitle(f"Optimal Control — Normalized Random-Walk Network  (R₀ = {R0:.2f})",
                  fontsize=13, fontweight="bold")
    fig1.tight_layout()

    # ==================================================================
    # FIGURE 2 — Most cost-effective strategy detail
    # ==================================================================
    best_Y = Y_store[best_orig_idx]
    best_U = U_store[best_orig_idx]
    best_label = lbl_srt[best_k]

    Ih_best  = best_Y[:, 1] + best_Y[:, 5]
    Iv_best  = best_Y[:, 3] + best_Y[:, 7]
    Ih_base_agg = Y_base[:, 1] + Y_base[:, 5]
    Iv_base_agg = Y_base[:, 3] + Y_base[:, 7]

    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8))
    fig2.canvas.manager.set_window_title("Most Cost-Effective Strategy")

    ax = axes2[0, 0]
    ax.plot(t, Ih_base_agg, "k-",  lw=2.0, label="No control")
    ax.plot(t, Ih_best,     "r--", lw=2.2, label=best_label)
    ax.set_xlabel("Time", fontsize=12);  ax.set_ylabel("I_h1(t) + I_h2(t)", fontsize=12)
    ax.set_title("Total Infected Humans: Baseline vs Best Strategy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11);  ax.grid(True)

    ax = axes2[0, 1]
    ax.plot(t, Iv_base_agg, "k-",  lw=2.0, label="No control")
    ax.plot(t, Iv_best,     "b--", lw=2.2, label=best_label)
    ax.set_xlabel("Time", fontsize=12);  ax.set_ylabel("I_v1(t) + I_v2(t)", fontsize=12)
    ax.set_title("Total Infected Vectors: Baseline vs Best Strategy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11);  ax.grid(True)

    ax = axes2[1, 0]
    ax.plot(t, Y_base[:, 1], "k-",  lw=1.8, label="I_h1 (no ctrl)")
    ax.plot(t, Y_base[:, 5], "k--", lw=1.8, label="I_h2 (no ctrl)")
    ax.plot(t, best_Y[:, 1], "r-",  lw=2.0, label=f"I_h1 ({best_label})")
    ax.plot(t, best_Y[:, 5], "r--", lw=2.0, label=f"I_h2 ({best_label})")
    ax.set_xlabel("Time", fontsize=12);  ax.set_ylabel("I_h per patch", fontsize=12)
    ax.set_title("Patch-level Infected Humans", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9);  ax.grid(True)

    ax = axes2[1, 1]
    ctrl_names = ["u_1 (protection)", "u_2 (treatment)", "u_3 (vector ctrl)"]
    ctrl_clrs  = ["b", "r", "g"]
    ctrl_lsty  = ["-", "--", ":"]
    act_best   = active_mat[best_orig_idx]
    any_plotted = False
    for k in range(3):
        if act_best[k]:
            ax.plot(t, best_U[:, k], color=ctrl_clrs[k], ls=ctrl_lsty[k],
                    lw=2.2, label=ctrl_names[k])
            any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "No controls active", transform=ax.transAxes,
                ha="center", fontsize=13)
    ax.set_xlabel("Time", fontsize=12);  ax.set_ylabel("Control intensity", fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_title(f"Optimal Controls — {best_label}", fontsize=12, fontweight="bold")
    if any_plotted:
        ax.legend(fontsize=11)
    ax.grid(True)

    icer_best = ICER_val[best_k]
    icer_lbl = f"ICER = {icer_best:.4g}" if not np.isnan(icer_best) else "ICER = —"
    fig2.suptitle(f"Most Cost-Effective Strategy: {best_label}   ({icer_lbl})",
                  fontsize=14, fontweight="bold", color=(0.8, 0, 0))
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
