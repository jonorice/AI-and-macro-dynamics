#!/usr/bin/env python3
"""
Recreate Figures 1–3 (and optionally other steady-state figures) for
"AI and Macro Dynamics", using a correct GE linearisation that does NOT
impose replacement investment away from the steady state.

Key differences vs the earlier placeholder script:
- Consumption and the multiplier lambda_t are computed from the full
  resource constraint with endogenous investment flows:
    I^K_t = K_{t+1} - (1-delta)K_t
    X_t   = M_{t+1} - (1-delta_M)M_t
- The Euler system is therefore second-order in (K,M). We compute the
  local policy matrix J on the stable manifold via a companion-system
  eigen-decomposition.
- Figure 3 is generated from a nonlinear perfect-foresight transition
  (terminally anchored at the steady state), because a local linear
  approximation is unreliable far from the steady state.

Outputs (PNG) are written to the current directory by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, root


# ----------------------------
# Parameters and conversions
# ----------------------------

@dataclass(frozen=True)
class Params:
    beta: float
    alpha: float
    delta: float
    zeta: float
    delta_M: float
    phi: float
    chi: float
    sigma: float
    A: float = 1.0


def quarterly_beta(beta_a: float) -> float:
    return beta_a ** 0.25


def quarterly_depreciation(delta_a: float) -> float:
    return 1.0 - (1.0 - delta_a) ** 0.25


# ----------------------------
# CES consumption and prices
# ----------------------------

def ces_composite(C_T: float, D: float, theta: float, sigma: float) -> float:
    if abs(sigma - 1.0) < 1e-12:
        return (C_T ** theta) * (D ** (1.0 - theta))
    rho = (sigma - 1.0) / sigma
    return (theta * (C_T ** rho) + (1.0 - theta) * (D ** rho)) ** (1.0 / rho)


def dC_dCT(C_T: float, D: float, theta: float, sigma: float) -> float:
    """Derivative of CES composite C wrt C_T."""
    if abs(sigma - 1.0) < 1e-12:
        C = ces_composite(C_T, D, theta, sigma)
        return theta * C / C_T
    rho = (sigma - 1.0) / sigma
    inner = theta * (C_T ** rho) + (1.0 - theta) * (D ** rho)
    # dC/dC_T = theta * inner^{1/rho - 1} * C_T^{rho - 1}
    return theta * (inner ** (1.0 / rho - 1.0)) * (C_T ** (rho - 1.0))


def mrs(C_T: float, D: float, theta: float, sigma: float, floor: float = 1e-300) -> float:
    """Marginal rate of substitution p_D between D and C_T."""
    D_safe = max(D, floor)
    return (1.0 - theta) / theta * (C_T / D_safe) ** (1.0 / sigma)


# ----------------------------
# Steady state
# ----------------------------

def K_star_from_M(M: float, p: Params) -> float:
    """Steady-state K as a function of M from the K Euler equation."""
    denom = (p.beta ** (-1.0) - (1.0 - p.delta))
    Xi = p.alpha * p.A / denom
    return (Xi * (M ** p.zeta)) ** (1.0 / (1.0 - p.alpha))


def steady_state(theta: float, psi: float, p: Params) -> Dict[str, float]:
    """
    Steady state under the binding-capacity regime (p_D > chi),
    solving the M Euler equation after substituting K*(M).
    """

    def excess_return(M: float) -> float:
        if M <= 0.0:
            return 1e12
        K = K_star_from_M(M, p)
        Y = p.A * (K ** p.alpha) * (M ** p.zeta)
        D = psi * (M ** p.phi)
        C_T = Y - p.delta * K - p.delta_M * M - p.chi * D
        if C_T <= 0.0 or D <= 0.0:
            return 1e12
        p_D = mrs(C_T, D, theta, p.sigma)
        services = max(0.0, p_D - p.chi) * p.phi * psi * (M ** (p.phi - 1.0))
        G = (1.0 - p.delta_M) + p.zeta * (Y / M) + services
        return p.beta * G - 1.0

    lo = 1e-12
    hi = 1.0
    f_hi = excess_return(hi)
    it = 0
    while f_hi > 0.0 and it < 500:
        hi *= 2.0
        f_hi = excess_return(hi)
        it += 1

    sol = root_scalar(excess_return, bracket=(lo, hi), method="brentq")
    M = float(sol.root)
    K = K_star_from_M(M, p)
    Y = p.A * (K ** p.alpha) * (M ** p.zeta)
    D = psi * (M ** p.phi)
    C_T = Y - p.delta * K - p.delta_M * M - p.chi * D
    C = ces_composite(C_T, D, theta, p.sigma)
    p_D = mrs(C_T, D, theta, p.sigma)

    return {"K": K, "M": M, "Y": Y, "D": D, "C_T": C_T, "C": C, "p_D": p_D}


def calibrate_theta_psi(
    p: Params,
    s_D_target: float = 0.10,
    theta0: float = 0.85,
    psi0: float = 0.20,
    damp: float = 0.20,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> Tuple[float, float, Dict[str, float], int]:
    """
    Calibrate theta and psi to match the steady-state expenditure share on D.

    Shares under log utility satisfy:
      s_D = (p_D D) / (C_T + p_D D)
    and the CES demand system pins down theta via relative price p_D.
    """
    theta = theta0
    psi = psi0

    for n in range(1, max_iter + 1):
        ss = steady_state(theta, psi, p)
        p_D = ss["p_D"]
        C_T = ss["C_T"]
        M = ss["M"]

        ratio = (s_D_target / (1.0 - s_D_target)) * (p_D ** (p.sigma - 1.0))
        theta_new = 1.0 / (1.0 + ratio ** (1.0 / p.sigma))

        # D consistent with expenditure share s_D
        D_implied = (s_D_target / (1.0 - s_D_target)) * (C_T / p_D)
        psi_new = D_implied / (M ** p.phi)

        theta_upd = (1.0 - damp) * theta + damp * theta_new
        psi_upd = (1.0 - damp) * psi + damp * psi_new

        rel_theta = abs(theta_upd - theta) / max(theta, 1e-12)
        rel_psi = abs(psi_upd - psi) / max(psi, 1e-12)

        theta, psi = theta_upd, psi_upd

        if rel_theta < tol and rel_psi < tol:
            ss = steady_state(theta, psi, p)
            return float(theta), float(psi), ss, n

    raise RuntimeError("Calibration did not converge.")


# ----------------------------
# GE Euler system in (K,M)
# ----------------------------

def period_allocation(
    K_t: float,
    M_t: float,
    K_tp1: float,
    M_tp1: float,
    theta: float,
    psi: float,
    p: Params,
) -> Dict[str, float] | None:
    """
    Period-t allocation given current states and chosen next states.
    Endogenous investments:
      I^K_t = K_{t+1} - (1-delta)K_t
      X_t   = M_{t+1} - (1-delta_M)M_t
    """
    Y = p.A * (K_t ** p.alpha) * (M_t ** p.zeta)
    D = psi * (M_t ** p.phi)

    I_K = K_tp1 - (1.0 - p.delta) * K_t
    X = M_tp1 - (1.0 - p.delta_M) * M_t

    C_T = Y - I_K - X - p.chi * D
    if C_T <= 0.0:
        return None

    C = ces_composite(C_T, D, theta, p.sigma)
    lam = (1.0 / C) * dC_dCT(C_T, D, theta, p.sigma)
    p_D = mrs(C_T, D, theta, p.sigma)

    return {"Y": Y, "D": D, "I_K": I_K, "X": X, "C_T": C_T, "C": C, "lam": lam, "p_D": p_D}


def euler_residuals(
    z_t: np.ndarray,
    z_tp1: np.ndarray,
    z_tp2: np.ndarray,
    theta: float,
    psi: float,
    p: Params,
) -> np.ndarray:
    """
    Euler residuals at time t, allowing lambda_{t+1} to depend on (z_{t+1}, z_{t+2}).
    z vectors are [K, M] in levels.
    """
    K_t, M_t = float(z_t[0]), float(z_t[1])
    K1, M1 = float(z_tp1[0]), float(z_tp1[1])
    K2, M2 = float(z_tp2[0]), float(z_tp2[1])

    alloc_t = period_allocation(K_t, M_t, K1, M1, theta, psi, p)
    alloc_1 = period_allocation(K1, M1, K2, M2, theta, psi, p)

    if alloc_t is None or alloc_1 is None:
        return np.array([1e6, 1e6])

    lam_t = alloc_t["lam"]
    lam_1 = alloc_1["lam"]

    Y1 = alloc_1["Y"]
    p_D1 = alloc_1["p_D"]

    Rk = (1.0 - p.delta) + p.alpha * (Y1 / K1)

    services = max(0.0, p_D1 - p.chi) * p.phi * psi * (M1 ** (p.phi - 1.0))
    Rm = (1.0 - p.delta_M) + p.zeta * (Y1 / M1) + services

    return np.array([
        lam_t - p.beta * lam_1 * Rk,
        lam_t - p.beta * lam_1 * Rm,
    ])


def F_logs(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    theta: float,
    psi: float,
    p: Params,
) -> np.ndarray:
    """Euler residuals as a function of logs of (K,M) at t, t+1, t+2."""
    z0 = np.exp(x0)
    z1 = np.exp(x1)
    z2 = np.exp(x2)
    return euler_residuals(z0, z1, z2, theta, psi, p)


def jacobian_F_wrt(
    x0: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    which: int,
    theta: float,
    psi: float,
    p: Params,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Central-difference Jacobian of F_logs with respect to x0, x1, or x2 (each 2-vector).
    which = 0 (x0), 1 (x1), 2 (x2)
    """
    J = np.zeros((2, 2))
    for j in range(2):
        dx = np.zeros(2)
        dx[j] = eps
        if which == 0:
            f_plus = F_logs(x0 + dx, x1, x2, theta, psi, p)
            f_minus = F_logs(x0 - dx, x1, x2, theta, psi, p)
        elif which == 1:
            f_plus = F_logs(x0, x1 + dx, x2, theta, psi, p)
            f_minus = F_logs(x0, x1 - dx, x2, theta, psi, p)
        else:
            f_plus = F_logs(x0, x1, x2 + dx, theta, psi, p)
            f_minus = F_logs(x0, x1, x2 - dx, theta, psi, p)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


def policy_matrix_J(theta: float, psi: float, p: Params, ss: Dict[str, float], eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the local GE policy matrix J on the stable manifold, in log deviations:
      x_{t+1} = J x_t
    where x_t = [log(K_t/K*), log(M_t/M*)]'.

    The Euler block is second-order:
      A x_{t+2} + B x_{t+1} + C x_t = 0
    so we form the 4x4 companion system and extract the stable eigenspace.
    """
    zstar = np.array([ss["K"], ss["M"]])
    xstar = np.log(zstar)

    A = jacobian_F_wrt(xstar, xstar, xstar, 2, theta, psi, p, eps=eps)
    B = jacobian_F_wrt(xstar, xstar, xstar, 1, theta, psi, p, eps=eps)
    C = jacobian_F_wrt(xstar, xstar, xstar, 0, theta, psi, p, eps=eps)

    Ainv = np.linalg.inv(A)
    T = np.block([
        [-Ainv @ B, -Ainv @ C],
        [np.eye(2), np.zeros((2, 2))],
    ])

    eigvals, eigvecs = np.linalg.eig(T)
    stable = [i for i, v in enumerate(eigvals) if abs(v) < 1.0 - 1e-10]
    if len(stable) != 2:
        raise RuntimeError(f"Expected 2 stable eigenvalues, got {len(stable)}")

    V = eigvecs[:, stable]          # 4x2
    V1 = V[:2, :]                   # s_{t+1}
    V0 = V[2:, :]                   # s_t
    J = np.real_if_close(V1 @ np.linalg.inv(V0))

    return np.array(J, dtype=float), eigvals


# ----------------------------
# Nonlinear perfect-foresight transition (for Figure 3)
# ----------------------------

def period_allocation_safe(
    K_t: float,
    M_t: float,
    K_tp1: float,
    M_tp1: float,
    theta: float,
    psi: float,
    p: Params,
) -> Dict[str, float] | None:
    # Same as period_allocation but with a floor in the MRS to improve numerical robustness
    Y = p.A * (K_t ** p.alpha) * (M_t ** p.zeta)
    D = psi * (M_t ** p.phi)
    I_K = K_tp1 - (1.0 - p.delta) * K_t
    X = M_tp1 - (1.0 - p.delta_M) * M_t
    C_T = Y - I_K - X - p.chi * D
    if C_T <= 0.0:
        return None
    C = ces_composite(C_T, D, theta, p.sigma)
    lam = (1.0 / C) * dC_dCT(C_T, D, theta, p.sigma)
    p_D = mrs(C_T, D, theta, p.sigma, floor=1e-300)
    return {"Y": Y, "D": D, "I_K": I_K, "X": X, "C_T": C_T, "C": C, "lam": lam, "p_D": p_D}


def euler_residuals_safe(
    z_t: np.ndarray,
    z_tp1: np.ndarray,
    z_tp2: np.ndarray,
    theta: float,
    psi: float,
    p: Params,
) -> np.ndarray:
    K_t, M_t = float(z_t[0]), float(z_t[1])
    K1, M1 = float(z_tp1[0]), float(z_tp1[1])
    K2, M2 = float(z_tp2[0]), float(z_tp2[1])

    alloc_t = period_allocation_safe(K_t, M_t, K1, M1, theta, psi, p)
    alloc_1 = period_allocation_safe(K1, M1, K2, M2, theta, psi, p)

    if alloc_t is None or alloc_1 is None:
        return np.array([1e6, 1e6])

    lam_t = alloc_t["lam"]
    lam_1 = alloc_1["lam"]

    Y1 = alloc_1["Y"]
    p_D1 = alloc_1["p_D"]

    Rk = (1.0 - p.delta) + p.alpha * (Y1 / K1)

    M1_safe = max(M1, 1e-300)
    services = max(0.0, p_D1 - p.chi) * p.phi * psi * (M1_safe ** (p.phi - 1.0))
    Rm = (1.0 - p.delta_M) + p.zeta * (Y1 / M1_safe) + services

    return np.array([
        lam_t - p.beta * lam_1 * Rk,
        lam_t - p.beta * lam_1 * Rm,
    ])


def solve_transition_terminal(
    z0: np.ndarray,
    T: int,
    theta: float,
    psi: float,
    p: Params,
    z_terminal: np.ndarray,
    tol: float = 1e-10,
) -> Tuple[bool, np.ndarray]:
    """
    Solve for z1..z_{T-1} (levels, in logs) such that
      z_T = z_{T+1} = z_terminal
    and Euler equations hold for t = 0..T-2.
    """
    Kstar, Mstar = float(z_terminal[0]), float(z_terminal[1])

    guess = []
    for t in range(1, T):
        w = (T - t) / T
        K_guess = Kstar * math.exp(w * math.log(z0[0] / Kstar))
        M_guess = Mstar * math.exp(w * math.log(z0[1] / Mstar))
        guess.extend([math.log(K_guess), math.log(M_guess)])
    x_guess = np.array(guess)

    def residuals(x: np.ndarray) -> np.ndarray:
        z = [np.array(z0)]
        for t in range(1, T):
            logK, logM = x[2 * (t - 1) : 2 * (t - 1) + 2]
            z.append(np.array([math.exp(logK), math.exp(logM)]))
        z.append(np.array([Kstar, Mstar]))  # z_T
        z.append(np.array([Kstar, Mstar]))  # z_{T+1}

        res = []
        for t in range(0, T - 1):  # 0..T-2
            res.extend(euler_residuals_safe(z[t], z[t + 1], z[t + 2], theta, psi, p).tolist())
        return np.array(res)

    sol = root(residuals, x_guess, method="hybr", tol=tol)
    if not sol.success:
        return False, np.empty((0, 2))

    # Reconstruct path 0..T (z_T included) for plotting
    z_path = [np.array(z0)]
    for t in range(1, T):
        logK, logM = sol.x[2 * (t - 1) : 2 * (t - 1) + 2]
        z_path.append(np.array([math.exp(logK), math.exp(logM)]))
    z_path.append(np.array([Kstar, Mstar]))  # z_T
    return True, np.array(z_path)


# ----------------------------
# Figures
# ----------------------------

def fig1_irf_baseline(J: np.ndarray, p: Params, horizon: int = 40, fname: str = "fig1_corrected.png") -> None:
    """
    Figure 1: one-off 1% level shock to M0, K0 unchanged.
    GE: local policy x_{t+1}=J x_t.
    PE: replacement-investment benchmark implies m_t = (1-delta_M)^t m0 and k_t = 0.
    """
    m0 = math.log(1.01)
    x = np.zeros((horizon + 1, 2))
    x[0, :] = np.array([0.0, m0])
    for t in range(horizon):
        x[t + 1, :] = (J @ x[t, :])

    tgrid = np.arange(horizon + 1)

    m_ge = 100.0 * x[:, 1]
    k_ge = 100.0 * x[:, 0]
    m_pe = 100.0 * ((1.0 - p.delta_M) ** tgrid) * m0
    k_pe = np.zeros_like(tgrid, dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(tgrid, m_ge, label="GE")
    ax[0].plot(tgrid, m_pe, linestyle="--", label="PE")
    ax[0].set_xlabel("Quarters")
    ax[0].set_ylabel("Percent log deviation")
    ax[0].set_title("AI compute stock (M)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(tgrid, k_ge, label="GE")
    ax[1].plot(tgrid, k_pe, linestyle="--", label="PE")
    ax[1].set_xlabel("Quarters")
    ax[1].set_ylabel("Percent log deviation")
    ax[1].set_title("Tangible capital (K)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def fig2_longevity_shock(
    theta: float,
    psi: float,
    p: Params,
    ss_old: Dict[str, float],
    deltaM_drop: float = 0.02,
    horizon: int = 40,
    fname: str = "fig2_corrected.png",
) -> Tuple[np.ndarray, Dict[str, float], Params]:
    """
    Figure 2: permanent fall in quarterly depreciation delta_M by deltaM_drop.
    GE: local policy evaluated at the NEW steady state.
    PE: mechanical convergence with the new survival rate.
    """
    p_new = Params(
        beta=p.beta,
        alpha=p.alpha,
        delta=p.delta,
        zeta=p.zeta,
        delta_M=p.delta_M - deltaM_drop,
        phi=p.phi,
        chi=p.chi,
        sigma=p.sigma,
        A=p.A,
    )

    ss_new = steady_state(theta, psi, p_new)
    J_new, _ = policy_matrix_J(theta, psi, p_new, ss_new)

    x0 = np.log(np.array([ss_old["K"], ss_old["M"]])) - np.log(np.array([ss_new["K"], ss_new["M"]]))

    x = np.zeros((horizon + 1, 2))
    x[0, :] = x0
    for t in range(horizon):
        x[t + 1, :] = (J_new @ x[t, :])

    tgrid = np.arange(horizon + 1)
    m_ge = 100.0 * x[:, 1]
    m_pe = 100.0 * ((1.0 - p_new.delta_M) ** tgrid) * x0[1]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(tgrid, m_ge, label="GE")
    ax.plot(tgrid, m_pe, marker="o", linestyle="None", markersize=3, label="PE")
    ax.set_xlabel("Quarters")
    ax.set_ylabel("Percent log deviation (from new SS)")
    ax.set_title("Longevity shock, AI compute stock (M)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

    return J_new, ss_new, p_new


def fig3_diffusion(
    theta: float,
    psi: float,
    p: Params,
    ss: Dict[str, float],
    K0_share: float = 0.01,
    M0_share: float = 0.01,
    horizon: int = 160,
    fname: str = "fig3_corrected.png",
) -> None:
    """
    Figure 3: diffusion from a low initial capital state (K0_share, M0_share) of the SS.

    GE: nonlinear perfect-foresight transition, terminally anchored at SS.
    PE: mechanical diffusion benchmark from the linear difference equation
         M_{t+1} = (1-delta_M) M_t + delta_M M*
       starting at M0 = M0_share * M*.
    """
    z_terminal = np.array([ss["K"], ss["M"]])
    z0 = np.array([K0_share * ss["K"], M0_share * ss["M"]])

    ok, z_path = solve_transition_terminal(z0, T=horizon, theta=theta, psi=psi, p=p, z_terminal=z_terminal)
    if not ok:
        raise RuntimeError("Nonlinear transition solver failed for Figure 3.")

    ratio_ge = z_path[:, 1] / ss["M"]
    tgrid = np.arange(len(ratio_ge))

    # PE diffusion benchmark with fixed replacement investment delta_M * M*
    M0 = M0_share * ss["M"]
    ratio_pe = 1.0 - ((1.0 - p.delta_M) ** tgrid) * (1.0 - (M0 / ss["M"]))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(tgrid, ratio_ge, label="GE (nonlinear)")
    ax.plot(tgrid, ratio_pe, linestyle="--", label="PE")
    ax.set_xlabel("Quarters")
    ax.set_ylabel(r"$M_t / M^\ast$")
    ax.set_title("Diffusion from a low initial compute stock")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    # Baseline calibration (quarterly)
    beta_a = 0.99
    delta_a = 0.06
    delta_M_a = 0.33

    p = Params(
        beta=quarterly_beta(beta_a),
        alpha=0.30,
        delta=quarterly_depreciation(delta_a),
        zeta=0.06,
        delta_M=quarterly_depreciation(delta_M_a),
        phi=0.70,
        chi=0.006,
        sigma=1.50,
        A=1.0,
    )

    theta, psi, ss, iters = calibrate_theta_psi(p)
    J, eigvals_T = policy_matrix_J(theta, psi, p, ss)

    eig_J = np.linalg.eigvals(J)

    print("\nBaseline calibration")
    print(f"  theta = {theta:.6f}, psi = {psi:.6f} (iters {iters})")
    print(f"  K* = {ss['K']:.6f}, M* = {ss['M']:.6f}, p_D* = {ss['p_D']:.6f}")
    print("\nCorrect GE policy matrix J (log deviations):")
    print(J)
    print("\nEigenvalues of J:")
    print(eig_J)

    # Figures
    fig1_irf_baseline(J, p, horizon=40, fname="fig1_corrected.png")
    J_new, ss_new, p_new = fig2_longevity_shock(theta, psi, p, ss, deltaM_drop=0.02, horizon=40, fname="fig2_corrected.png")

    # Use the same initial shares as the earlier placeholder script, but now in a feasible nonlinear transition
    fig3_diffusion(theta, psi, p, ss, K0_share=0.01, M0_share=0.01, horizon=160, fname="fig3_corrected.png")

    print("\nSaved:")
    print("  fig1_corrected.png")
    print("  fig2_corrected.png")
    print("  fig3_corrected.png")


if __name__ == "__main__":
    main()
