import numpy as np
import pandas as pd
from math import gamma as gamma_fn, sqrt, pi
from typing import Tuple, Dict, Any, Optional
from numpy.typing import NDArray

def simulate_gjr_garch_t(
    nobs: int, 
    omega: float, 
    alpha: float, 
    gamma: float, 
    beta: float, 
    nu: float,
    burn: int = 5000, 
    seed: int = 123, 
    h0: Optional[float] = None, 
    floor: float = 1e-14
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    simulate gjr-garch(1,1) with student t innovations standardized to var(z)=1
    eps_t = sqrt(h_t) * z_t
    h_t   = omega + alpha*eps_{t-1}^2 + gamma*eps_{t-1}^2*1{eps_{t-1}<0} + beta*h_{t-1}
    """
    if nu <= 2:
        raise ValueError("nu must be > 2")
    rng = np.random.default_rng(seed)
    # z ~ t_nu scaled to var 1
    t_scale = sqrt((nu - 2.0) / nu)
    z = rng.standard_t(df=nu, size=int(nobs) + int(burn)) * t_scale
    kappa = alpha + 0.5 * gamma + beta
    
    h_prev: float
    if h0 is None:
        if kappa < 1.0:
            h_prev = omega / (1.0 - kappa)
        else:
            h_prev = max(omega, floor)
    else:
        if h0 <= 0:
            raise ValueError("h0 must be > 0")
        h_prev = float(h0)
        
    n_total = int(nobs) + int(burn)
    h = np.empty(n_total, dtype=float)
    eps = np.empty(n_total, dtype=float)
    
    for t in range(n_total):
        eps_t = sqrt(h_prev) * z[t]
        h[t] = h_prev
        eps[t] = eps_t
        ind = 1.0 if eps_t < 0.0 else 0.0
        h_next = omega + alpha * (eps_t * eps_t) + gamma * (eps_t * eps_t) * ind + beta * h_prev
        if h_next <= floor:
            h_next = floor
        h_prev = h_next
        
    return eps[burn:], h[burn:]

def acf(x: NDArray[np.float64], nlags: int) -> NDArray[np.float64]:
    """autocorrelation for lags 0..nlags"""
    x_arr = np.asarray(x, dtype=float)
    x_centered = x_arr - x_arr.mean()
    n = x_centered.size
    c = np.correlate(x_centered, x_centered, mode="full")[n - 1:n + nlags]
    c0 = c[0]
    return c / c0 if c0 > 0 else np.zeros(nlags + 1)

def crosscorr(x: NDArray[np.float64], y: NDArray[np.float64], nlags: int) -> NDArray[np.float64]:
    """corr(x_t, y_{t+k}) for k=0..nlags"""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    vx = np.mean(x_centered * x_centered)
    vy = np.mean(y_centered * y_centered)
    if vx <= 0 or vy <= 0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1, dtype=float)
    for k in range(nlags + 1):
        out[k] = np.mean(x_centered[:x_centered.size - k] * y_centered[k:]) / sqrt(vx * vy)
    return out

def m4_student_t_standardized(nu: float) -> float:
    """E[z^4] for z = t_nu scaled to var 1 (requires nu>4)"""
    if nu <= 4:
        raise ValueError("need nu>4 for finite fourth moment")
    return 3.0 * (nu - 2.0) / (nu - 4.0)

def abs_moment_student_t_standardized(nu: float, r: float) -> float:
    """E[|z|^r] for z = t_nu scaled to var 1 (requires r<nu and nu>2)"""
    if nu <= 2:
        raise ValueError("need nu>2")
    if r >= nu:
        raise ValueError("need r < nu")
    return (nu - 2.0) ** (r / 2.0) * gamma_fn((r + 1.0) / 2.0) * gamma_fn((nu - r) / 2.0) / (
        sqrt(pi) * gamma_fn(nu / 2.0)
    )

def theoretical_acf_sq_returns_gjr(
    omega: float, 
    alpha: float, 
    gamma: float, 
    beta: float, 
    nu: float, 
    nlags: int
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """
    theoretical acf of s_t=eps_t^2 for symmetric z with var 1 and finite 4th moment
    rho(k) = rho(1)*kappa^(k-1), k>=1
    """
    kappa = alpha + 0.5 * gamma + beta
    if kappa >= 1.0:
        raise ValueError("need alpha + 0.5*gamma + beta < 1")
    m4 = m4_student_t_standardized(nu)
    a2 = alpha * alpha + alpha * gamma + 0.5 * gamma * gamma
    eta = beta * beta + 2.0 * beta * (alpha + 0.5 * gamma) + a2 * m4
    if eta >= 1.0:
        raise ValueError("need eta < 1 for finite E[h^2] (finite fourth moment of eps)")
    hbar = omega / (1.0 - kappa)
    eh2 = (omega * omega) * (1.0 + kappa) / ((1.0 - kappa) * (1.0 - eta))
    var_s = m4 * eh2 - hbar * hbar
    cov1 = omega * hbar + ((alpha + 0.5 * gamma) * m4 + beta) * eh2 - hbar * hbar
    rho1 = cov1 / var_s
    rho = np.empty(nlags + 1, dtype=float)
    rho[0] = 1.0
    for k in range(1, nlags + 1):
        rho[k] = rho1 * (kappa ** (k - 1))
    info = {"kappa": kappa, "m4": m4, "eta": eta, "hbar": hbar, "eh2": eh2, "rho1": rho1}
    return rho, info

def theoretical_crosscorr_return_future_sq(
    omega: float, 
    alpha: float, 
    gamma: float, 
    beta: float, 
    nu: float, 
    nlags: int, 
    eh32: float
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """
    corr(eps_t, eps_{t+k}^2), k>=1
    corr(k) = corr(1)*kappa^(k-1)
    level uses eh32 = E[h^(3/2)] (estimated from a long stationary run)
    """
    kappa = alpha + 0.5 * gamma + beta
    if kappa >= 1.0:
        raise ValueError("need kappa < 1")
    hbar = omega / (1.0 - kappa)
    _, info = theoretical_acf_sq_returns_gjr(omega, alpha, gamma, beta, nu, 1)
    m4 = info["m4"]
    eh2 = info["eh2"]
    var_s = m4 * eh2 - hbar * hbar
    ez_abs3 = abs_moment_student_t_standardized(nu, 3.0)
    cov1 = -0.5 * gamma * eh32 * ez_abs3
    corr1 = cov1 / sqrt(hbar * var_s)
    out = np.empty(nlags + 1, dtype=float)
    out[0] = 0.0
    for k in range(1, nlags + 1):
        out[k] = corr1 * (kappa ** (k - 1))
    info2 = {"kappa": kappa, "cov1": cov1, "corr1": corr1, "ez_abs3": ez_abs3}
    return out, info2

def simulate_garch_t(
    nobs: int, 
    omega: float, 
    alpha: float, 
    beta: float, 
    nu: float,
    burn: int = 5000, 
    seed: int = 123, 
    h0: Optional[float] = None, 
    floor: float = 1e-14
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    simulate garch(1,1) with student t innovations standardized to var(z)=1
    eps_t = sqrt(h_t) * z_t
    h_t   = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
    """
    if nu <= 2:
        raise ValueError("nu must be > 2")
    rng = np.random.default_rng(seed)
    t_scale = sqrt((nu - 2.0) / nu)
    z = rng.standard_t(df=nu, size=int(nobs) + int(burn)) * t_scale
    kappa = alpha + beta
    
    h_prev: float
    if h0 is None:
        if kappa < 1.0:
            h_prev = omega / (1.0 - kappa)
        else:
            h_prev = max(omega, floor)
    else:
        if h0 <= 0:
            raise ValueError("h0 must be > 0")
        h_prev = float(h0)
        
    n_total = int(nobs) + int(burn)
    h = np.empty(n_total, dtype=float)
    eps = np.empty(n_total, dtype=float)
    for t in range(n_total):
        eps_t = sqrt(h_prev) * z[t]
        h[t] = h_prev
        eps[t] = eps_t
        h_next = omega + alpha * (eps_t * eps_t) + beta * h_prev
        if h_next <= floor:
            h_next = floor
        h_prev = h_next
    return eps[burn:], h[burn:]

def theoretical_acf_sq_returns_garch(
    omega: float, 
    alpha: float, 
    beta: float, 
    nu: float, 
    nlags: int
) -> Tuple[NDArray[np.float64], Dict[str, float]]:
    """
    theoretical acf of s_t=eps_t^2 for symmetric garch(1,1) with standardized t innovations
    """
    kappa = alpha + beta
    if kappa >= 1.0:
        raise ValueError("need alpha + beta < 1")
    m4 = m4_student_t_standardized(nu)
    eta = beta * beta + 2.0 * alpha * beta + (alpha * alpha) * m4
    if eta >= 1.0:
        raise ValueError("need eta < 1 for finite E[h^2] (finite fourth moment of eps)")
    hbar = omega / (1.0 - kappa)
    eh2 = (omega * omega) * (1.0 + kappa) / ((1.0 - kappa) * (1.0 - eta))
    var_s = m4 * eh2 - hbar * hbar
    cov1 = omega * hbar + (alpha * m4 + beta) * eh2 - hbar * hbar
    rho1 = cov1 / var_s
    rho = np.empty(nlags + 1, dtype=float)
    rho[0] = 1.0
    for k in range(1, nlags + 1):
        rho[k] = rho1 * (kappa ** (k - 1))
    info = {"kappa": kappa, "m4": m4, "eta": eta, "hbar": hbar, "eh2": eh2, "rho1": rho1}
    return rho, info