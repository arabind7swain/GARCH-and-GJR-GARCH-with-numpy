"""
simulate a garch(1,1) with standardized student t innovations and compare
empirical autocorrelations of squared returns to closed-form theoretical values;
print results across multiple seeds.
"""

import numpy as np
import pandas as pd
from garch import theoretical_acf_sq_returns_garch, simulate_garch_t, acf


def main():
    # fixed parameters (printed once)
    omega = 1.0e-6
    alpha = 0.05
    beta = 0.90
    nu = 8.0

    nobs = 20000
    burn = 5000
    nlags = 5

    # compute once so we can print kappa/eta once too
    _, info = theoretical_acf_sq_returns_garch(omega, alpha, beta, nu, nlags)
    kappa = info["kappa"]
    eta = info["eta"]

    print("#obs:", nobs)
    print("\nparameters")
    print(f"omega={omega} alpha={alpha} beta={beta} nu={nu}")
    print(f"kappa = alpha + beta = {kappa:.6g}")
    print(f"eta   = {eta:.6g} (need <1 for finite fourth moment)")
    print("")

    seed0 = 123
    nsim = 3
    for i in range(nsim):
        main_loop(seed0 + 1000000 * i, omega, alpha, beta, nu, nobs, burn, nlags)


def main_loop(seed, omega, alpha, beta, nu, nobs, burn, nlags):
    eps, h = simulate_garch_t(
        nobs=nobs,
        omega=omega,
        alpha=alpha,
        beta=beta,
        nu=nu,
        burn=burn,
        seed=seed,
    )
    s = eps * eps

    # theory vs empirical acf of squared returns
    rho_th, info = theoretical_acf_sq_returns_garch(omega, alpha, beta, nu, nlags)
    rho_emp = acf(s, nlags)

    # store acf table in pandas dataframe
    lags = np.arange(1, nlags + 1, dtype=int)

    df_acf_sq = pd.DataFrame({
        "lag": lags,
        "rho_theory": rho_th[1:],
        "rho_empirical": rho_emp[1:],
    })
    df_acf_sq["diff"] = df_acf_sq["rho_empirical"] - df_acf_sq["rho_theory"]

    rmse_acf_sq = float(np.sqrt(np.mean(df_acf_sq["diff"].to_numpy() ** 2)))

    # per-simulation outputs (fixed parameters are not reprinted)
    print(f"seed: {seed}")
    print("")

    print("acf of squared returns (eps^2): theory vs empirical")
    print(df_acf_sq.to_string(index=False))
    print("")
    print(f"rmse (lags 1..{nlags}) = {rmse_acf_sq:.6g}")
    print("")


if __name__ == "__main__":
    main()

