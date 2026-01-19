# GARCH-from-scratch: Simulation & Theory
This repository provides a pure-Python implementation (using NumPy and Pandas) for simulating and validating GARCH-family models. It is mostly for learning purposes. It focuses on comparing empirical results from simulated paths against closed-form theoretical expectations autocorrelations of squared returns. 

## Models
- symmetric GARCH(1,1) with Student t innovations
- GJR-GARCH(1,1) (a.k.a. GJR / threshold GARCH) with Student t innovations
The GJR script also compares the leverage cross-correlation corr(ε_t, ε_{t+k}^2) to its theoretical geometric decay (the level uses a simulation estimate of E[h^(3/2)]). All formulas in the code assume symmetric innovations with E[z^2]=1.
## File structure
- `garch.py`
  
  - core simulation functions
  
  - theory
- `maingarch.py`
  - Prints a table comparing theoretical vs empirical ACF of squared returns for symmetric GARCH(1,1).

- `maingarch-grj.py`

  - Prints a table comparing theoretical vs empirical ACF of squared returns for GJR-GARCH(1,1)
 
  - prints a table comparing theoretical vs empirical corr(ε_t, ε_{t+k}^2)

 ## Usage

 ### Prerequisites

 ```bash
pip install numpy pandas
```
### Running Simulations

The main scripts execute multiple simulations with different seeds to verify the consistency of the empirical results against theory.

Run symmetric GARCH(1,1):

```bash
python maingarch.py
```

Run GJR-GARCH(1,1):

```bash
python maingarch_gjr.py
```

The ACF comparison table(s) are displayed as pandas DataFrames. An RMSE across lags is used as a quick "match quality" metric.

## Changing run parameters

Edit the constants near the top of `main()` in `maingarch.py` or `maingarch_gjr.py`:

- ω, α, β, ν (and γ for GJR)
- `nobs` number of kept observations
- `burn` burn-in length
- `nlags` number of ACF lags to compare
- `nsim` number of simulations
- seed schedule (currently `seed0 + 1000000*i`)


## Model definitions

### Standardized Student t innovations

The code uses numpy's `standard_t(df=ν)` and rescales it so Var(z)=1:

- if x ~ t_ν, then Var(x) = ν/(ν-2) for ν>2
- set z = x * sqrt((ν-2)/ν) so Var(z)=1

This is what the simulators use in:

- `simulate_garch_t`
- `simulate_gjr_garch_t`

### Symmetric GARCH(1,1)

ε_t = sqrt(h_t) * z_t

h_t = ω + α * ε_{t-1}^2 + β * h_{t-1}

Stationarity (finite unconditional variance) requires:

κ = α + β < 1

### GJR-GARCH(1,1)

ε_t = sqrt(h_t) * z_t

h_t = ω + α * ε_{t-1}^2 + γ * ε_{t-1}^2 * 1{ε_{t-1}<0} + β * h_{t-1}

For symmetric z_t, P(ε_{t-1}<0)=1/2, so stationarity (finite unconditional
variance) requires:

κ = α + β + 0.5*γ < 1

## Theoretical ACF of squared returns

Let s_t = ε_t^2. Under symmetric innovations with E[z^2]=1 and finite fourth
moment, the autocorrelation of s_t is geometric:

ρ_s(k) = Corr(s_{t+k}, s_t) = ρ_s(1) * κ^(k-1),  k>=1

The code computes ρ_s(1) in closed form by solving for E[h] and E[h^2].

### Fourth-moment condition

To have finite Var(ε_t^2) (and therefore a well-defined ACF of squared
returns), you also need a finite fourth moment and a contraction condition
for E[h^2]. The code checks this via `η < 1`.

For standardized Student t, you need ν > 4 and:

m4 = E[z^4] = 3*(ν-2)/(ν-4)

#### Symmetric GARCH(1,1): η

η = β^2 + 2*α*β + α^2*m4

#### GJR-GARCH(1,1): η

Define:

a2 = α^2 + α*γ + 0.5*γ^2

η = β^2 + 2*β*(α + 0.5*γ) + a2*m4

## Leverage cross-correlation for GJR

The GJR script also reports:

corr(ε_t, ε_{t+k}^2), k>=1

For symmetric z_t, this cross-correlation decays geometrically at the same
rate κ. The level depends on E[h^(3/2)], which is not computed in closed
form in this repo; `xgarch_gjr.py` estimates it from the simulated path and
plugs it into `theoretical_crosscorr_return_future_sq`.

For symmetric GARCH(1,1), the corresponding leverage cross-correlations are
zero by symmetry, so `xgarch.py` does not compute them.
