Pure-Python (numpy/pandas) simulation and basic theory checks for:

  - symmetric GARCH(1,1) with Student t innovations
  - GJR-GARCH(1,1) (a.k.a. GJR / threshold GARCH) with Student t innovations

The scripts compare empirical autocorrelations from simulation to closed-form theoretical autocorrelations of squared returns. The GJR script also compares the leverage cross-correlation corr(ε_t, ε_{t+k}^2) to its theoretical geometric decay (the level uses a simulation estimate of E[h^(3/2)]).

All formulas in the code assume symmetric innovations with E[z^2]=1.
