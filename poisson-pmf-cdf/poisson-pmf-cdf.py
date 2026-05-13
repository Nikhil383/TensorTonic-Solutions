import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    def poisson_pmf(lam, x):
        # log(x!) for numerical stability
        log_fact = np.sum(np.log(np.arange(1, x + 1))) if x > 0 else 0
        
        return np.exp(-lam + x * np.log(lam) - log_fact)

    # PMF at k
    pmf = poisson_pmf(lam, k)

    # CDF = sum of PMFs from 0 to k
    cdf = 0
    for i in range(k + 1):
        cdf += poisson_pmf(lam, i)

    return pmf, cdf