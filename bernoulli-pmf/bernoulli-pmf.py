import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here
    # Validate probability
    if p < 0 or p > 1:
        return None
    
    x = np.array(x)
    
    # PMF: P(X=1)=p, P(X=0)=1-p
    pmf = np.where(x == 1, p, 1 - p)
    
    # Moments
    mean = float(p)
    variance = float(p * (1 - p))
    
    return pmf, mean, variance
    