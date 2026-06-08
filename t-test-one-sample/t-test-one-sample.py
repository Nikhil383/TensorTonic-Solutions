import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x=np.array(x,dtype=float)
    n=len(x)
    sample_mean=np.mean(x)
    standard=np.sqrt(np.sum((x-sample_mean)**2)/(n-1))
    t=(sample_mean-mu0)/(standard/np.sqrt(n))
    return t