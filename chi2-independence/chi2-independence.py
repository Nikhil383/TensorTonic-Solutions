import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C=np.array(C,dtype=float)
    row_i=np.sum(C,axis=1)
    col_j=np.sum(C,axis=0)
    total=np.sum(C)
    expected=np.outer(row_i,col_j)/total
    chi_2=np.sum((C - expected) ** 2 / expected)
    return chi_2,expected