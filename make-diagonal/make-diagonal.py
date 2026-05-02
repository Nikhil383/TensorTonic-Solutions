import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    v=np.array(v)
    n = len(v)
    mat = np.zeros((n, n))

    for i in range(n):
        mat[i, i] = v[i]

    return mat
