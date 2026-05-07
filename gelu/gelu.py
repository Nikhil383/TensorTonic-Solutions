import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x=np.array(x,dtype=float)
    erfs=np.vectorize(math.erf)(x / np.sqrt(2))
    gelu=(0.5*x)*(1+erfs)
    return gelu
