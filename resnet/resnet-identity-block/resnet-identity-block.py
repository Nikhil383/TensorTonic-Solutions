import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)

    z1 = x @ W1.T
    a1 = np.maximum(0, z1)

    z2 = a1 @ W2.T
    y = np.maximum(0, z2 + x)

    return y.tolist()