import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Write code here
    V = np.array(V, dtype=float)   # ensure numeric + avoid aliasing
    V_new = V.copy()               # no in-place modification

    # TD error
    delta = r + gamma * V[s_next] - V[s]

    # update only state s
    V_new[s] = V[s] + alpha * delta

    return V_new