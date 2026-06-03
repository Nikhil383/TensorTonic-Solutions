import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g)
    norm = np.linalg.norm(g)

    # No clipping needed
    if norm == 0 or max_norm <= 0 or norm <= max_norm:
        return g.copy()

    # Clip gradients
    return g * (max_norm / norm)