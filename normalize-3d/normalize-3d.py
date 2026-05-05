import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.array(v)

    # Compute norm
    norm = np.linalg.norm(v, axis=-1, keepdims=True)

    # Avoid division by zero
    norm = np.where(norm == 0, 1, norm)

    return v / norm