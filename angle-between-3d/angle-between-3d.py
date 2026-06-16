import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v=np.array(v,dtype=float)
    w=np.array(w,dtype=float)
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)

    # Return NaN if either vector is effectively zero
    if norm_v < 1e-10 or norm_w < 1e-10:
        return np.nan

    cos_theta = np.dot(v, w) / (norm_v * norm_w)

    # Protect against floating-point roundoff
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)