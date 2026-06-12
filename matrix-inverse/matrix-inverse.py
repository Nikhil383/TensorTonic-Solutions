import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here

    A = np.array(A, dtype=float)

    # Check if A is a 2D square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    # Check if A is singular
    if abs(np.linalg.det(A)) < 1e-10:
        return None

    # Compute and return the inverse
    return np.linalg.inv(A)
