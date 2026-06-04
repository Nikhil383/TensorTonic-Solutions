import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix, dtype=float)

        # Must be 2D and square
        if matrix.ndim != 2:
            return None

        if matrix.shape[0] != matrix.shape[1]:
            return None

        if matrix.size == 0:
            return None

        eigenvalues = np.linalg.eigvals(matrix)

        # Sort by real part, then imaginary part
        idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
        eigenvalues = eigenvalues[idx]

        return eigenvalues

    except:
        return None