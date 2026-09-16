import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    result = x.copy().astype(float)
    for grad in gradients_F:
        result = result + np.dot(result, grad)
    return result

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    result = x.copy().astype(float)
    for grad in gradients_F:
        result = np.dot(result, grad)
    return result