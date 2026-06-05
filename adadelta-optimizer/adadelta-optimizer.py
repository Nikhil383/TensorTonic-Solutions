import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Write code here
    try:
        w = np.asarray(w, dtype=float)
        grad = np.asarray(grad, dtype=float)
        E_grad_sq = np.asarray(E_grad_sq, dtype=float)
        E_update_sq = np.asarray(E_update_sq, dtype=float)

        # Step 1: Update squared gradient average
        E_grad_sq = rho * E_grad_sq + (1 - rho) * (grad ** 2)

        # Step 2: Compute parameter update
        delta_w = -(
            np.sqrt(E_update_sq + eps)
            / np.sqrt(E_grad_sq + eps)
        ) * grad

        # Step 3: Update squared update average
        E_update_sq = rho * E_update_sq + (1 - rho) * (delta_w ** 2)

        # Step 4: Update parameters
        w = w + delta_w

        return w, E_grad_sq, E_update_sq

    except:
        return None