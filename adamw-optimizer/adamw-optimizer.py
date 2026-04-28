import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    w = np.array(w, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)

    # Store previous weights
    w_prev = w.copy()

    # Update moments (NO bias correction)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Apply update exactly as formula
    w = w_prev - lr * (weight_decay * w_prev) \
              - lr * (m / (np.sqrt(v) + eps))

    return w, m, v

    return w, m, v
    
    