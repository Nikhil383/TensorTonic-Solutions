import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    dh=np.array(dh,dtype=float)
    x_t, h_prev, h_t, W, U, b = cache
    x_t=np.array(x_t,dtype=float)
    h_prev=np.array(h_prev,dtype=float)
    h_t=np.array(h_t,dtype=float)
    W=np.array(W,dtype=float)
    U=np.array(U,dtype=float)
    
    dz=dh*(1-h_t**2)

    dx_t = W.T @ dz
    dh_prev = U.T @ dz
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz

    return dx_t, dh_prev, dW, dU, db
    
