import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    # Validate dimensions
    if x.ndim != 4:
        raise ValueError("x must have shape (N, C_in, H, W)")

    if W.ndim != 4:
        raise ValueError("W must have shape (C_out, C_in, K_h, K_w)")

    if b.ndim != 1:
        raise ValueError("b must have shape (C_out,)")

    N, C_in, H, W_in = x.shape
    C_out, Cw, K_h, K_w = W.shape

    if C_in != Cw:
        raise ValueError("Input channels must match filter channels")

    if b.shape[0] != C_out:
        raise ValueError("Bias size must match C_out")

    # Output dimensions
    H_out = H - K_h + 1
    W_out = W_in - K_w + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError("Kernel size cannot be larger than input size")

    # Output tensor
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)

    # Convolution
    for i in range(H_out):
        for j in range(W_out):

            # (N, C_in, K_h, K_w)
            patch = x[:, :, i:i+K_h, j:j+K_w]

            # Vectorized multiply + sum
            # Result shape: (N, C_out)
            out[:, :, i, j] = np.tensordot(
                patch,
                W,
                axes=([1, 2, 3], [1, 2, 3])
            ) + b

    return out