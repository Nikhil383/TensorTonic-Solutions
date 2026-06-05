def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    H = len(X)
    W = len(X[0])

    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1

    pooled = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_val = float('-inf')

            for r in range(pool_size):
                for c in range(pool_size):
                    value = X[i * stride + r][j * stride + c]
                    if value > max_val:
                        max_val = value

            row.append(max_val)

        pooled.append(row)

    return pooled