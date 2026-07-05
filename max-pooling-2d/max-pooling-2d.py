import numpy as np
def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """

    H = len(X)
    W = len(X[0])

    output = []

    for i in range(0, H, pool_size):
        row = []
        for j in range(0, W, pool_size):
            max_val = float("-inf")

            for r in range(i, i + pool_size):
                for c in range(j, j + pool_size):
                    max_val = max(max_val, X[r][c])

            row.append(max_val)

        output.append(row)
    return output