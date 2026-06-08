import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)

    # Create and shuffle indices
    indices = np.arange(n)

    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    # Yield batches
    for start in range(0, n, batch_size):
        end = start + batch_size

        if drop_last and end > n:
            break

        batch_idx = indices[start:end]

        yield X[batch_idx], y[batch_idx]