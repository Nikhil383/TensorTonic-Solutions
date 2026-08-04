import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    if k < 2:
        raise ValueError("k must be at least 2")

    if k > N:
        raise ValueError("k cannot be greater than N")

    indices = np.arange(N)

    # Shuffle indices
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    # Split indices into k folds
    folds = np.array_split(indices, k)

    result = []

    for i in range(k):
        val_idx = folds[i]

        train_idx = np.concatenate([
            folds[j]
            for j in range(k)
            if j != i
        ])

        result.append((train_idx, val_idx))

    return result
    