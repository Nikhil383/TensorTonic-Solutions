import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.

    Parameters:
        X : array-like
        y : array-like
        test_size : float
        rng : np.random.Generator or None

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)

    train_idx = []
    test_idx = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]

        # Shuffle indices for this class
        shuffled = idx.copy()
        if rng is not None:
            rng.shuffle(shuffled)
        else:
            np.random.shuffle(shuffled)

        n = len(shuffled)

        # Number of test samples
        n_test = int(round(n * test_size))

        # Ensure at least one train sample remains when possible
        if n > 1:
            n_test = max(1, min(n_test, n - 1))
        else:
            n_test = 0

        # Restore original ordering within each split
        test_idx.extend(sorted(shuffled[:n_test]))
        train_idx.extend(sorted(shuffled[n_test:]))

    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test