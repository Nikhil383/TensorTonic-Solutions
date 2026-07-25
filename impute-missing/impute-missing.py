import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.array(X, dtype=float, copy=True)

    if X.ndim == 1:
        if strategy == "mean":
            stat = np.nanmean(X)
        elif strategy == "median":
            stat = np.nanmedian(X)
        else:
            raise ValueError("strategy must be 'mean' or 'median'")

        if np.isnan(stat):
            stat = 0

        X[np.isnan(X)] = stat
        return X

    if strategy == "mean":
        stats = np.nanmean(X, axis=0)
    elif strategy == "median":
        stats = np.nanmedian(X, axis=0)
    else:
        raise ValueError("strategy must be 'mean' or 'median'")

    stats = np.where(np.isnan(stats), 0, stats)

    mask = np.isnan(X)
    X[mask] = stats[np.where(mask)[1]]

    return X