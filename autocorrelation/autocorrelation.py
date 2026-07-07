import numpy as np

def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.

    Parameters:
        series (list or np.ndarray): Input time series.
        max_lag (int): Maximum lag to compute.

    Returns:
        list: Autocorrelation values from lag 0 to max_lag.
    """
    series = np.array(series, dtype=float)
    n = len(series)

    # Mean of the series
    mean = np.mean(series)

    # Total variance (γ0)
    gamma0 = np.sum((series - mean) ** 2)

    # Handle constant series
    if gamma0 == 0:
        return [1.0] + [0.0] * max_lag

    autocorr = []

    for k in range(max_lag + 1):
        autocov = 0

        for t in range(n - k):
            autocov += (series[t] - mean) * (series[t + k] - mean)

        autocorr.append(autocov / gamma0)

    return autocorr