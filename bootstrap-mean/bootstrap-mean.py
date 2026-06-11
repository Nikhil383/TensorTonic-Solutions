import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.asarray(x)
    n = len(x)

    boot_means = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        if rng is not None:
            indices = rng.integers(0, n, size=n)
        else:
            indices = np.random.randint(0, n, size=n)

        sample = x[indices]
        boot_means[b] = np.mean(sample)

    alpha = (1 - ci) / 2

    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1 - alpha)

    return boot_means, lower, upper
