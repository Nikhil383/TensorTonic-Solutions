def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    n = len(y_true)

    # Initialize bin sums and counts
    count = [0] * n_bins
    sum_true = [0.0] * n_bins
    sum_pred = [0.0] * n_bins

    # Assign samples to bins
    for y, p in zip(y_true, y_pred):
        if p == 1.0:
            idx = n_bins - 1
        else:
            idx = int(p * n_bins)

        count[idx] += 1
        sum_true[idx] += y
        sum_pred[idx] += p

    # Compute ECE
    ece = 0.0

    for i in range(n_bins):
        if count[i] > 0:
            acc = sum_true[i] / count[i]
            conf = sum_pred[i] / count[i]
            ece += (count[i] / n) * abs(acc - conf)

    return ece
    