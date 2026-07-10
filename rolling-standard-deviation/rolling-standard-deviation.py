def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    result = []

    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]

        # Compute mean
        mean = sum(window) / window_size

        # Compute population variance
        variance = sum((x - mean) ** 2 for x in window) / window_size

        # Compute standard deviation
        std = math.sqrt(variance)

        result.append(std)

    return result