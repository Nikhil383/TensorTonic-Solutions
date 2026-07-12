def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    n = len(series)

    if n == 0:
        return []

    # Initialize level and trend
    level = series[0]
    trend = series[1] - series[0] if n > 1 else 0

    levels = [level]

    for i in range(1, n):
        prev_level = level

        # Update level
        level = alpha * series[i] + (1 - alpha) * (level + trend)

        # Update trend
        trend = beta * (level - prev_level) + (1 - beta) * trend

        levels.append(level)

    return levels