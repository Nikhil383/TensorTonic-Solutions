def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    ema=[values[0]]
    last=len(values)
    for i in range(1,last):
        new_ema= alpha * values[i] + (1 - alpha) * ema[-1]
        ema.append(new_ema)
    return ema