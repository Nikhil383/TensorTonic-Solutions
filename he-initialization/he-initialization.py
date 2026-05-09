def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    limit = (6 / fan_in) ** 0.5

    scaled_weights = []

    for row in W:
        scaled_row = []

        for w in row:
            scaled = w * 2 * limit - limit
            scaled_row.append(scaled)

        scaled_weights.append(scaled_row)

    return scaled_weights