def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    # Compute Xavier limit
    limit = math.sqrt(6 / (fan_in + fan_out))

    # Scale each weight from [0,1] -> [-limit, limit]
    scaled_W = []

    for row in W:
        scaled_row = []
        for value in row:
            scaled = value * 2 * limit - limit
            scaled_row.append(round(scaled, 4))
        scaled_W.append(scaled_row)

    return scaled_W