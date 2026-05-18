def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    normalized_matrix = []

    for row in matrix:

        # Get only non-zero ratings
        rated_values = [v for v in row if v != 0]

        # If no ratings exist, keep row as zeros
        if not rated_values:
            normalized_matrix.append([0.0 for _ in row])
            continue

        # Compute mean of rated values
        mean_rating = sum(rated_values) / len(rated_values)

        # Normalize ratings
        normalized_row = [
            v - mean_rating if v != 0 else 0.0
            for v in row
        ]

        normalized_matrix.append(normalized_row)

    return normalized_matrix