import math

def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.

    Parameters:
        ratings_matrix: List of lists where
                        rows = users
                        columns = items
                        Missing ratings are represented by 0.
        item_i: Index of first item
        item_j: Index of second item

    Returns:
        Similarity score
    """

    numerator = 0
    denominator_i = 0
    denominator_j = 0

    for user in ratings_matrix:

        # Compute user's average rating (ignoring zeros)
        rated_items = [rating for rating in user if rating != 0]

        if len(rated_items) == 0:
            continue

        user_avg = sum(rated_items) / len(rated_items)

        rating_i = user[item_i]
        rating_j = user[item_j]

        # Only consider users who rated both items
        if rating_i != 0 and rating_j != 0:

            adj_i = rating_i - user_avg
            adj_j = rating_j - user_avg

            numerator += adj_i * adj_j
            denominator_i += adj_i ** 2
            denominator_j += adj_j ** 2

    if denominator_i == 0 or denominator_j == 0:
        return 0

    return numerator / math.sqrt(denominator_i * denominator_j)