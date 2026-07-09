import numpy as np

def user_based_cf_prediction(similarities, ratings):
    similarities = np.array(similarities)
    ratings = np.array(ratings)

    mask = similarities > 0
    similarities = similarities[mask]
    ratings = ratings[mask]

    if len(similarities) == 0:
        return 0.0

    return np.sum(similarities * ratings) / np.sum(similarities)