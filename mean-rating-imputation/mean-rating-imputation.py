def mean_rating_imputation(ratings_matrix, mode):
    # Make a copy so the original is unchanged
    result = [row[:] for row in ratings_matrix]

    if mode == "user":
        for i in range(len(result)):
            non_zero = [x for x in result[i] if x != 0]
            mean = sum(non_zero) / len(non_zero) if non_zero else 0

            for j in range(len(result[i])):
                if result[i][j] == 0:
                    result[i][j] = mean

    elif mode == "item":
        rows = len(result)
        cols = len(result[0])

        for j in range(cols):
            non_zero = [result[i][j] for i in range(rows) if result[i][j] != 0]
            mean = sum(non_zero) / len(non_zero) if non_zero else 0

            for i in range(rows):
                if result[i][j] == 0:
                    result[i][j] = mean

    return result