import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    Returns a 2D list:
        rows    -> test samples
        columns -> classes
    """

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    # Handle a single test sample
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    classes = np.unique(y_train)

    result = []

    for x in X_test:

        class_scores = []

        for c in classes:

            # Select samples belonging to class c
            X_c = X_train[y_train == c]

            # Prior P(y=c)
            prior = len(X_c) / len(X_train)

            # Bernoulli probability with Laplace smoothing
            prob = (X_c.sum(axis=0) + 1) / (len(X_c) + 2)

            # log P(x | y=c)
            log_likelihood = np.sum(
                x * np.log(prob) +
                (1 - x) * np.log(1 - prob)
            )

            # log P(y=c) + log P(x | y=c)
            log_posterior = np.log(prior) + log_likelihood

            class_scores.append(float(log_posterior))

        result.append(class_scores)

    return result