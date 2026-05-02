import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    # Step 1: get unique classes and their counts
    values, counts = np.unique(y_train, return_counts=True)

    # Step 2: find majority class
    majority_class = values[np.argmax(counts)]

    # Step 3: create prediction array
    y_pred = np.full(len(X_test), majority_class)

    return y_pred