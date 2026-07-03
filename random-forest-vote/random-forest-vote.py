import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions=np.array(predictions)
    final_predictions = []

    for i in range(predictions.shape[1]):  # Loop over columns
        values, counts = np.unique(predictions[:, i], return_counts=True)
        final_predictions.append(values[np.argmax(counts)])

    return final_predictions