import numpy as np

def min_max_scaling(data):
    data = np.array(data, dtype=float)

    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    ranges = max_vals - min_vals

    # Prevent division by zero
    ranges[ranges == 0] = 1

    scaled_data = (data - min_vals) / ranges

    return scaled_data.tolist()