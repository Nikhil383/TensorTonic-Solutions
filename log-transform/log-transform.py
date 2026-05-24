import math
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    logs=[math.log1p(v) for v in values]
    return logs