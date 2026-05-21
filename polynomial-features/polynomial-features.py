def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    lc=[[x**i for i in range(degree+1)] for x in values]
    return lc