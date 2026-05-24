def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here
    k=len(weights)
    w_sum=sum(weights)
    output=[]
    for i in range(0,len(values)-k+1):
        sums=sum(weights[j]*values[i+j] for j in range(k))
        output.append(sums/w_sum)
    return output