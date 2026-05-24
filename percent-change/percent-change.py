def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    # Write code here
    result=[]
    for i in range(1,len(series)):
        try:
            a=(series[i]-series[i-1])/series[i-1]
            
        except ZeroDivisionError:
            a=0
        result.append(a)
    return result   