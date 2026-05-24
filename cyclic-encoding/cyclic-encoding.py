def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    result=[]
    for v in values:
        angle=2 * math.pi * v / period
        result.append([math.sin(angle), math.cos(angle)])
    return result