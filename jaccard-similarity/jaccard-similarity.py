def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    if len(set_a) and len(set_b)==0:
        return 0.0
    set_a=set(set_a)
    set_b=set(set_b)
    try:
        js=len(set_a & set_b)/len(set_a | set_b)
        return js
    except ZeroDivisionError:
        return 0.0
    
    return js