def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    mapping={ordering[i]: i for i in range(len(ordering))}
    return [mapping[v] for v in values]